import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Football Match Predictor")
st.write(
    "Приложение создано на основе ноутбука `final_ver3.ipynb` и датасета `England CSV.csv`. "
    "Модель прогнозирует вероятность победы команды с учетом места матча, соперника "
    "и rolling averages за последние 3 игры. Названия команд нормализуются, чтобы не было дублей вроде Brighton / Brighton & Hove Albion."
)


REQUIRED_COLUMNS = [
    "Date",
    "Season",
    "HomeTeam",
    "AwayTeam",
    "FTH Goals",
    "FTA Goals",
    "FT Result",
    "H Shots",
    "A Shots",
    "H SOT",
    "A SOT",
    "H Fouls",
    "A Fouls",
    "H Corners",
    "A Corners",
    "H Yellow",
    "A Yellow",
    "H Red",
    "A Red",
]

BASE_COLS = ["GF", "GA", "SH", "SOT", "Fouls", "Yellow", "Red"]
ROLLING_COLS = [f"{col}_rolling" for col in BASE_COLS]
BASE_PREDICTORS = ["venue_code", "opp_code"]
FINAL_PREDICTORS = BASE_PREDICTORS + ROLLING_COLS
DEFAULT_DATA_PATH = "England CSV.csv"

TEAM_NAME_MAP = {
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Ipswich Town": "Ipswich",
}


def normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["HomeTeam", "AwayTeam"]:
        df[col] = df[col].replace(TEAM_NAME_MAP)
    return df


@st.cache_data
def load_default_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def check_required_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def prepare_matches(df: pd.DataFrame, start_season: str, end_season: str) -> tuple[pd.DataFrame, dict, dict]:
    df = df.copy()
    df = normalize_team_names(df)

    df = df[df["Season"].between(start_season, end_season)].copy()

    home = df.copy()
    home["Venue"] = "Home"
    home["Team"] = df["HomeTeam"]
    home["Opponent"] = df["AwayTeam"]
    home["GF"] = df["FTH Goals"]
    home["GA"] = df["FTA Goals"]
    home["Result"] = df["FT Result"].map({"H": "Win", "D": "Draw", "A": "Loss"})
    home["SH"] = df["H Shots"]
    home["SOT"] = df["H SOT"]
    home["Corners"] = df["H Corners"]
    home["Fouls"] = df["H Fouls"]
    home["Yellow"] = df["H Yellow"]
    home["Red"] = df["H Red"]

    away = df.copy()
    away["Venue"] = "Away"
    away["Team"] = df["AwayTeam"]
    away["Opponent"] = df["HomeTeam"]
    away["GF"] = df["FTA Goals"]
    away["GA"] = df["FTH Goals"]
    away["Result"] = df["FT Result"].map({"H": "Loss", "D": "Draw", "A": "Win"})
    away["SH"] = df["A Shots"]
    away["SOT"] = df["A SOT"]
    away["Corners"] = df["A Corners"]
    away["Fouls"] = df["A Fouls"]
    away["Yellow"] = df["A Yellow"]
    away["Red"] = df["A Red"]

    matches = pd.concat([home, away], ignore_index=True)

    matches = matches[
        [
            "Date",
            "Season",
            "Team",
            "Venue",
            "Opponent",
            "GF",
            "GA",
            "Result",
            "SH",
            "SOT",
            "Corners",
            "Fouls",
            "Yellow",
            "Red",
        ]
    ].copy()

    matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True, errors="coerce")
    matches = matches.dropna(subset=["Date", "Team", "Opponent", "Result"])
    matches = matches.sort_values("Date").reset_index(drop=True)

    matches["venue_code"] = matches["Venue"].astype("category").cat.codes
    matches["opp_code"] = matches["Opponent"].astype("category").cat.codes
    matches["target"] = (matches["Result"] == "Win").astype(int)

    venue_categories = matches["Venue"].astype("category").cat.categories
    opponent_categories = matches["Opponent"].astype("category").cat.categories

    venue_map = {name: code for code, name in enumerate(venue_categories)}
    opponent_map = {name: code for code, name in enumerate(opponent_categories)}

    return matches, venue_map, opponent_map


def rolling_averages(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("Date").copy()
    rolling_stats = group[BASE_COLS].rolling(3, closed="left").mean()
    group[ROLLING_COLS] = rolling_stats
    group = group.dropna(subset=ROLLING_COLS)
    return group


def add_rolling_features(matches: pd.DataFrame) -> pd.DataFrame:
    matches_rolling = matches.groupby("Team", group_keys=False).apply(rolling_averages)
    return matches_rolling.reset_index(drop=True)


def train_models(matches_rolling: pd.DataFrame, split_date: pd.Timestamp):
    train = matches_rolling[matches_rolling["Date"] < split_date].copy()
    test = matches_rolling[matches_rolling["Date"] >= split_date].copy()

    X_train = train[FINAL_PREDICTORS]
    y_train = train["target"]

    X_test = test[FINAL_PREDICTORS]
    y_test = test["target"]

    rf_model = RandomForestClassifier(
        n_estimators=50,
        min_samples_split=10,
        random_state=1,
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    results = pd.DataFrame(
        {
            "Model": ["Random Forest", "Logistic Regression"],
            "Accuracy": [
                accuracy_score(y_test, rf_preds),
                accuracy_score(y_test, lr_preds),
            ],
            "Precision": [
                precision_score(y_test, rf_preds, zero_division=0),
                precision_score(y_test, lr_preds, zero_division=0),
            ],
            "Recall": [
                recall_score(y_test, rf_preds, zero_division=0),
                recall_score(y_test, lr_preds, zero_division=0),
            ],
            "F1": [
                f1_score(y_test, rf_preds, zero_division=0),
                f1_score(y_test, lr_preds, zero_division=0),
            ],
        }
    )

    predictions = {
        "Random Forest": rf_preds,
        "Logistic Regression": lr_preds,
    }

    models = {
        "Random Forest": rf_model,
        "Logistic Regression": lr_model,
    }

    return models, results, predictions, train, test, y_test


def get_latest_rolling_values(matches: pd.DataFrame, team: str) -> dict | None:
    team_matches = matches[matches["Team"] == team].sort_values("Date")

    if len(team_matches) < 3:
        return None

    last_3_matches = team_matches.tail(3)
    values = last_3_matches[BASE_COLS].mean()

    return {f"{col}_rolling": values[col] for col in BASE_COLS}


def create_prediction_row(
    matches: pd.DataFrame,
    venue_map: dict,
    opponent_map: dict,
    team: str,
    opponent: str,
    venue: str,
) -> pd.DataFrame | None:
    rolling_values = get_latest_rolling_values(matches, team)

    if rolling_values is None:
        return None

    if venue not in venue_map or opponent not in opponent_map:
        return None

    row = {
        "venue_code": venue_map[venue],
        "opp_code": opponent_map[opponent],
    }
    row.update(rolling_values)

    return pd.DataFrame([row])[FINAL_PREDICTORS]


with st.sidebar:
    st.header("Настройки данных")

    uploaded_file = st.file_uploader(
        "Можно загрузить свой CSV",
        type=["csv"],
        help="Если рядом с app.py лежит England CSV.csv, приложение загрузит его автоматически.",
    )

    st.header("Настройки модели")
    split_date_value = st.date_input("Train/Test split date", value=pd.to_datetime("2022-01-01"))
    model_choice = st.selectbox("Модель для прогноза", ["Random Forest", "Logistic Regression"])


if uploaded_file is not None:
    raw_df = load_uploaded_csv(uploaded_file)
elif os.path.exists(DEFAULT_DATA_PATH):
    raw_df = load_default_csv(DEFAULT_DATA_PATH)
else:
    st.info("Загрузи CSV-файл или положи `England CSV.csv` в одну папку с `app.py`.")
    st.stop()


missing_columns = check_required_columns(raw_df)

if missing_columns:
    st.error("В CSV не хватает нужных колонок:")
    st.write(missing_columns)
    st.stop()


seasons = sorted(raw_df["Season"].dropna().unique())

with st.sidebar:
    start_season = st.selectbox(
        "Начальный сезон",
        seasons,
        index=seasons.index("2014/15") if "2014/15" in seasons else 0,
    )
    end_season = st.selectbox(
        "Последний сезон",
        seasons,
        index=seasons.index("2024/25") if "2024/25" in seasons else len(seasons) - 1,
    )


matches, venue_map, opponent_map = prepare_matches(raw_df, start_season, end_season)
matches_rolling = add_rolling_features(matches)

if matches_rolling.empty:
    st.error("После rolling averages данных не осталось. Выбери больше сезонов.")
    st.stop()

split_date = pd.to_datetime(split_date_value)
train_check = matches_rolling[matches_rolling["Date"] < split_date]
test_check = matches_rolling[matches_rolling["Date"] >= split_date]

if train_check.empty or test_check.empty:
    st.error("Train или test выборка пустая. Измени дату разделения.")
    st.stop()

models, results, predictions, train, test, y_test = train_models(matches_rolling, split_date)
selected_model = models[model_choice]
selected_preds = predictions[model_choice]


st.subheader("1. Информация о данных")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Строк в CSV", len(raw_df))
col2.metric("После Home/Away", len(matches))
col3.metric("После rolling", len(matches_rolling))
col4.metric("Команд", matches["Team"].nunique())

with st.expander("Посмотреть подготовленные данные"):
    st.dataframe(matches_rolling.head(50), use_container_width=True)


st.subheader("2. Анализ данных")

eda_col1, eda_col2 = st.columns(2)

with eda_col1:
    st.write("Распределение результатов")
    result_counts = matches["Result"].value_counts()
    st.bar_chart(result_counts)

with eda_col2:
    st.write("Средние голы по сезонам")
    goals_by_season = matches.groupby("Season")[["GF", "GA"]].mean()
    st.line_chart(goals_by_season)


st.subheader("3. Обучение и оценка модели")

st.write("Сравнение моделей")
st.dataframe(results.round(3), use_container_width=True)

selected_result = results[results["Model"] == model_choice].iloc[0]

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Accuracy", round(selected_result["Accuracy"], 3))
metric_col2.metric("Precision", round(selected_result["Precision"], 3))
metric_col3.metric("Recall", round(selected_result["Recall"], 3))
metric_col4.metric("F1", round(selected_result["F1"], 3))

st.write(f"Confusion Matrix: {model_choice}")
cm = confusion_matrix(y_test, selected_preds)
cm_df = pd.DataFrame(
    cm,
    index=["Actual 0: no win", "Actual 1: win"],
    columns=["Predicted 0", "Predicted 1"],
)
st.dataframe(cm_df, use_container_width=True)


st.subheader("4. Прогноз матча")

teams = sorted(matches["Team"].unique())

home_col, away_col = st.columns(2)

with home_col:
    home_team = st.selectbox("Home Team", teams)

with away_col:
    away_options = [team for team in teams if team != home_team]
    away_team = st.selectbox("Away Team", away_options)

if st.button("Predict"):
    home_row = create_prediction_row(matches, venue_map, opponent_map, home_team, away_team, "Home")
    away_row = create_prediction_row(matches, venue_map, opponent_map, away_team, home_team, "Away")

    if home_row is None or away_row is None:
        st.error("Недостаточно данных для прогноза. Нужно минимум 3 прошлых матча для каждой команды.")
    else:
        home_win_proba = selected_model.predict_proba(home_row)[0][1]
        away_win_proba = selected_model.predict_proba(away_row)[0][1]

        result_col1, result_col2 = st.columns(2)
        result_col1.metric(f"Победа {home_team}", f"{home_win_proba:.1%}")
        result_col2.metric(f"Победа {away_team}", f"{away_win_proba:.1%}")

        if home_win_proba > away_win_proba:
            st.success(f"Модель больше склоняется к победе: {home_team}")
        elif away_win_proba > home_win_proba:
            st.success(f"Модель больше склоняется к победе: {away_team}")
        else:
            st.warning("Модель не видит явного фаворита.")

        st.caption(
            "Важно: модель обучена как бинарная классификация `win / not win`. "
            "Поэтому вероятность ничьей отдельно не считается."
        )

        input_data = pd.concat([home_row, away_row], ignore_index=True)
        input_data.insert(0, "Team", [home_team, away_team])
        input_data.insert(1, "Venue", ["Home", "Away"])

        with st.expander("Данные, которые получила модель"):
            st.dataframe(input_data, use_container_width=True)


st.subheader("5. Вывод")
st.write(
    "Random Forest с rolling averages повторяет финальную идею из ноутбука: "
    "используются `venue_code`, `opp_code` и средние показатели команды за последние 3 матча. "
    "Модель можно использовать как учебный проект для портфолио, но для реального спортивного прогноза "
    "нужно добавить больше факторов: составы, травмы, xG, силу соперника, домашнюю форму и коэффициенты букмекеров."
)
