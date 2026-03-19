import os
import sys
import json
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from poisson_model import load_data, calculate_team_ratings, predict_match

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

CSV_FILES = [
    'E0_2021.csv', 'E0_2122.csv', 'E0_2223.csv',
    'E0_2324.csv', 'E0_2425.csv', 'E0_2526.csv',
]


def load_full_data():
    dfs = []
    for f in CSV_FILES:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            df = pd.read_csv(path, encoding='latin1')
            dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data = data.dropna(subset=['FTHG', 'FTAG', 'HST', 'AST', 'FTR'])
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data = data.sort_values('Date').reset_index(drop=True)
    data['xg_home'] = data['HST'] * 0.35
    data['xg_away'] = data['AST'] * 0.35
    return data


def get_recent_form(data, team, date, n=5, venue='all'):
    if venue == 'home':
        games = data[(data['HomeTeam'] == team) & (data['Date'] < date)]
        results = games.tail(n)['FTR'].map({'H': 1, 'D': 0.5, 'A': 0}).values
    elif venue == 'away':
        games = data[(data['AwayTeam'] == team) & (data['Date'] < date)]
        results = games.tail(n)['FTR'].map({'A': 1, 'D': 0.5, 'H': 0}).values
    else:
        home = data[(data['HomeTeam'] == team) & (data['Date'] < date)].copy()
        home['pts'] = home['FTR'].map({'H': 1, 'D': 0.5, 'A': 0})
        away = data[(data['AwayTeam'] == team) & (data['Date'] < date)].copy()
        away['pts'] = away['FTR'].map({'A': 1, 'D': 0.5, 'H': 0})
        combined = pd.concat([home[['Date', 'pts']], away[['Date', 'pts']]])
        combined = combined.sort_values('Date').tail(n)
        results = combined['pts'].values

    if len(results) == 0:
        return 0.5
    return round(float(np.mean(results)), 3)


def get_rest_days(data, team, date):
    home = data[(data['HomeTeam'] == team) & (data['Date'] < date)]
    away = data[(data['AwayTeam'] == team) & (data['Date'] < date)]
    all_games = pd.concat([home[['Date']], away[['Date']]])
    if len(all_games) == 0:
        return 7
    last_game = all_games['Date'].max()
    return min((date - last_game).days, 21)


def convert_odds_to_prob(home_odds, draw_odds, away_odds):
    try:
        h = 1 / float(home_odds)
        d = 1 / float(draw_odds)
        a = 1 / float(away_odds)
        total = h + d + a
        return round(h / total, 4), round(d / total, 4), round(a / total, 4)
    except Exception:
        return 0.45, 0.27, 0.28


def build_features(data, ratings, avg_home, avg_away):
    features = []
    labels = []

    result_map = {'H': 0, 'D': 1, 'A': 2}

    for idx, row in data.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        date = row['Date']

        if home not in ratings or away not in ratings:
            continue

        try:
            poisson_pred = predict_match(home, away, ratings, avg_home, avg_away)
        except Exception:
            continue

        home_form = get_recent_form(data, home, date, n=5)
        away_form = get_recent_form(data, away, date, n=5)
        home_rest = get_rest_days(data, home, date)
        away_rest = get_rest_days(data, away, date)

        odds_home_prob, odds_draw_prob, odds_away_prob = 0.45, 0.27, 0.28
        if all(col in row for col in ['B365H', 'B365D', 'B365A']):
            try:
                odds_home_prob, odds_draw_prob, odds_away_prob = convert_odds_to_prob(
                    row['B365H'], row['B365D'], row['B365A']
                )
            except Exception:
                pass

        feature_row = [
            poisson_pred['home_win_pct'] / 100,
            poisson_pred['draw_pct'] / 100,
            poisson_pred['away_win_pct'] / 100,
            poisson_pred['xg_home'],
            poisson_pred['xg_away'],
            home_form,
            away_form,
            home_rest,
            away_rest,
            odds_home_prob,
            odds_draw_prob,
            odds_away_prob,
        ]

        features.append(feature_row)
        labels.append(result_map[row['FTR']])

    return np.array(features), np.array(labels)


def train_catboost(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        random_seed=42,
        verbose=100,
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    y_pred_proba = model.predict_proba(X_test)
    ll = log_loss(y_test, y_pred_proba)
    home_brier = brier_score_loss(
        (y_test == 0).astype(int), y_pred_proba[:, 0]
    )

    print(f"\nModel performance on test set:")
    print(f"  Log loss: {ll:.4f}")
    print(f"  Home win Brier score: {home_brier:.4f}")

    return model


def save_model(model):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'catboost_model.cbm')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")


def load_model():
    model_path = os.path.join(MODELS_DIR, 'catboost_model.cbm')
    if not os.path.exists(model_path):
        return None
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


def predict_with_catboost(poisson_pred, home_form, away_form,
                           home_rest, away_rest,
                           odds_home_prob=0.45, odds_draw_prob=0.27,
                           odds_away_prob=0.28):
    model = load_model()
    if model is None:
        return poisson_pred

    features = np.array([[
        poisson_pred['home_win_pct'] / 100,
        poisson_pred['draw_pct'] / 100,
        poisson_pred['away_win_pct'] / 100,
        poisson_pred['xg_home'],
        poisson_pred['xg_away'],
        home_form,
        away_form,
        home_rest,
        away_rest,
        odds_home_prob,
        odds_draw_prob,
        odds_away_prob,
    ]])

    proba = model.predict_proba(features)[0]

    poisson_pred['home_win_pct'] = round(float(proba[0]) * 100, 1)
    poisson_pred['draw_pct'] = round(float(proba[1]) * 100, 1)
    poisson_pred['away_win_pct'] = round(float(proba[2]) * 100, 1)
    poisson_pred['catboost_refined'] = True

    return poisson_pred


if __name__ == '__main__':
    print("--- The Gaffer: Training CatBoost Model ---\n")

    data = load_full_data()
    print(f"Loaded {len(data)} matches for training\n")

    ratings, avg_home, avg_away = calculate_team_ratings(data)
    print(f"Ratings calculated for {len(ratings)} teams\n")

    print("Building feature matrix...")
    print("This may take a few minutes...\n")
    features, labels = build_features(data, ratings, avg_home, avg_away)
    print(f"Feature matrix: {features.shape[0]} samples, {features.shape[1]} features\n")

    print("Training CatBoost model...")
    model = train_catboost(features, labels)

    save_model(model)

    print("\n--- Test: CatBoost refined prediction ---")
    test_poisson = predict_match('Arsenal', 'Man City', ratings, avg_home, avg_away)
    print(f"Poisson only:   Arsenal {test_poisson['home_win_pct']}% | Draw {test_poisson['draw_pct']}% | Man City {test_poisson['away_win_pct']}%")

    refined = predict_with_catboost(
        test_poisson.copy(),
        home_form=0.7,
        away_form=0.4,
        home_rest=7,
        away_rest=4,
    )
    print(f"CatBoost refined: Arsenal {refined['home_win_pct']}% | Draw {refined['draw_pct']}% | Man City {refined['away_win_pct']}%")