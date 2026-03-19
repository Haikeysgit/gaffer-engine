import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import poisson
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

CSV_FILES = [
    'E0_2021.csv', 'E0_2122.csv', 'E0_2223.csv',
    'E0_2324.csv', 'E0_2425.csv', 'E0_2526.csv',
]

HOME_ADVANTAGE = 1.2
XG_WEIGHT = 0.6
GOALS_WEIGHT = 0.4


def load_data():
    dfs = []
    for f in CSV_FILES:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            df = pd.read_csv(path, encoding='latin1')
            dfs.append(df)
        else:
            print(f"Warning: {f} not found, skipping")
    data = pd.concat(dfs, ignore_index=True)
    data = data.dropna(subset=['FTHG', 'FTAG', 'HST', 'AST'])
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data = data.sort_values('Date').reset_index(drop=True)
    data['xg_home'] = data['HST'] * 0.35
    data['xg_away'] = data['AST'] * 0.35
    print(f"Loaded {len(data)} matches across all seasons")
    return data


def time_decay_weights(dates, xi=0.0065):
    latest = dates.max()
    days_ago = (latest - dates).dt.days
    return np.exp(-xi * days_ago)


def calculate_team_ratings(data):
    data = data.copy()
    data['weight'] = time_decay_weights(data['Date'])

    avg_home_goals = np.average(data['FTHG'], weights=data['weight'])
    avg_away_goals = np.average(data['FTAG'], weights=data['weight'])
    avg_home_xg = np.average(data['xg_home'], weights=data['weight'])
    avg_away_xg = np.average(data['xg_away'], weights=data['weight'])

    teams = sorted(
        set(data['HomeTeam'].unique()) | set(data['AwayTeam'].unique())
    )

    ratings = {}
    for team in teams:
        home_games = data[data['HomeTeam'] == team]
        away_games = data[data['AwayTeam'] == team]

        if len(home_games) > 0:
            w = home_games['weight']
            goal_home_att = np.average(home_games['FTHG'], weights=w) / avg_home_goals
            goal_home_def = np.average(home_games['FTAG'], weights=w) / avg_away_goals
            xg_home_att = np.average(home_games['xg_home'], weights=w) / avg_home_xg
            xg_home_def = np.average(home_games['xg_away'], weights=w) / avg_away_xg
            home_attack = (XG_WEIGHT * xg_home_att) + (GOALS_WEIGHT * goal_home_att)
            home_defence = (XG_WEIGHT * xg_home_def) + (GOALS_WEIGHT * goal_home_def)
        else:
            home_attack = 1.0
            home_defence = 1.0

        if len(away_games) > 0:
            w = away_games['weight']
            goal_away_att = np.average(away_games['FTAG'], weights=w) / avg_away_goals
            goal_away_def = np.average(away_games['FTHG'], weights=w) / avg_home_goals
            xg_away_att = np.average(away_games['xg_away'], weights=w) / avg_away_xg
            xg_away_def = np.average(away_games['xg_home'], weights=w) / avg_home_xg
            away_attack = (XG_WEIGHT * xg_away_att) + (GOALS_WEIGHT * goal_away_att)
            away_defence = (XG_WEIGHT * xg_away_def) + (GOALS_WEIGHT * goal_away_def)
        else:
            away_attack = 1.0
            away_defence = 1.0

        ratings[team] = {
            'home_attack': round(home_attack, 4),
            'home_defence': round(home_defence, 4),
            'away_attack': round(away_attack, 4),
            'away_defence': round(away_defence, 4),
        }

    return ratings, avg_home_goals, avg_away_goals


def predict_match(home_team, away_team, ratings, avg_home_goals, avg_away_goals, max_goals=5):
    if home_team not in ratings:
        raise ValueError(f"Team not found in ratings: {home_team}")
    if away_team not in ratings:
        raise ValueError(f"Team not found in ratings: {away_team}")

    h = ratings[home_team]
    a = ratings[away_team]

    lambda_home = (
        h['home_attack'] * a['away_defence'] * avg_home_goals * HOME_ADVANTAGE
    )
    lambda_away = (
        a['away_attack'] * h['home_defence'] * avg_away_goals
    )

    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i][j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)

    matrix = matrix / matrix.sum()

    home_win = float(np.sum(np.tril(matrix, -1)))
    draw = float(np.sum(np.diag(matrix)))
    away_win = float(np.sum(np.triu(matrix, 1)))

    flat = [
        {'home': i, 'away': j, 'probability': round(matrix[i][j] * 100, 1)}
        for i in range(max_goals + 1)
        for j in range(max_goals + 1)
    ]
    flat.sort(key=lambda x: x['probability'], reverse=True)
    top_3 = flat[:3]

    return {
        'home_team': home_team,
        'away_team': away_team,
        'xg_home': round(lambda_home, 2),
        'xg_away': round(lambda_away, 2),
        'home_win_pct': round(home_win * 100, 1),
        'draw_pct': round(draw * 100, 1),
        'away_win_pct': round(away_win * 100, 1),
        'top_3': top_3,
        'scoreline_matrix': matrix.tolist(),
    }


if __name__ == '__main__':
    print("--- The Gaffer: Poisson Model (xG Enhanced) ---\n")

    data = load_data()
    ratings, avg_home, avg_away = calculate_team_ratings(data)
    print(f"Ratings calculated for {len(ratings)} teams\n")

    test_matches = [
        ('Arsenal', 'Man City'),
        ('Liverpool', 'Chelsea'),
        ('Newcastle', 'Sunderland'),
        ('Tottenham', "Nott'm Forest"),
    ]

    for home, away in test_matches:
        try:
            result = predict_match(home, away, ratings, avg_home, avg_away)
            print(f"{home} vs {away}")
            print(f"  xG: {result['xg_home']} — {result['xg_away']}")
            print(f"  Outcome: {home} {result['home_win_pct']}% | Draw {result['draw_pct']}% | {away} {result['away_win_pct']}%")
            print(f"  Top pick: {home} {result['top_3'][0]['home']}–{result['top_3'][0]['away']} {away} ({result['top_3'][0]['probability']}%)\n")
        except ValueError as e:
            print(f"  Skipped: {e}\n")