import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

CSV_FILES = [
    'E0_2021.csv', 'E0_2122.csv', 'E0_2223.csv',
    'E0_2324.csv', 'E0_2425.csv', 'E0_2526.csv',
]

TIME_DECAY_XI = 0.0065
XG_WEIGHT = 0.6
GOALS_WEIGHT = 0.4


def load_data():
    dfs = []
    for f in CSV_FILES:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            df = pd.read_csv(path, encoding='latin1')
            dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data = data.dropna(subset=['FTHG', 'FTAG', 'HST', 'AST'])
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data = data.sort_values('Date').reset_index(drop=True)
    data['xg_home'] = data['HST'] * 0.35
    data['xg_away'] = data['AST'] * 0.35
    return data


def time_decay_weights(dates, xi=TIME_DECAY_XI):
    latest = dates.max()
    days_ago = (latest - dates).dt.days
    return np.exp(-xi * days_ago)


def calculate_combined_ratings(data):
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


def save_combined_ratings(ratings):
    print("Saving combined ratings to Supabase...")
    for team_name, r in ratings.items():
        team_id = team_name.lower().replace(" ", "_").replace("'", "")
        data = {
            "team_id": team_id,
            "team_name": team_name,
            "home_attack": r["home_attack"],
            "home_defence": r["home_defence"],
            "away_attack": r["away_attack"],
            "away_defence": r["away_defence"],
        }
        supabase.table("team_ratings").upsert(data).execute()
    print(f"Saved combined ratings for {len(ratings)} teams")


if __name__ == "__main__":
    print("--- The Gaffer: Calculating Combined Ratings ---\n")
    data = load_data()
    print(f"Loaded {len(data)} matches\n")
    ratings, avg_home, avg_away = calculate_combined_ratings(data)

    print("Top 5 home attack ratings:")
    sorted_home = sorted(ratings.items(), key=lambda x: x[1]['home_attack'], reverse=True)[:5]
    for team, r in sorted_home:
        print(f"  {team}: {r['home_attack']}")

    print("\nTop 5 home defence ratings (lower = stronger defence):")
    sorted_def = sorted(ratings.items(), key=lambda x: x[1]['home_defence'])[:5]
    for team, r in sorted_def:
        print(f"  {team}: {r['home_defence']}")

    save_combined_ratings(ratings)
    print("\nDone. Combined ratings written to Supabase.")