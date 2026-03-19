import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

CSV_FILES = [
    'E0_2021.csv', 'E0_2122.csv', 'E0_2223.csv',
    'E0_2324.csv', 'E0_2425.csv', 'E0_2526.csv',
]

XG_PER_SHOT_ON_TARGET = 0.35
TIME_DECAY_XI = 0.0065


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
    data['xg_home'] = data['HST'] * XG_PER_SHOT_ON_TARGET
    data['xg_away'] = data['AST'] * XG_PER_SHOT_ON_TARGET
    return data


def time_decay_weights(dates, xi=TIME_DECAY_XI):
    latest = dates.max()
    days_ago = (latest - dates).dt.days
    return np.exp(-xi * days_ago)


def calculate_xg_ratings(data):
    data = data.copy()
    data['weight'] = time_decay_weights(data['Date'])

    teams = sorted(
        set(data['HomeTeam'].unique()) | set(data['AwayTeam'].unique())
    )

    ratings = {}
    for team in teams:
        home_games = data[data['HomeTeam'] == team]
        away_games = data[data['AwayTeam'] == team]

        if len(home_games) > 0:
            xg_home_for = np.average(home_games['xg_home'], weights=home_games['weight'])
            xg_home_against = np.average(home_games['xg_away'], weights=home_games['weight'])
        else:
            xg_home_for = 1.0
            xg_home_against = 1.0

        if len(away_games) > 0:
            xg_away_for = np.average(away_games['xg_away'], weights=away_games['weight'])
            xg_away_against = np.average(away_games['xg_home'], weights=away_games['weight'])
        else:
            xg_away_for = 1.0
            xg_away_against = 1.0

        ratings[team] = {
            'xg_home_for': round(xg_home_for, 3),
            'xg_home_against': round(xg_home_against, 3),
            'xg_away_for': round(xg_away_for, 3),
            'xg_away_against': round(xg_away_against, 3),
        }

        print(f"  {team}: xG home {ratings[team]['xg_home_for']} scored / {ratings[team]['xg_home_against']} conceded | away {ratings[team]['xg_away_for']} / {ratings[team]['xg_away_against']}")

    return ratings


def save_xg_to_supabase(ratings):
    print("\nSaving xG ratings to Supabase...")
    for team_name, r in ratings.items():
        team_id = team_name.lower().replace(" ", "_").replace("'", "")
        data = {
            "team_id": team_id,
            "team_name": team_name,
            "xg_rolling_home": r["xg_home_for"],
            "xg_rolling_away": r["xg_away_for"],
            "xg_home_against": r["xg_home_against"],
            "xg_away_against": r["xg_away_against"],
        }
        supabase.table("team_ratings").upsert(data).execute()
    print(f"Saved xG ratings for {len(ratings)} teams")


if __name__ == "__main__":
    print("--- The Gaffer: Calculating xG Ratings from CSV ---\n")
    data = load_data()
    print(f"Loaded {len(data)} matches with shot data\n")
    ratings = calculate_xg_ratings(data)
    save_xg_to_supabase(ratings)
    print("\nDone. xG ratings written to team_ratings table.")