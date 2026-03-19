import os
import sys
import json
from supabase import create_client
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from poisson_model import load_data, calculate_team_ratings, predict_match
from fetch_fixtures import fetch_upcoming_fixtures, normalize_team_name

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def write_team_ratings(ratings):
    print("Writing team ratings to Supabase...")
    for team_name, r in ratings.items():
        data = {
            "team_id": team_name.lower().replace(" ", "_"),
            "team_name": team_name,
            "home_attack": r["home_attack"],
            "home_defence": r["home_defence"],
            "away_attack": r["away_attack"],
            "away_defence": r["away_defence"],
        }
        supabase.table("team_ratings").upsert(data).execute()
    print(f"Written {len(ratings)} team ratings\n")


def write_prediction(prediction, fixture_id, home_team, away_team, kickoff_time, gameweek):
    data = {
        "fixture_id": str(fixture_id),
        "home_team": home_team,
        "away_team": away_team,
        "kickoff_time": kickoff_time,
        "gameweek": gameweek,
        "scoreline_matrix": json.dumps(prediction["scoreline_matrix"]),
        "top_pick_home": prediction["top_3"][0]["home"],
        "top_pick_away": prediction["top_3"][0]["away"],
        "top_pick_probability": prediction["top_3"][0]["probability"],
        "second_pick": json.dumps(prediction["top_3"][1]),
        "third_pick": json.dumps(prediction["top_3"][2]),
        "home_win_pct": prediction["home_win_pct"],
        "draw_pct": prediction["draw_pct"],
        "away_win_pct": prediction["away_win_pct"],
        "xg_home": prediction["xg_home"],
        "xg_away": prediction["xg_away"],
        "lineup_adjusted": False,
    }
    supabase.table("gaffer_predictions").upsert(data).execute()
    print(f"Predicted: GW{gameweek} {home_team} vs {away_team}")
    print(f"  Top pick: {home_team} {prediction['top_3'][0]['home']}–{prediction['top_3'][0]['away']} {away_team} ({prediction['top_3'][0]['probability']}%)")
    print(f"  Outcome: {home_team} {prediction['home_win_pct']}% | Draw {prediction['draw_pct']}% | {away_team} {prediction['away_win_pct']}%")
    print(f"  xG: {home_team} {prediction['xg_home']} — {away_team} {prediction['xg_away']}\n")


if __name__ == "__main__":
    print("--- The Gaffer: Full Prediction Pipeline ---\n")

    data = load_data()
    ratings, avg_home, avg_away = calculate_team_ratings(data)
    print(f"Ratings ready for {len(ratings)} teams\n")

    write_team_ratings(ratings)

    matches = fetch_upcoming_fixtures(days_ahead=14)

    if not matches:
        print("No upcoming fixtures found.")
        exit()

    print(f"Running predictions for {len(matches)} fixtures...\n")

    skipped = 0
    for match in matches:
        fixture_id = str(match["id"])
        home_team = normalize_team_name(match["homeTeam"]["name"])
        away_team = normalize_team_name(match["awayTeam"]["name"])
        kickoff_time = match["utcDate"]
        gameweek = match.get("matchday")

        if home_team not in ratings or away_team not in ratings:
            print(f"Skipping {home_team} vs {away_team} — team not in ratings")
            skipped += 1
            continue

        prediction = predict_match(home_team, away_team, ratings, avg_home, avg_away)
        write_prediction(prediction, fixture_id, home_team, away_team, kickoff_time, gameweek)

    print(f"Pipeline complete. {len(matches) - skipped} predictions saved, {skipped} skipped.")