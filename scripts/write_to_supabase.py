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


def log_match_performance():
    print("--- Logging post-match performance ---")

    response = supabase.table("gaffer_predictions")\
        .select("*")\
        .eq("lineup_adjusted", False)\
        .execute()

    predictions = response.data
    if not predictions:
        print("No predictions to check.")
        return

    import requests as req
    headers = {"X-Auth-Token": os.getenv("FOOTBALL_DATA_API_KEY")}

    for pred in predictions:
        fixture_id = pred["fixture_id"]

        if fixture_id.startswith("test_"):
            continue

        already_logged = supabase.table("gaffer_performance")\
            .select("id")\
            .eq("fixture_id", fixture_id)\
            .execute()

        if already_logged.data:
            continue

        url = f"https://api.football-data.org/v4/matches/{fixture_id}"
        resp = req.get(url, headers=headers, timeout=10)

        if resp.status_code != 200:
            continue

        match_data = resp.json()
        status = match_data.get("status")

        if status != "FINISHED":
            continue

        score = match_data.get("score", {}).get("fullTime", {})
        actual_home = score.get("home")
        actual_away = score.get("away")

        if actual_home is None or actual_away is None:
            continue

        predicted_home = pred["top_pick_home"]
        predicted_away = pred["top_pick_away"]

        if actual_home > actual_away:
            actual_outcome = "H"
        elif actual_home < actual_away:
            actual_outcome = "A"
        else:
            actual_outcome = "D"

        if pred["home_win_pct"] > pred["draw_pct"] and pred["home_win_pct"] > pred["away_win_pct"]:
            predicted_outcome = "H"
        elif pred["away_win_pct"] > pred["draw_pct"] and pred["away_win_pct"] > pred["home_win_pct"]:
            predicted_outcome = "A"
        else:
            predicted_outcome = "D"

        outcome_correct = predicted_outcome == actual_outcome

        home_prob = pred["home_win_pct"] / 100
        draw_prob = pred["draw_pct"] / 100
        away_prob = pred["away_win_pct"] / 100

        if actual_outcome == "H":
            brier = (1 - home_prob) ** 2 + draw_prob ** 2 + away_prob ** 2
        elif actual_outcome == "D":
            brier = home_prob ** 2 + (1 - draw_prob) ** 2 + away_prob ** 2
        else:
            brier = home_prob ** 2 + draw_prob ** 2 + (1 - away_prob) ** 2

        performance_data = {
            "fixture_id": fixture_id,
            "predicted_top_pick": f"{predicted_home}-{predicted_away}",
            "actual_result": f"{actual_home}-{actual_away}",
            "outcome_correct": outcome_correct,
            "brier_score": round(brier / 3, 4),
            "gameweek": pred["gameweek"],
        }

        supabase.table("gaffer_performance").insert(performance_data).execute()
        print(f"Logged: {pred['home_team']} vs {pred['away_team']} — predicted {predicted_outcome}, actual {actual_outcome}, correct: {outcome_correct}")


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
        write_prediction(
            prediction,
            fixture_id,
            home_team,
            away_team,
            kickoff_time,
            gameweek,
        )

    print(f"\nPipeline complete. {len(matches) - skipped} predictions saved, {skipped} skipped.")
    print("\nChecking for finished matches to log...")
    log_match_performance()