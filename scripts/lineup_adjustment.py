import os
import sys
import json
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from poisson_model import load_data, calculate_team_ratings, predict_match
from catboost_model import predict_with_catboost, load_model
from fetch_fixtures import normalize_team_name

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
HEADERS = {"X-Auth-Token": API_KEY}
BASE_URL = "https://api.football-data.org/v4"

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

KEY_PLAYERS = {
    "Arsenal": {
        "attackers": ["Bukayo Saka", "Leandro Trossard", "Gabriel Martinelli"],
        "defenders": ["David Raya", "William Saliba"],
        "attack_importance": 0.15,
        "defence_importance": 0.12,
    },
    "Man City": {
        "attackers": ["Erling Haaland", "Phil Foden", "Kevin De Bruyne"],
        "defenders": ["Ederson", "Ruben Dias"],
        "attack_importance": 0.18,
        "defence_importance": 0.10,
    },
    "Liverpool": {
        "attackers": ["Mohamed Salah", "Luis Diaz", "Darwin Nunez"],
        "defenders": ["Alisson", "Virgil van Dijk"],
        "attack_importance": 0.18,
        "defence_importance": 0.12,
    },
    "Chelsea": {
        "attackers": ["Cole Palmer", "Nicolas Jackson"],
        "defenders": ["Robert Sanchez", "Levi Colwill"],
        "attack_importance": 0.20,
        "defence_importance": 0.10,
    },
    "Man United": {
        "attackers": ["Rasmus Hojlund", "Bruno Fernandes"],
        "defenders": ["Andre Onana", "Lisandro Martinez"],
        "attack_importance": 0.18,
        "defence_importance": 0.12,
    },
    "Tottenham": {
        "attackers": ["Son Heung-min", "Dominic Solanke"],
        "defenders": ["Guglielmo Vicario", "Cristian Romero"],
        "attack_importance": 0.16,
        "defence_importance": 0.10,
    },
    "Newcastle": {
        "attackers": ["Alexander Isak", "Anthony Gordon"],
        "defenders": ["Nick Pope", "Fabian Schar"],
        "attack_importance": 0.20,
        "defence_importance": 0.12,
    },
    "Aston Villa": {
        "attackers": ["Ollie Watkins", "Morgan Rogers"],
        "defenders": ["Emiliano Martinez", "Ezri Konsa"],
        "attack_importance": 0.18,
        "defence_importance": 0.10,
    },
    "Brighton": {
        "attackers": ["Joao Pedro", "Kaoru Mitoma"],
        "defenders": ["Bart Verbruggen", "Lewis Dunk"],
        "attack_importance": 0.15,
        "defence_importance": 0.10,
    },
    "Brentford": {
        "attackers": ["Bryan Mbeumo", "Yoane Wissa"],
        "defenders": ["Mark Flekken", "Kristoffer Ajer"],
        "attack_importance": 0.18,
        "defence_importance": 0.10,
    },
    "Fulham": {
        "attackers": ["Raul Jimenez", "Andreas Pereira"],
        "defenders": ["Bernd Leno", "Joachim Andersen"],
        "attack_importance": 0.15,
        "defence_importance": 0.10,
    },
    "West Ham": {
        "attackers": ["Jarrod Bowen", "Mohammed Kudus"],
        "defenders": ["Lukasz Fabianski", "Aaron Cresswell"],
        "attack_importance": 0.18,
        "defence_importance": 0.10,
    },
    "Wolves": {
        "attackers": ["Matheus Cunha", "Hwang Hee-chan"],
        "defenders": ["Jose Sa", "Max Kilman"],
        "attack_importance": 0.15,
        "defence_importance": 0.10,
    },
    "Everton": {
        "attackers": ["Dominic Calvert-Lewin", "Iliman Ndiaye"],
        "defenders": ["Jordan Pickford", "James Tarkowski"],
        "attack_importance": 0.15,
        "defence_importance": 0.10,
    },
    "Crystal Palace": {
        "attackers": ["Eberechi Eze", "Jean-Philippe Mateta"],
        "defenders": ["Dean Henderson", "Marc Guehi"],
        "attack_importance": 0.18,
        "defence_importance": 0.10,
    },
    "Bournemouth": {
        "attackers": ["Evanilson", "Antoine Semenyo"],
        "defenders": ["Kepa Arrizabalaga", "Ilya Zabarnyi"],
        "attack_importance": 0.15,
        "defence_importance": 0.10,
    },
    "Nott'm Forest": {
        "attackers": ["Chris Wood", "Callum Hudson-Odoi"],
        "defenders": ["Matz Sels", "Murillo"],
        "attack_importance": 0.15,
        "defence_importance": 0.12,
    },
    "Leicester": {
        "attackers": ["Jamie Vardy", "Stephy Mavididi"],
        "defenders": ["Danny Ward", "Caleb Okoli"],
        "attack_importance": 0.15,
        "defence_importance": 0.10,
    },
    "Southampton": {
        "attackers": ["Adam Armstrong", "Cameron Archer"],
        "defenders": ["Aaron Ramsdale", "Jan Bednarek"],
        "attack_importance": 0.15,
        "defence_importance": 0.10,
    },
    "Ipswich": {
        "attackers": ["Liam Delap", "Omari Hutchinson"],
        "defenders": ["Arijanet Muric", "Luke Woolfenden"],
        "attack_importance": 0.15,
        "defence_importance": 0.10,
    },
    "Leeds": {
        "attackers": ["Patrick Bamford", "Wilfried Gnonto"],
        "defenders": ["Illan Meslier", "Pascal Struijk"],
        "attack_importance": 0.15,
        "defence_importance": 0.10,
    },
    "Burnley": {
        "attackers": ["Lyle Foster", "Wilson Odobert"],
        "defenders": ["James Trafford", "Connor Roberts"],
        "attack_importance": 0.15,
        "defence_importance": 0.10,
    },
    "Sunderland": {
        "attackers": ["Eliezer Mayenda", "Romaine Mundle"],
        "defenders": ["Anthony Patterson", "Dan Ballard"],
        "attack_importance": 0.15,
        "defence_importance": 0.10,
    },
}


def get_upcoming_matches(minutes_ahead=75):
    now = datetime.now(timezone.utc)
    date_from = now.strftime("%Y-%m-%d")
    date_to = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    url = f"{BASE_URL}/competitions/PL/matches"
    params = {"status": "SCHEDULED", "dateFrom": date_from, "dateTo": date_to}
    response = requests.get(url, headers=HEADERS, params=params, timeout=10)

    if response.status_code != 200:
        print(f"Error fetching matches: {response.status_code}")
        return []

    matches = response.json().get("matches", [])
    upcoming = []

    for match in matches:
        kickoff = datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00"))
        minutes_until = (kickoff - now).total_seconds() / 60
        if 0 < minutes_until <= minutes_ahead:
            upcoming.append(match)
            print(f"Match kicking off in {int(minutes_until)} mins: {match['homeTeam']['name']} vs {match['awayTeam']['name']}")

    return upcoming


def fetch_lineup(fixture_id):
    url = f"{BASE_URL}/matches/{fixture_id}"
    response = requests.get(url, headers=HEADERS, timeout=10)
    if response.status_code != 200:
        return None
    data = response.json()
    return data.get("lineups")


def calculate_lineup_adjustment(team_name, starting_xi):
    if team_name not in KEY_PLAYERS:
        return 1.0, 1.0

    key = KEY_PLAYERS[team_name]
    attack_multiplier = 1.0
    defence_multiplier = 1.0

    starting_names = [p.get("name", "") for p in starting_xi]

    for attacker in key["attackers"]:
        if not any(attacker.lower() in name.lower() for name in starting_names):
            attack_multiplier -= key["attack_importance"]
            print(f"  Key attacker absent: {attacker} — attack reduced by {key['attack_importance']}")

    for defender in key["defenders"]:
        if not any(defender.lower() in name.lower() for name in starting_names):
            defence_multiplier -= key["defence_importance"]
            print(f"  Key defender absent: {defender} — defence weakened by {key['defence_importance']}")

    return max(attack_multiplier, 0.6), max(defence_multiplier, 0.7)


def update_prediction_with_lineup(fixture_id, home_team, away_team,
                                   home_lineup, away_lineup,
                                   ratings, avg_home, avg_away):
    print(f"\nAdjusting prediction for {home_team} vs {away_team}")

    home_att_mult, home_def_mult = calculate_lineup_adjustment(home_team, home_lineup)
    away_att_mult, away_def_mult = calculate_lineup_adjustment(away_team, away_lineup)

    adjusted_ratings = {}
    for team, r in ratings.items():
        adjusted_ratings[team] = r.copy()

    adjusted_ratings[home_team]['home_attack'] *= home_att_mult
    adjusted_ratings[home_team]['home_defence'] *= home_def_mult
    adjusted_ratings[away_team]['away_attack'] *= away_att_mult
    adjusted_ratings[away_team]['away_defence'] *= away_def_mult

    prediction = predict_match(home_team, away_team, adjusted_ratings, avg_home, avg_away)

    catboost = load_model()
    if catboost:
        prediction = predict_with_catboost(
            prediction,
            home_form=0.5,
            away_form=0.5,
            home_rest=7,
            away_rest=7,
        )

    import json as json_module
    data = {
        "fixture_id": str(fixture_id),
        "scoreline_matrix": json_module.dumps(prediction["scoreline_matrix"]),
        "top_pick_home": prediction["top_3"][0]["home"],
        "top_pick_away": prediction["top_3"][0]["away"],
        "top_pick_probability": prediction["top_3"][0]["probability"],
        "second_pick": json_module.dumps(prediction["top_3"][1]),
        "third_pick": json_module.dumps(prediction["top_3"][2]),
        "home_win_pct": prediction["home_win_pct"],
        "draw_pct": prediction["draw_pct"],
        "away_win_pct": prediction["away_win_pct"],
        "xg_home": prediction["xg_home"],
        "xg_away": prediction["xg_away"],
        "lineup_adjusted": True,
    }

    supabase.table("gaffer_predictions").upsert(data).execute()
    print(f"  Updated: {home_team} {prediction['home_win_pct']}% | Draw {prediction['draw_pct']}% | {away_team} {prediction['away_win_pct']}%")
    print(f"  Top pick: {home_team} {prediction['top_3'][0]['home']}–{prediction['top_3'][0]['away']} {away_team}")


if __name__ == "__main__":
    print("--- The Gaffer: Pre-Kickoff Lineup Adjustment ---\n")

    upcoming = get_upcoming_matches(minutes_ahead=75)

    if not upcoming:
        print("No matches kicking off in the next 75 minutes. Nothing to update.")
        exit()

    print(f"\nFound {len(upcoming)} match(es) to update\n")

    data = load_data()
    ratings, avg_home, avg_away = calculate_team_ratings(data)

    for match in upcoming:
        fixture_id = match["id"]
        home_team = normalize_team_name(match["homeTeam"]["name"])
        away_team = normalize_team_name(match["awayTeam"]["name"])

        lineups = fetch_lineup(fixture_id)

        if not lineups or len(lineups) < 2:
            print(f"Lineups not available yet for {home_team} vs {away_team} — skipping")
            continue

        home_lineup = lineups[0].get("startXI", [])
        away_lineup = lineups[1].get("startXI", [])

        if not home_lineup or not away_lineup:
            print(f"Starting XI not confirmed yet for {home_team} vs {away_team} — skipping")
            continue

        update_prediction_with_lineup(
            fixture_id, home_team, away_team,
            home_lineup, away_lineup,
            ratings, avg_home, avg_away
        )

    print("\nLineup adjustment complete.")