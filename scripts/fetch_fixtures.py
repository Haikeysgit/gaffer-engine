import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

HEADERS = {"X-Auth-Token": API_KEY}
BASE_URL = "https://api.football-data.org/v4"

TEAM_NAME_MAP = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds",
    "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest",
    "Southampton FC": "Southampton",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
}


def normalize_team_name(api_name):
    return TEAM_NAME_MAP.get(api_name, api_name)


def fetch_upcoming_fixtures(days_ahead=14):
    today = datetime.utcnow().date()
    date_to = today + timedelta(days=days_ahead)

    url = f"{BASE_URL}/competitions/PL/matches"
    params = {
        "status": "SCHEDULED",
        "dateFrom": today.strftime("%Y-%m-%d"),
        "dateTo": date_to.strftime("%Y-%m-%d"),
    }

    print(f"Fetching fixtures from {today} to {date_to}...")
    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code != 200:
        print(f"Error fetching fixtures: {response.status_code}")
        print(response.text)
        return []

    data = response.json()
    matches = data.get("matches", [])
    print(f"Found {len(matches)} upcoming fixtures")
    return matches


def save_fixtures_to_supabase(matches):
    saved = 0
    for match in matches:
        fixture_id = str(match["id"])
        home_team = normalize_team_name(match["homeTeam"]["name"])
        away_team = normalize_team_name(match["awayTeam"]["name"])
        kickoff_time = match["utcDate"]
        gameweek = match.get("matchday")

        data = {
            "fixture_id": fixture_id,
            "home_team": home_team,
            "away_team": away_team,
            "kickoff_time": kickoff_time,
            "gameweek": gameweek,
            "lineup_adjusted": False,
        }

        supabase.table("gaffer_predictions").upsert(data).execute()
        print(f"Saved: GW{gameweek} {home_team} vs {away_team} ({kickoff_time})")
        saved += 1

    print(f"\nTotal fixtures saved: {saved}")


if __name__ == "__main__":
    print("--- The Gaffer: Fetching Fixtures ---\n")
    matches = fetch_upcoming_fixtures(days_ahead=14)

    if matches:
        save_fixtures_to_supabase(matches)
        print("\nFixtures saved to Supabase. Ready to predict.")
    else:
        print("No upcoming fixtures found.")