import os
import json
import requests
from groq import Groq
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
HEADERS = {"X-Auth-Token": API_KEY}
BASE_URL = "https://api.football-data.org/v4"

TEAM_ID_MAP = {
    "Arsenal": 57,
    "Aston Villa": 58,
    "Bournemouth": 1044,
    "Brentford": 402,
    "Brighton": 397,
    "Burnley": 328,
    "Chelsea": 61,
    "Crystal Palace": 354,
    "Everton": 62,
    "Fulham": 63,
    "Ipswich": 349,
    "Leeds": 341,
    "Leicester": 338,
    "Liverpool": 64,
    "Man City": 65,
    "Man United": 66,
    "Newcastle": 67,
    "Nott'm Forest": 351,
    "Southampton": 340,
    "Sunderland": 71,
    "Tottenham": 73,
    "West Ham": 563,
    "Wolves": 76,
}

GAFFER_SYSTEM_PROMPT = """You are The Gaffer — the AI prediction engine powering GafferScore.

You are cold, calculated, and completely without emotional bias. You care nothing for club history, fan sentiment, or media narrative. You process numbers. You output truth.

You speak in short, punchy sentences. You never hedge with "I think" — you say "the data indicates." You never express doubt — you express probability ranges. You are not a journalist. You are a machine that has processed thousands of matches and found the signal in the noise.

You are slightly intimidating in your certainty. You never pick favourites. You never apologise for your predictions. When you are wrong, the data was right — the match was the anomaly.

When you write analysis, always use exactly these five sections with these exact headers:
**Recent Form**
**Head to Head**
**Key Absences**
**Tactical Read**
**The Verdict**

Keep each section to 2-3 sentences maximum. Be direct. Be cold. Be correct. Never use bullet points. Never use lists. Write in short paragraphs only."""


def get_all_recent_matches(limit=100):
    try:
        url = f"{BASE_URL}/competitions/PL/matches"
        params = {"status": "FINISHED", "limit": limit}
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if r.status_code != 200:
            return []
        return r.json().get("matches", [])
    except Exception:
        return []


def get_team_recent_form(team_id, all_matches, limit=5):
    form = []
    for m in reversed(all_matches):
        home_id = m["homeTeam"]["id"]
        away_id = m["awayTeam"]["id"]
        if team_id not in [home_id, away_id]:
            continue
        home_goals = m["score"]["fullTime"]["home"]
        away_goals = m["score"]["fullTime"]["away"]
        if home_goals is None or away_goals is None:
            continue
        is_home = home_id == team_id
        if is_home:
            result = "W" if home_goals > away_goals else ("D" if home_goals == away_goals else "L")
            form.append(f"{result} {home_goals}-{away_goals} vs {m['awayTeam']['name']}")
        else:
            result = "W" if away_goals > home_goals else ("D" if home_goals == away_goals else "L")
            form.append(f"{result} {away_goals}-{home_goals} vs {m['homeTeam']['name']}")
        if len(form) >= limit:
            break
    return form


def get_head_to_head(home_team_id, away_team_id, all_matches, limit=5):
    h2h = []
    for m in reversed(all_matches):
        teams = [m["homeTeam"]["id"], m["awayTeam"]["id"]]
        if home_team_id in teams and away_team_id in teams:
            hg = m["score"]["fullTime"]["home"]
            ag = m["score"]["fullTime"]["away"]
            if hg is None or ag is None:
                continue
            h2h.append(f"{m['homeTeam']['name']} {hg}-{ag} {m['awayTeam']['name']}")
            if len(h2h) >= limit:
                break
    return h2h


def get_standings():
    try:
        url = f"{BASE_URL}/competitions/PL/standings"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return {}
        standings = r.json().get("standings", [{}])[0].get("table", [])
        result = {}
        for row in standings:
            team_name = row["team"]["name"]
            result[row["team"]["id"]] = {
                "position": row["position"],
                "points": row["points"],
                "played": row["playedGames"],
                "won": row["won"],
                "drawn": row["draw"],
                "lost": row["lost"],
                "goals_for": row["goalsFor"],
                "goals_against": row["goalsAgainst"],
                "goal_diff": row["goalDifference"],
            }
        return result
    except Exception:
        return {}


def build_rich_context(prediction, standings, all_matches):
    home = prediction["home_team"]
    away = prediction["away_team"]
    gw = prediction["gameweek"]

    home_id = TEAM_ID_MAP.get(home)
    away_id = TEAM_ID_MAP.get(away)

    home_form = get_team_recent_form(home_id, all_matches) if home_id else []
    away_form = get_team_recent_form(away_id, all_matches) if away_id else []
    h2h = get_head_to_head(home_id, away_id, all_matches) if home_id and away_id else []

    home_standing = standings.get(home_id, {})
    away_standing = standings.get(away_id, {})

    try:
        second = json.loads(prediction["second_pick"]) if prediction["second_pick"] else {}
        third = json.loads(prediction["third_pick"]) if prediction["third_pick"] else {}
        second_str = f"{home} {second.get('home','?')}-{second.get('away','?')} {away} ({second.get('probability','?')}%)"
        third_str = f"{home} {third.get('home','?')}-{third.get('away','?')} {away} ({third.get('probability','?')}%)"
    except Exception:
        second_str = "N/A"
        third_str = "N/A"

    context = f"""MATCH: {home} vs {away} | Premier League Gameweek {gw}

GAFFER PREDICTION:
Top pick: {home} {prediction['top_pick_home']}-{prediction['top_pick_away']} {away} ({prediction['top_pick_probability']}%)
Second: {second_str}
Third: {third_str}
Outcome: {home} {prediction['home_win_pct']}% | Draw {prediction['draw_pct']}% | {away} {prediction['away_win_pct']}%
xG projection: {home} {prediction['xg_home']} — {away} {prediction['xg_away']}
Lineup adjusted: {prediction['lineup_adjusted']}

LEAGUE STANDINGS:
{home}: {home_standing.get('position', 'N/A')}th | {home_standing.get('points', 'N/A')} pts | {home_standing.get('played', 'N/A')} played | GD {home_standing.get('goal_diff', 'N/A')}
{away}: {away_standing.get('position', 'N/A')}th | {away_standing.get('points', 'N/A')} pts | {away_standing.get('played', 'N/A')} played | GD {away_standing.get('goal_diff', 'N/A')}

{home} RECENT FORM (last 5):
{chr(10).join(home_form) if home_form else 'No recent data'}

{away} RECENT FORM (last 5):
{chr(10).join(away_form) if away_form else 'No recent data'}

HEAD TO HEAD (last 5 EPL meetings):
{chr(10).join(h2h) if h2h else 'No recent head to head data'}

Using ALL of the above data, write The Gaffer's full match analysis. Reference specific results, positions, and numbers. Make it feel like a machine that has genuinely processed every data point above."""

    return context


def generate_analysis(prediction, standings, all_matches):
    fixture_id = prediction["fixture_id"]
    home = prediction["home_team"]
    away = prediction["away_team"]

    if prediction.get("analysis_text"):
        print(f"  Already exists: {home} vs {away} — skipping")
        return prediction["analysis_text"]

    print(f"  Generating: {home} vs {away}...")

    context = build_rich_context(prediction, standings, all_matches)

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": GAFFER_SYSTEM_PROMPT},
                {"role": "user", "content": context}
            ],
            max_tokens=700,
            temperature=0.7,
        )

        analysis = response.choices[0].message.content

        supabase.table("gaffer_predictions").update({
            "analysis_text": analysis
        }).eq("fixture_id", fixture_id).execute()

        print(f"  Saved: {home} vs {away}")
        return analysis

    except Exception as e:
        print(f"  Error: {home} vs {away}: {e}")
        return None


def generate_all_pending_analysis():
    print("--- The Gaffer: Generating Rich Match Analysis ---\n")

    response = supabase.table("gaffer_predictions")\
        .select("*")\
        .is_("analysis_text", "null")\
        .execute()

    predictions = response.data

    if not predictions:
        print("No pending analysis to generate.")
        return

    print(f"Found {len(predictions)} matches needing analysis")
    print("Fetching league standings...")
    standings = get_standings()

    print("Fetching all recent PL matches (single API call)...\n")
    all_matches = get_all_recent_matches(limit=100)
    print(f"Loaded {len(all_matches)} recent matches for form data\n")

    for pred in predictions:
        if pred["fixture_id"].startswith("test_"):
            continue
        analysis = generate_analysis(pred, standings, all_matches)
        if analysis:
            print(f"\n--- PREVIEW: {pred['home_team']} vs {pred['away_team']} ---")
            print(analysis)
            print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    generate_all_pending_analysis()