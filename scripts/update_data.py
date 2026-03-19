import os
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

CURRENT_SEASON_URL = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
CURRENT_SEASON_FILE = os.path.join(DATA_DIR, "E0_2526.csv")


def download_latest_csv():
    print("Downloading latest E0_2526.csv from football-data.co.uk...")
    response = requests.get(CURRENT_SEASON_URL, timeout=15)
    if response.status_code == 200:
        with open(CURRENT_SEASON_FILE, 'wb') as f:
            f.write(response.content)
        print("E0_2526.csv updated successfully")
    else:
        print(f"Failed to download: HTTP {response.status_code}")


if __name__ == "__main__":
    download_latest_csv()