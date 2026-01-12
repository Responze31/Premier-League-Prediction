import requests
from src.config import HEADERS, BASE_URL

def get_matches(season):
    url = f"{BASE_URL}/competitions/PL/matches?season={season}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()["matches"]
