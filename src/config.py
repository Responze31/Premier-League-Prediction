import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TOKEN")
if TOKEN is None:
    raise ValueError("API token not found in environment variables")

HEADERS = {"X-Auth-Token": TOKEN}
BASE_URL = "https://api.football-data.org/v4"
