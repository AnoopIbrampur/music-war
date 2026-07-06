"""Central configuration for the Music WAR project.

All tunable constants live here so the pipeline, models, and dashboard
share a single source of truth. Secrets come from `.env` (never hardcoded).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------- paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- secrets
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")

MUSICBRAINZ_APP_NAME = os.getenv("MUSICBRAINZ_APP_NAME", "music-war")
MUSICBRAINZ_APP_VERSION = os.getenv("MUSICBRAINZ_APP_VERSION", "0.1.0")
MUSICBRAINZ_CONTACT = os.getenv("MUSICBRAINZ_CONTACT", "")

DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{DATA_DIR / 'music_war.db'}"
BILLBOARD_FALLBACK_CSV = os.getenv("BILLBOARD_FALLBACK_CSV", "")

# ---------------------------------------------------------------- ingestion
SPOTIFY_AUDIO_FEATURES_BATCH = 100   # max ids per audio-features request
SPOTIFY_MAX_RETRIES = 5
MUSICBRAINZ_RATE_LIMIT_SECONDS = 1.0  # MB policy: 1 request/second
BILLBOARD_CHART_START_YEAR = 2000
BILLBOARD_CHART_END_YEAR = 2025
FUZZY_MATCH_THRESHOLD = 88            # rapidfuzz token_set_ratio cutoff

# ---------------------------------------------------------------- cleaning
MIN_POPULARITY = 1        # drop tracks below this Spotify popularity
FEATURED_PATTERNS = (
    r"\bfeat\.?\b", r"\bft\.?\b", r"\bfeaturing\b", r"\bwith\b",
)

# ---------------------------------------------------------------- modeling
MIN_TRACKS_ARTIST = 5      # eligibility threshold for an artist column
MIN_TRACKS_PRODUCER = 5
MIN_TRACKS_SONGWRITER = 5
MIN_POPULARITY_FOR_ELIGIBILITY = 20
FEATURED_BILLING_WEIGHT = 0.5   # primary artists get 1.0
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
LASSO_ALPHAS = (0.001, 0.01, 0.1, 1.0)
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_AUDIO_CLUSTERS = 8
BOOTSTRAP_ITERATIONS = 100

# Composite success score weights (must sum to 1.0)
SCORE_WEIGHTS = {
    "spotify_popularity": 0.40,
    "billboard_peak": 0.30,
    "billboard_weeks": 0.20,
    "longevity": 0.10,
}
NO_CHART_PENALTY = 0.95   # multiplier for tracks that never charted

ERA_BUCKETS = [
    (1990, 1999, "90s"),
    (2000, 2004, "early_2000s"),
    (2005, 2009, "late_2000s"),
    (2010, 2019, "2010s"),
    (2020, 2029, "early_2020s"),
]
