"""Bulk real-data ingestion.

Spotify locked its live API endpoints (audio-features, popularity, genres)
for apps created after Nov 2024, so live ingestion can't populate the model.
Instead we use a bulk Spotify export — real tracks, artists, popularity, and
audio features — published on Hugging Face
(``maharshipandya/spotify-tracks-dataset``, 114k tracks, 114 genres).

This is real Spotify data; it is simply a pre-lockdown snapshot rather than a
live pull. For a retrospective WAR analysis that distinction is immaterial —
popularity is a snapshot either way.

Limitations of this source (handled gracefully downstream):
  * no release dates  -> era features become "unknown"
  * no producer/songwriter credits -> only Artist WAR unless MusicBrainz
    enrichment is run separately.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from config import settings
from src.ingestion.spotify_client import parse_featured_artists
from src.processing.cleaner import standardize_artist_name

logger = logging.getLogger(__name__)

DATASET = "maharshipandya/spotify-tracks-dataset"
PARQUET_URL = (
    f"https://huggingface.co/api/datasets/{DATASET}/parquet/default/train/0.parquet"
)
RAW_PATH = settings.RAW_DIR / "bulk" / "spotify_tracks_raw.parquet"

# Collaborators are packed into one ";"-separated string in this dataset.
_ARTIST_SEP = ";"


def download_spotify_dataset(force: bool = False) -> pd.DataFrame:
    """Download (and cache) the real Spotify tracks parquet."""
    if RAW_PATH.exists() and not force:
        logger.info("Using cached bulk dataset at %s", RAW_PATH)
        return pd.read_parquet(RAW_PATH)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading real Spotify dataset from Hugging Face ...")
    resp = requests.get(PARQUET_URL, timeout=180)
    resp.raise_for_status()
    df = pd.read_parquet(io.BytesIO(resp.content))
    df.to_parquet(RAW_PATH, index=False)
    logger.info("Downloaded %d rows -> %s", len(df), RAW_PATH)
    return df


def _split_artists(artists_field: str) -> list[str]:
    """Split the ';'-separated artists string into individual names."""
    if not isinstance(artists_field, str):
        return []
    return [a.strip() for a in artists_field.split(_ARTIST_SEP) if a.strip()]


def load_bulk_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (tracks, track_artists, artists) in the pipeline's schema.

    Artist ids are derived from standardized names so the same artist maps to
    a single WAR column across all their tracks (the dataset ships no ids).
    """
    raw = download_spotify_dataset()

    # Same track_id recurs under multiple genre labels; keep the most popular.
    raw = (
        raw.sort_values("popularity", ascending=False)
        .drop_duplicates(subset="track_id")
        .reset_index(drop=True)
    )

    tracks = pd.DataFrame(
        {
            "track_id": raw["track_id"],
            "track_name": raw["track_name"].fillna("Unknown"),
            "album_id": raw["album_name"].fillna("").map(lambda s: "al_" + standardize_artist_name(s)),
            "album_name": raw["album_name"],
            "album_type": "album",
            "release_date": pd.NA,  # not present in this dataset
            "duration_ms": raw["duration_ms"],
            "explicit": raw["explicit"].astype(bool),
            "spotify_popularity": raw["popularity"].astype(float),
            "primary_genre": raw["track_genre"],
            "danceability": raw["danceability"],
            "energy": raw["energy"],
            "key": raw["key"],
            "loudness": raw["loudness"],
            "mode": raw["mode"],
            "speechiness": raw["speechiness"],
            "acousticness": raw["acousticness"],
            "instrumentalness": raw["instrumentalness"],
            "liveness": raw["liveness"],
            "valence": raw["valence"],
            "tempo": raw["tempo"],
            "time_signature": raw["time_signature"],
            "billboard_peak_position": pd.NA,
            "billboard_weeks_on_chart": pd.NA,
        }
    )

    # Build the track-artist bridge from the ";"-separated field plus any
    # "feat." names still embedded in the title.
    ta_rows: list[dict] = []
    artist_display: dict[str, str] = {}
    artist_genre: dict[str, str] = {}
    for _, r in raw.iterrows():
        names = _split_artists(r["artists"])
        _, title_feats = parse_featured_artists(str(r["track_name"]))
        for feat in title_feats:
            if feat not in names:
                names.append(feat)
        for order, name in enumerate(names, start=1):
            aid = "ar_" + standardize_artist_name(name)
            if not aid or aid == "ar_":
                continue
            artist_display.setdefault(aid, name)
            artist_genre.setdefault(aid, r["track_genre"])
            ta_rows.append(
                {
                    "track_id": r["track_id"],
                    "artist_id": aid,
                    "role": "primary_artist" if order == 1 else "featured_artist",
                    "billing_order": order,
                }
            )

    track_artists = pd.DataFrame(ta_rows).drop_duplicates(subset=["track_id", "artist_id"])

    artists = pd.DataFrame(
        {
            "artist_id": list(artist_display),
            "artist_name": [artist_display[a] for a in artist_display],
            "primary_genre": [artist_genre[a] for a in artist_display],
            "followers": pd.NA,
            "artist_popularity": pd.NA,
        }
    )

    logger.info(
        "Bulk load: %d tracks, %d artists, %d track-artist links",
        len(tracks), len(artists), len(track_artists),
    )
    return tracks, track_artists, artists
