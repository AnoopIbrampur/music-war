"""Data cleaning: dedup, name standardisation, genre rollup, missing data.

Every transform is pure (DataFrame in, DataFrame out) and idempotent —
running the cleaner twice yields the same result.
"""

from __future__ import annotations

import logging
import re
import unicodedata

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

# Spotify's micro-genres rolled up to ~20 parent genres. First keyword hit
# wins, so more specific tokens come first.
GENRE_ROLLUP: list[tuple[str, str]] = [
    ("k-pop", "kpop"), ("kpop", "kpop"),
    ("afrobeat", "afrobeats"), ("afroswing", "afrobeats"),
    ("hip hop", "hip_hop"), ("hip_hop", "hip_hop"), ("rap", "hip_hop"),
    ("drill", "hip_hop"), ("trap", "hip_hop"), ("grime", "hip_hop"),
    ("r&b", "rnb"), ("rnb", "rnb"), ("soul", "rnb"), ("funk", "rnb"),
    ("metal", "metal"), ("metalcore", "metal"),
    ("punk", "punk"), ("emo", "punk"), ("hardcore", "punk"),
    ("indie", "indie"), ("alternative", "indie"), ("alt ", "indie"), ("shoegaze", "indie"),
    ("house", "electronic"), ("edm", "electronic"), ("techno", "electronic"),
    ("dubstep", "electronic"), ("electro", "electronic"), ("dance", "electronic"),
    ("reggaeton", "latin"), ("latin", "latin"), ("salsa", "latin"),
    ("bachata", "latin"), ("corrido", "latin"),
    ("reggae", "reggae"), ("dancehall", "reggae"),
    ("country", "country"), ("bluegrass", "country"), ("americana", "country"),
    ("folk", "folk"), ("singer-songwriter", "folk"),
    ("jazz", "jazz"), ("bebop", "jazz"), ("swing", "jazz"),
    ("classical", "classical"), ("orchestra", "classical"), ("piano", "classical"),
    ("blues", "blues"),
    ("gospel", "gospel"), ("christian", "gospel"), ("worship", "gospel"),
    ("soundtrack", "soundtrack"), ("score", "soundtrack"),
    ("rock", "rock"), ("grunge", "rock"),
    ("pop", "pop"),
]

_WHITESPACE_RE = re.compile(r"\s+")
_THE_PREFIX_RE = re.compile(r"^the\s+", re.IGNORECASE)


def standardize_artist_name(name: str) -> str:
    """Canonical key for an artist name.

    Handles "The Weeknd" vs "Weeknd", "JAY-Z" vs "Jay-Z" vs "Jay Z",
    accents, and stray whitespace. Used only for matching/dedup — display
    names keep their original casing.
    """
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = _THE_PREFIX_RE.sub("", s)
    s = re.sub(r"[-_.]", " ", s)
    s = re.sub(r"[^\w\s&$]", "", s)
    return _WHITESPACE_RE.sub(" ", s).strip()


def rollup_genre(raw_genre: str | None) -> str:
    """Map a granular Spotify genre string to one of ~20 parent genres."""
    if not raw_genre or not isinstance(raw_genre, str):
        return "other"
    g = raw_genre.lower()
    for keyword, parent in GENRE_ROLLUP:
        if keyword in g:
            return parent
    return "other"


def deduplicate_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate tracks (same song collected from multiple playlists).

    Exact track_id dupes go first; then near-dupes — same standardized
    (name, primary artist) — keep the most popular version (e.g. album cut
    over compilation re-release).
    """
    df = tracks.drop_duplicates(subset=["track_id"]).copy()
    if "primary_artist_name" in df.columns:
        df["_dedup_key"] = (
            df["track_name"].str.lower().str.strip()
            + "::"
            + df["primary_artist_name"].map(standardize_artist_name)
        )
        df = (
            df.sort_values("spotify_popularity", ascending=False)
            .drop_duplicates(subset=["_dedup_key"])
            .drop(columns=["_dedup_key"])
        )
    return df.reset_index(drop=True)


def flag_special_albums(tracks: pd.DataFrame) -> pd.DataFrame:
    """Flag compilations, soundtracks, and holiday music for separate handling."""
    df = tracks.copy()
    album = df.get("album_name", pd.Series("", index=df.index)).fillna("").str.lower()
    name = df["track_name"].fillna("").str.lower()
    df["is_compilation"] = (df.get("album_type", "") == "compilation") | album.str.contains(
        "greatest hits|best of|compilation", regex=True
    )
    df["is_soundtrack"] = album.str.contains("soundtrack|motion picture|original score", regex=True)
    df["is_holiday"] = (album + " " + name).str.contains(
        "christmas|holiday|xmas|hanukkah", regex=True
    )
    return df


def missing_data_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts and rates, sorted worst-first."""
    report = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_rate": (df.isna().mean() * 100).round(2),
        }
    ).sort_values("missing_rate", ascending=False)
    report.index.name = "column"
    return report.reset_index()


def clean_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pass: dedupe, filter, standardise, flag specials."""
    n0 = len(tracks)
    df = deduplicate_tracks(tracks)
    df = df[df["spotify_popularity"].fillna(0) >= settings.MIN_POPULARITY]
    df = flag_special_albums(df)
    if "primary_genre" in df.columns:
        df["primary_genre"] = df["primary_genre"].map(rollup_genre)
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df = df.dropna(subset=["release_year"])
    df["release_year"] = df["release_year"].astype(int)
    logger.info("Cleaning: %d -> %d tracks", n0, len(df))
    return df.reset_index(drop=True)
