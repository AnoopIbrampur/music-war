"""Feature engineering: buckets, encodings, audio clusters, and the
composite success score that the WAR models predict.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import settings

logger = logging.getLogger(__name__)

AUDIO_FEATURES = [
    "danceability", "energy", "valence", "tempo", "loudness",
    "speechiness", "acousticness", "instrumentalness",
]


def era_bucket(year: int | float) -> str:
    """Map a release year onto the project's five era buckets."""
    if pd.isna(year):
        return "unknown"
    for start, end, label in settings.ERA_BUCKETS:
        if start <= int(year) <= end:
            return label
    return "pre_90s" if int(year) < 1990 else "unknown"


def season_released(release_date: str | None) -> str:
    """Quarter of release (summer drops behave differently)."""
    if release_date is None or pd.isna(release_date):
        return "unknown"
    ts = pd.to_datetime(release_date, errors="coerce")
    if pd.isna(ts):
        return "unknown"
    return f"Q{(int(ts.month) - 1) // 3 + 1}"


def tempo_bucket(tempo: float | None) -> str:
    if tempo is None or pd.isna(tempo):
        return "unknown"
    if tempo < 100:
        return "slow"
    if tempo < 130:
        return "mid"
    return "fast"


def duration_bucket(duration_ms: float | None) -> str:
    if duration_ms is None or pd.isna(duration_ms):
        return "unknown"
    minutes = duration_ms / 60000
    if minutes < 3:
        return "short"
    if minutes <= 4:
        return "standard"
    return "long"


def add_audio_clusters(tracks: pd.DataFrame,
                       n_clusters: int = settings.N_AUDIO_CLUSTERS,
                       random_state: int = settings.RANDOM_STATE) -> pd.DataFrame:
    """K-means "sound profile" clusters over the standardized audio features."""
    df = tracks.copy()
    available = [f for f in AUDIO_FEATURES if f in df.columns]
    if not available:
        logger.warning("No audio features available — skipping sound-profile clustering")
        return df
    X = df[available].fillna(df[available].median())
    scaled = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["audio_cluster"] = km.fit_predict(scaled)
    return df


def compute_composite_score(tracks: pd.DataFrame) -> pd.DataFrame:
    """Composite success score (0–100).

    Weights: Spotify popularity 40%, inverse-normalised Billboard peak 30%,
    weeks-on-chart 20%, longevity bonus 10%. Tracks that never charted use
    popularity alone with a small penalty (avoids rewarding the selection
    bias of the charted subset).
    """
    df = tracks.copy()
    w = settings.SCORE_WEIGHTS

    pop = df["spotify_popularity"].fillna(0).clip(0, 100)
    peak = pd.to_numeric(df.get("billboard_peak_position"), errors="coerce")
    weeks = pd.to_numeric(df.get("billboard_weeks_on_chart"), errors="coerce")
    charted = peak.notna()

    peak_score = ((101 - peak) / 100 * 100).where(charted, 0)           # rank 1 -> 100
    weeks_score = (weeks.clip(upper=52) / 52 * 100).where(charted, 0)   # cap at a year
    # Longevity: old tracks still popular today have durable appeal
    age = (2026 - df["release_year"].fillna(2026)).clip(lower=0)
    longevity = (pop * (age / 35)).clip(0, 100)

    score = (
        w["spotify_popularity"] * pop
        + w["billboard_peak"] * peak_score
        + w["billboard_weeks"] * weeks_score
        + w["longevity"] * longevity
    )
    # Non-charted: popularity-only, lightly penalised, same 0-100 scale
    fallback = (w["spotify_popularity"] + w["billboard_peak"] + w["billboard_weeks"]) * pop \
        + w["longevity"] * longevity
    score = score.where(charted, fallback * settings.NO_CHART_PENALTY)

    df["composite_success_score"] = score.clip(0, 100).round(2)
    return df


def engineer_track_features(tracks: pd.DataFrame) -> pd.DataFrame:
    """All track-level engineered features in one pass."""
    df = tracks.copy()
    df["era"] = df["release_year"].map(era_bucket)
    df["season_released"] = df["release_date"].map(season_released)
    df["tempo_bucket"] = df["tempo"].map(tempo_bucket) if "tempo" in df else "unknown"
    df["duration_bucket"] = df["duration_ms"].map(duration_bucket)
    df["is_explicit"] = df.get("explicit", False).astype(int)
    df = add_audio_clusters(df)
    df = compute_composite_score(df)
    return df


def engineer_collab_features(tracks: pd.DataFrame, track_artists: pd.DataFrame,
                             track_producers: pd.DataFrame) -> pd.DataFrame:
    """Collaboration features: crew sizes and feature flags per track."""
    df = tracks.copy()
    artist_counts = track_artists.groupby("track_id").size().rename("num_artists_on_track")
    producer_counts = track_producers.groupby("track_id").size().rename("num_producers_on_track")
    has_feature = (
        track_artists[track_artists["role"] == "featured_artist"]
        .groupby("track_id").size().gt(0).rename("has_feature")
    )
    df = (
        df.merge(artist_counts, left_on="track_id", right_index=True, how="left")
        .merge(producer_counts, left_on="track_id", right_index=True, how="left")
        .merge(has_feature, left_on="track_id", right_index=True, how="left")
    )
    df["num_artists_on_track"] = df["num_artists_on_track"].fillna(1).astype(int)
    df["num_producers_on_track"] = df["num_producers_on_track"].fillna(0).astype(int)
    df["has_feature"] = df["has_feature"].eq(True).astype(int)
    return df


def engineer_artist_features(track_artists: pd.DataFrame, tracks: pd.DataFrame,
                             artists: pd.DataFrame) -> pd.DataFrame:
    """Artist-level aggregates: prolificness, career span, genre diversity."""
    merged = track_artists.merge(
        tracks[["track_id", "release_year", "primary_genre", "spotify_popularity"]],
        on="track_id", how="left",
    )
    agg = merged.groupby("artist_id").agg(
        artist_track_count=("track_id", "nunique"),
        career_start_year=("release_year", "min"),
        career_end_year=("release_year", "max"),
        artist_genre_diversity=("primary_genre", "nunique"),
        artist_avg_popularity=("spotify_popularity", "mean"),
    )
    agg["career_length"] = agg["career_end_year"] - agg["career_start_year"]
    out = artists.merge(agg, on="artist_id", how="left")
    out["total_tracks_in_dataset"] = out["artist_track_count"].fillna(0).astype(int)
    return out
