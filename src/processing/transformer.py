"""Transformer: reshape merged ingestion output into star-schema frames.

Bridges the gap between the collector's wide parquet files and the
DataFrames that ``db_manager.load_star_schema`` and the modeling layer
expect.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_producer_dim(track_producers: pd.DataFrame) -> pd.DataFrame:
    """Producer dimension from the track-producer bridge."""
    if track_producers.empty:
        return pd.DataFrame(columns=["producer_id", "producer_name", "total_tracks_produced"])
    dim = (
        track_producers.groupby("producer_id")
        .agg(
            producer_name=("producer_name", "first"),
            total_tracks_produced=("track_id", "nunique"),
        )
        .reset_index()
    )
    return dim


def build_songwriter_dim(track_songwriters: pd.DataFrame) -> pd.DataFrame:
    if track_songwriters.empty:
        return pd.DataFrame(columns=["songwriter_id", "songwriter_name", "total_tracks_written"])
    dim = (
        track_songwriters.groupby("songwriter_id")
        .agg(
            songwriter_name=("songwriter_name", "first"),
            total_tracks_written=("track_id", "nunique"),
        )
        .reset_index()
    )
    return dim


def credits_to_bridges(credits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split raw MusicBrainz credits into producer and songwriter bridges.

    Credit names double as ids (prefixed by role) because MusicBrainz mbids
    are missing for a share of rows.
    """
    if credits.empty:
        empty_p = pd.DataFrame(columns=["track_id", "producer_id", "producer_name"])
        empty_s = pd.DataFrame(columns=["track_id", "songwriter_id", "songwriter_name"])
        return empty_p, empty_s

    def _make_id(row: pd.Series, prefix: str) -> str:
        return row["mbid"] if isinstance(row.get("mbid"), str) and row["mbid"] else (
            prefix + row["name"].lower().replace(" ", "_")
        )

    producers = credits[credits["role"] == "producer"].copy()
    producers["producer_id"] = producers.apply(_make_id, axis=1, prefix="pr_")
    producers = producers.rename(columns={"name": "producer_name"})[
        ["track_id", "producer_id", "producer_name"]
    ].drop_duplicates(subset=["track_id", "producer_id"])

    writers = credits[credits["role"] == "songwriter"].copy()
    writers["songwriter_id"] = writers.apply(_make_id, axis=1, prefix="sw_")
    writers = writers.rename(columns={"name": "songwriter_name"})[
        ["track_id", "songwriter_id", "songwriter_name"]
    ].drop_duplicates(subset=["track_id", "songwriter_id"])

    return producers, writers


def spotify_rows_to_bridge(tracks: pd.DataFrame) -> pd.DataFrame:
    """Explode Spotify's per-track artist lists into the track-artist bridge."""
    rows: list[dict] = []
    for _, t in tracks.iterrows():
        ids = t.get("artist_ids") or []
        for order, artist_id in enumerate(ids, start=1):
            rows.append(
                {
                    "track_id": t["track_id"],
                    "artist_id": artist_id,
                    "role": "primary_artist" if order == 1 else "featured_artist",
                    "billing_order": order,
                }
            )
    return pd.DataFrame(rows, columns=["track_id", "artist_id", "role", "billing_order"])
