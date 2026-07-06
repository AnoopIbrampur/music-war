"""End-to-end pipeline: ingest -> clean -> engineer -> load DB -> model.

Run with real APIs (requires .env credentials):
    python -m src.pipeline

Run in demo mode on synthetic data (no credentials needed):
    python -m src.pipeline --demo
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from config import settings
from src.database.db_manager import DBManager, load_star_schema
from src.ingestion import synthetic
from src.modeling.model_evaluator import evaluate
from src.modeling.sparse_matrix_builder import build_design_matrix
from src.modeling.war_calculator import bootstrap_confidence_intervals, fit_war_models
from src.processing.cleaner import clean_tracks, missing_data_report
from src.processing.feature_engineer import (
    engineer_artist_features,
    engineer_collab_features,
    engineer_track_features,
)
from src.processing.transformer import build_producer_dim, build_songwriter_dim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("pipeline")


def load_demo_data():
    ds = synthetic.generate()
    return ds.tracks, ds.track_artists, ds.track_producers, ds.track_songwriters, ds.artists


def load_real_data():
    from src.ingestion.data_collector import DataCollector
    from src.processing.transformer import credits_to_bridges, spotify_rows_to_bridge

    collector = DataCollector()
    merged = collector.run()
    credits = pd.read_parquet(settings.PROCESSED_DIR / "checkpoint_mb_credits.parquet")
    track_producers, track_songwriters = credits_to_bridges(credits)
    track_artists = spotify_rows_to_bridge(merged)
    artists = pd.DataFrame()  # enriched separately via SpotifyClient.get_artists
    return merged, track_artists, track_producers, track_songwriters, artists


def run(demo: bool = False, bootstrap: bool = False) -> dict:
    tracks, track_artists, track_producers, track_songwriters, artists = (
        load_demo_data() if demo else load_real_data()
    )

    # ---- processing
    tracks = clean_tracks(tracks)
    missing_data_report(tracks).to_csv(settings.PROCESSED_DIR / "missing_data_report.csv", index=False)
    tracks = engineer_track_features(tracks)
    tracks = engineer_collab_features(tracks, track_artists, track_producers)
    artists = engineer_artist_features(track_artists, tracks, artists)

    # keep bridges consistent with the cleaned track set
    keep = set(tracks["track_id"])
    track_artists = track_artists[track_artists["track_id"].isin(keep)]
    track_producers = track_producers[track_producers["track_id"].isin(keep)]
    track_songwriters = track_songwriters[track_songwriters["track_id"].isin(keep)]

    # ---- database
    db = DBManager()
    producers_dim = build_producer_dim(track_producers)
    songwriters_dim = build_songwriter_dim(track_songwriters)
    counts = load_star_schema(
        db, tracks, artists, track_artists, producers_dim, songwriters_dim,
        track_producers, track_songwriters,
    )
    logger.info("DB row counts: %s", counts)

    # ---- modeling
    dm = build_design_matrix(
        tracks, track_artists, track_producers, track_songwriters,
        artist_names=dict(zip(artists["artist_id"], artists["artist_name"])),
        producer_names=dict(zip(producers_dim["producer_id"], producers_dim["producer_name"])),
        songwriter_names=dict(zip(songwriters_dim["songwriter_id"], songwriters_dim["songwriter_name"])),
    )
    result = fit_war_models(dm, tracks)
    war = result.war_table
    if bootstrap:
        cis = bootstrap_confidence_intervals(dm, result.ridge_model.alpha_)
        war = war.merge(cis, on="entity_id", how="left")
    metrics = evaluate(dm, result, tracks)

    # ---- outputs for the dashboard
    tracks.to_parquet(settings.PROCESSED_DIR / "tracks_features.parquet", index=False)
    track_artists.to_parquet(settings.PROCESSED_DIR / "bridge_track_artist.parquet", index=False)
    artists.to_parquet(settings.PROCESSED_DIR / "dim_artist.parquet", index=False)
    war.to_parquet(settings.MODELS_DIR / "war_results.parquet", index=False)
    (settings.MODELS_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Pipeline complete — %d tracks, %d WAR entities", len(tracks), len(war))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music WAR pipeline")
    parser.add_argument("--demo", action="store_true", help="use synthetic data (no API keys)")
    parser.add_argument("--bootstrap", action="store_true", help="compute bootstrap CIs (slower)")
    args = parser.parse_args()
    run(demo=args.demo, bootstrap=args.bootstrap)
