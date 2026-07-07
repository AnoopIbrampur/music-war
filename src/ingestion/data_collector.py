"""Master data collector: orchestrates Spotify -> MusicBrainz -> Billboard.

Every stage checkpoints its output to ``data/processed/`` as parquet, so an
interrupted run resumes where it stopped instead of re-hitting APIs. Stages
degrade gracefully: missing credentials or a blocked source shrink coverage
but never crash the pipeline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import settings
from src.ingestion.billboard_client import get_hot100, match_to_tracks, summarize_chart_runs
from src.ingestion.musicbrainz_client import MusicBrainzClient
from src.ingestion.spotify_client import SpotifyClient, collect_seed_tracks

logger = logging.getLogger(__name__)

CHECKPOINTS = {
    "spotify": "checkpoint_spotify_tracks.parquet",
    "artists": "checkpoint_spotify_artists.parquet",
    "credits": "checkpoint_mb_credits.parquet",
    "billboard": "checkpoint_billboard.parquet",
}

_CREDIT_COLUMNS = ["track_id", "name", "mbid", "role"]


class DataCollector:
    """Runs the three-source ingestion pipeline with checkpoint/resume."""

    def __init__(self, out_dir: Path | None = None) -> None:
        self.out_dir = out_dir or settings.PROCESSED_DIR
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_log: dict = {"started_at": datetime.now(timezone.utc).isoformat(), "stages": {}}

    def _checkpoint_path(self, stage: str) -> Path:
        return self.out_dir / CHECKPOINTS[stage]

    def _load_or_run(self, stage: str, fn) -> pd.DataFrame:
        """Resume from a checkpoint if present, else run the stage and save."""
        path = self._checkpoint_path(stage)
        if path.exists():
            logger.info("Resuming %s from checkpoint %s", stage, path.name)
            return pd.read_parquet(path)
        df = fn()
        df.to_parquet(path, index=False)
        self.run_log["stages"][stage] = {"rows": len(df)}
        return df

    # ---------------------------------------------------------------- stages
    def collect_spotify(self) -> pd.DataFrame:
        def _run() -> pd.DataFrame:
            client = SpotifyClient()
            rows = collect_seed_tracks(client)
            df = pd.DataFrame(rows)
            ids = df["track_id"].dropna().tolist()
            feats = pd.DataFrame(client.get_audio_features(ids))
            if feats.empty:
                return df
            feats = feats.rename(columns={"id": "track_id"})
            return df.merge(feats, on="track_id", how="left")

        return self._load_or_run("spotify", _run)

    def collect_artists(self, tracks: pd.DataFrame) -> pd.DataFrame:
        """Full artist metadata (name, genres, followers, popularity)."""
        def _run() -> pd.DataFrame:
            client = SpotifyClient()
            unique_ids = sorted({aid for ids in tracks["artist_ids"] for aid in (ids or []) if aid})
            artists = client.get_artists(unique_ids)
            rows = [
                {
                    "artist_id": a["id"],
                    "artist_name": a["name"],
                    "primary_genre": (a.get("genres") or ["other"])[0],
                    "followers": (a.get("followers") or {}).get("total"),
                    "artist_popularity": a.get("popularity"),
                }
                for a in artists
            ]
            return pd.DataFrame(rows)

        return self._load_or_run("artists", _run)

    def collect_credits(self, tracks: pd.DataFrame) -> pd.DataFrame:
        def _run() -> pd.DataFrame:
            client = MusicBrainzClient()
            rows: list[dict] = []
            for _, t in tqdm(tracks.iterrows(), total=len(tracks), desc="MusicBrainz credits"):
                primary = (t.get("artist_names") or ["unknown"])[0]
                for credit in client.get_track_credits(str(t["clean_track_name"]), str(primary)):
                    rows.append({"track_id": t["track_id"], **credit})
            return pd.DataFrame(rows, columns=["track_id", "name", "mbid", "role"])

        return self._load_or_run("credits", _run)

    def collect_billboard(self, tracks: pd.DataFrame) -> pd.DataFrame:
        def _run() -> pd.DataFrame:
            weekly = get_hot100()
            runs = summarize_chart_runs(weekly)
            return match_to_tracks(runs, tracks)

        return self._load_or_run("billboard", _run)

    # ------------------------------------------------------------------ main
    def run(self, skip_musicbrainz: bool = False, skip_billboard: bool = False) -> pd.DataFrame:
        """Execute all stages and write the merged dataset to parquet.

        ``skip_musicbrainz`` avoids the 1 req/s credit enrichment crawl
        (useful when no MUSICBRAINZ_CONTACT is configured, or for a quick
        demo run). ``skip_billboard`` avoids the unofficial chart scrape.
        Both degrade gracefully: downstream code just sees empty frames.
        """
        tracks = self.collect_spotify()
        self.collect_artists(tracks)

        if skip_musicbrainz:
            credits = pd.DataFrame(columns=_CREDIT_COLUMNS)
        else:
            credits = self.collect_credits(tracks)

        merged = tracks if skip_billboard else self.collect_billboard(tracks)
        if skip_billboard:
            merged = merged.assign(billboard_peak_position=pd.NA, billboard_weeks_on_chart=pd.NA)

        producers = credits[credits["role"] == "producer"]
        writers = credits[credits["role"] == "songwriter"]
        merged.to_parquet(self.out_dir / "merged_tracks.parquet", index=False)
        producers.to_parquet(self.out_dir / "track_producers.parquet", index=False)
        writers.to_parquet(self.out_dir / "track_songwriters.parquet", index=False)

        self.run_log["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.run_log["totals"] = {
            "tracks": len(merged),
            "producer_credits": len(producers),
            "songwriter_credits": len(writers),
        }
        (self.out_dir / "collection_run_log.json").write_text(json.dumps(self.run_log, indent=2))
        logger.info("Collection complete: %s", self.run_log["totals"])
        return merged
