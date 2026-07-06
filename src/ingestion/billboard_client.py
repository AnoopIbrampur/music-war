"""Billboard chart ingestion.

Primary path uses the ``billboard.py`` library. If chart access is blocked,
falls back to a local Kaggle-style Hot 100 CSV (set BILLBOARD_FALLBACK_CSV
in .env). Either way the output shape is identical, so downstream code
never cares which source ran.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from rapidfuzz import fuzz

from config import settings

logger = logging.getLogger(__name__)

CHART_COLUMNS = [
    "chart_date", "rank", "track_name", "artist", "peak_position", "weeks_on_chart",
]


def fetch_hot100_weeks(start_year: int = settings.BILLBOARD_CHART_START_YEAR,
                       end_year: int = settings.BILLBOARD_CHART_END_YEAR,
                       step_weeks: int = 1) -> pd.DataFrame:
    """Pull weekly Hot 100 snapshots via billboard.py.

    ``step_weeks`` lets you subsample (e.g. every 4th week) for faster runs.
    """
    import billboard  # imported lazily; optional dependency at runtime

    rows: list[dict] = []
    current = date(start_year, 1, 7)
    end = min(date(end_year, 12, 31), date.today())
    while current <= end:
        try:
            chart = billboard.ChartData("hot-100", date=current.isoformat())
        except Exception:
            logger.exception("Failed to fetch Hot 100 for %s; continuing", current)
            current += timedelta(weeks=step_weeks)
            continue
        for entry in chart:
            rows.append(
                {
                    "chart_date": chart.date,
                    "rank": entry.rank,
                    "track_name": entry.title,
                    "artist": entry.artist,
                    "peak_position": entry.peakPos,
                    "weeks_on_chart": entry.weeks,
                }
            )
        current += timedelta(weeks=step_weeks)
    return pd.DataFrame(rows, columns=CHART_COLUMNS)


def load_fallback_csv(path: str) -> pd.DataFrame:
    """Load a Kaggle Hot 100 dump and normalise its columns to CHART_COLUMNS."""
    df = pd.read_csv(path)
    rename = {
        "date": "chart_date", "week_id": "chart_date",
        "song": "track_name", "title": "track_name",
        "performer": "artist",
        "peak_rank": "peak_position", "peak_pos": "peak_position",
        "weeks_on_board": "weeks_on_chart", "weeks": "weeks_on_chart",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    missing = [c for c in CHART_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Fallback CSV missing columns: {missing}")
    return df[CHART_COLUMNS]


def get_hot100() -> pd.DataFrame:
    """Best-effort Hot 100 history: live charts first, CSV fallback second."""
    try:
        df = fetch_hot100_weeks()
        if not df.empty:
            return df
    except Exception:
        logger.exception("billboard.py path failed")
    if settings.BILLBOARD_FALLBACK_CSV:
        logger.info("Using Billboard fallback CSV: %s", settings.BILLBOARD_FALLBACK_CSV)
        return load_fallback_csv(settings.BILLBOARD_FALLBACK_CSV)
    logger.warning("No Billboard data available — pipeline degrades to Spotify-only")
    return pd.DataFrame(columns=CHART_COLUMNS)


def summarize_chart_runs(weekly: pd.DataFrame) -> pd.DataFrame:
    """Collapse weekly snapshots into one row per (track, artist) chart run.

    Adds a coarse trajectory label comparing each song's first and last
    observed ranks: rising / falling / stable.
    """
    if weekly.empty:
        return pd.DataFrame(
            columns=["track_name", "artist", "peak_position", "weeks_on_chart", "trajectory"]
        )
    weekly = weekly.sort_values("chart_date")
    grouped = weekly.groupby(["track_name", "artist"], as_index=False).agg(
        peak_position=("rank", "min"),
        weeks_on_chart=("rank", "size"),
        first_rank=("rank", "first"),
        last_rank=("rank", "last"),
    )

    def _trajectory(row: pd.Series) -> str:
        delta = row["first_rank"] - row["last_rank"]  # positive = climbing
        if delta > 5:
            return "rising"
        if delta < -5:
            return "falling"
        return "stable"

    grouped["trajectory"] = grouped.apply(_trajectory, axis=1)
    return grouped.drop(columns=["first_rank", "last_rank"])


def match_to_tracks(chart_runs: pd.DataFrame, tracks: pd.DataFrame,
                    threshold: int = settings.FUZZY_MATCH_THRESHOLD) -> pd.DataFrame:
    """Fuzzy-join chart runs onto the Spotify track table.

    Blocks on the first letter of the track name to keep the comparison
    count tractable, then requires both title and artist to clear the
    similarity threshold.
    """
    if chart_runs.empty or tracks.empty:
        return tracks.assign(billboard_peak_position=pd.NA, billboard_weeks_on_chart=pd.NA)

    chart_runs = chart_runs.copy()
    chart_runs["_block"] = chart_runs["track_name"].str[:1].str.lower()
    matches: dict[str, tuple[int, int]] = {}

    for _, track in tracks.iterrows():
        name = str(track["track_name"])
        artist = str(track.get("primary_artist_name", track.get("artist_names", "")))
        block = chart_runs[chart_runs["_block"] == name[:1].lower()]
        best_score, best_row = 0.0, None
        for _, run in block.iterrows():
            title_score = fuzz.token_set_ratio(name.lower(), run["track_name"].lower())
            if title_score < threshold:
                continue
            artist_score = fuzz.token_set_ratio(artist.lower(), run["artist"].lower())
            score = (title_score + artist_score) / 2
            if artist_score >= threshold and score > best_score:
                best_score, best_row = score, run
        if best_row is not None:
            matches[track["track_id"]] = (
                int(best_row["peak_position"]), int(best_row["weeks_on_chart"])
            )

    out = tracks.copy()
    out["billboard_peak_position"] = out["track_id"].map(
        lambda t: matches.get(t, (pd.NA, pd.NA))[0]
    )
    out["billboard_weeks_on_chart"] = out["track_id"].map(
        lambda t: matches.get(t, (pd.NA, pd.NA))[1]
    )
    logger.info("Matched %d/%d tracks to Billboard runs", len(matches), len(tracks))
    return out
