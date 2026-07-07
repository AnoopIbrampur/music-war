"""WAR calculation: ridge (primary) and lasso (comparison) regressions on
the sparse design matrix.

The coefficient on a person's indicator column is their WAR: the expected
lift in composite success score for a track they're on, above a
replacement-level person, controlling for genre, era, sound profile, and
everyone else on the track. The model intercept plus control effects is
the replacement-level baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LassoCV, Ridge, RidgeCV
from sklearn.model_selection import train_test_split

from config import settings
from src.modeling.sparse_matrix_builder import DesignMatrix

logger = logging.getLogger(__name__)


@dataclass
class WARResult:
    war_table: pd.DataFrame        # one row per eligible person
    ridge_model: RidgeCV
    lasso_model: LassoCV
    replacement_level: float
    split: dict                    # train/test indices for the evaluator


def _stratify_labels(dm: DesignMatrix, tracks: pd.DataFrame | None) -> np.ndarray | None:
    """Stratify the split by genre when the genre column is available."""
    if tracks is None or "primary_genre" not in tracks.columns:
        return None
    labels = tracks.set_index("track_id").loc[dm.track_ids, "primary_genre"].to_numpy()
    values, counts = np.unique(labels, return_counts=True)
    if (counts < 2).any():   # sklearn refuses to stratify singleton classes
        return None
    return labels


def fit_war_models(dm: DesignMatrix, tracks: pd.DataFrame | None = None) -> WARResult:
    """Fit ridge + lasso and assemble the WAR table."""
    idx = np.arange(dm.X.shape[0])
    train_idx, test_idx = train_test_split(
        idx,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
        stratify=_stratify_labels(dm, tracks),
    )
    X_train, y_train = dm.X[train_idx], dm.y[train_idx]

    ridge = RidgeCV(alphas=settings.RIDGE_ALPHAS)
    ridge.fit(X_train, y_train)
    logger.info("RidgeCV chose alpha=%.3g", ridge.alpha_)

    lasso = LassoCV(alphas=settings.LASSO_ALPHAS, cv=5, random_state=settings.RANDOM_STATE,
                    max_iter=5000)
    lasso.fit(X_train.toarray() if X_train.shape[1] < 5000 else X_train, y_train)

    person_idx = dm.person_slice
    rows = []
    for i in person_idx:
        meta = dm.person_columns[dm.columns[i]]
        rows.append(
            {
                "entity_id": meta["entity_id"],
                "name": meta["name"],
                "role": meta["role"],
                "n_tracks": meta["n_tracks"],
                "war_per_track": float(ridge.coef_[i]),
                "total_war": float(ridge.coef_[i]) * meta["n_tracks"],
                "lasso_war": float(lasso.coef_[i]),
                "survives_lasso": bool(abs(lasso.coef_[i]) > 1e-8),
            }
        )
    table = pd.DataFrame(rows)
    table["percentile_rank"] = (
        table.groupby("role")["war_per_track"].rank(pct=True).mul(100).round(1)
    )
    replacement_level = float(ridge.intercept_)

    return WARResult(
        war_table=table.sort_values("war_per_track", ascending=False).reset_index(drop=True),
        ridge_model=ridge,
        lasso_model=lasso,
        replacement_level=replacement_level,
        split={"train_idx": train_idx, "test_idx": test_idx},
    )


def bootstrap_confidence_intervals(dm: DesignMatrix, alpha_value: float,
                                   n_iterations: int = settings.BOOTSTRAP_ITERATIONS,
                                   random_state: int = settings.RANDOM_STATE) -> pd.DataFrame:
    """Bootstrap 95% CIs for every person coefficient.

    Resamples tracks with replacement and refits a Ridge at the CV-chosen
    alpha each time. Wide intervals flag people whose WAR rests on a
    handful of tracks.
    """
    rng = np.random.default_rng(random_state)
    person_idx = np.array(dm.person_slice)
    coefs = np.empty((n_iterations, len(person_idx)))

    n = dm.X.shape[0]
    for b in range(n_iterations):
        sample = rng.integers(0, n, size=n)
        model = Ridge(alpha=alpha_value)
        model.fit(dm.X[sample], dm.y[sample])
        coefs[b] = model.coef_[person_idx]

    lo, hi = np.percentile(coefs, [2.5, 97.5], axis=0)
    return pd.DataFrame(
        {
            "entity_id": [dm.person_columns[dm.columns[i]]["entity_id"] for i in person_idx],
            "war_ci_low": lo,
            "war_ci_high": hi,
        }
    )


def enrich_artist_war(war_table: pd.DataFrame, tracks: pd.DataFrame,
                      track_artists: pd.DataFrame, artists: pd.DataFrame | None = None
                      ) -> pd.DataFrame:
    """Augment artist rows with popularity, consistency, and overperformance.

    Adds columns used by the dashboard's "Beyond Popularity" analysis:
      * avg_popularity   – mean raw Spotify popularity of the artist's tracks
      * avg_track_score  – mean composite success score
      * score_std        – std of composite scores (low = consistent hitmaker)
      * overperformance  – WAR minus what their popularity alone predicts
                           (positive = overdelivers for their fame level)
      * genre            – primary genre (for per-genre leaderboards)
    Non-artist rows keep NaN for these columns.
    """
    war = war_table.copy()
    merged = track_artists.merge(
        tracks[["track_id", "spotify_popularity", "composite_success_score"]],
        on="track_id", how="left",
    )
    agg = merged.groupby("artist_id").agg(
        avg_popularity=("spotify_popularity", "mean"),
        avg_track_score=("composite_success_score", "mean"),
        score_std=("composite_success_score", "std"),
    )
    for col in ["avg_popularity", "avg_track_score", "score_std"]:
        war[col] = war["entity_id"].map(agg[col])

    if artists is not None and "primary_genre" in artists.columns:
        genre_map = dict(zip(artists["artist_id"], artists["primary_genre"]))
        war["genre"] = war["entity_id"].map(genre_map)

    # Overperformance: residual of WAR against a linear fit on avg popularity,
    # computed over eligible artists only.
    mask = (war["role"] == "artist") & war["avg_popularity"].notna()
    if mask.sum() >= 2:
        x = war.loc[mask, "avg_popularity"].to_numpy()
        y = war.loc[mask, "war_per_track"].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        war.loc[mask, "overperformance"] = y - (slope * x + intercept)
    return war


def interpret(row: pd.Series, replacement_level: float) -> str:
    """Plain-language reading of one WAR table row."""
    direction = "adds" if row["war_per_track"] >= 0 else "costs"
    return (
        f"{row['name']} {direction} {abs(row['war_per_track']):.1f} points to a track's "
        f"success score versus a replacement-level {row['role']} "
        f"(baseline {replacement_level:.1f}), over {row['n_tracks']} tracks — "
        f"{row['percentile_rank']:.0f}th percentile among eligible {row['role']}s."
    )
