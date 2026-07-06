"""Sparse design matrix construction for the WAR regression.

Each row is a track; columns are binary indicators for every *eligible*
artist / producer / songwriter, plus control columns (genre, era, buckets,
crew size, audio cluster). Below-threshold people carry no column — they
blend into the intercept, which IS the replacement level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from config import settings

logger = logging.getLogger(__name__)

CONTROL_CATEGORICALS = ["primary_genre", "era", "duration_bucket", "tempo_bucket", "audio_cluster"]
CONTROL_NUMERICS = ["is_explicit", "num_artists_on_track"]


@dataclass
class DesignMatrix:
    """Sparse X, target y, and the metadata needed to interpret coefficients."""

    X: sparse.csr_matrix
    y: np.ndarray
    track_ids: list[str]
    columns: list[str]              # human-readable column names, index-aligned with X
    person_columns: dict[str, dict] # col name -> {role, entity_id, name, n_tracks}

    @property
    def person_slice(self) -> list[int]:
        return [i for i, c in enumerate(self.columns) if c in self.person_columns]


def eligible_entities(bridge: pd.DataFrame, id_col: str, tracks: pd.DataFrame,
                      min_tracks: int) -> pd.Series:
    """Entities with >= min_tracks credits on sufficiently popular tracks."""
    popular = tracks.loc[
        tracks["spotify_popularity"] >= settings.MIN_POPULARITY_FOR_ELIGIBILITY, "track_id"
    ]
    counts = bridge[bridge["track_id"].isin(popular)].groupby(id_col)["track_id"].nunique()
    return counts[counts >= min_tracks]


def _person_block(tracks: pd.DataFrame, bridge: pd.DataFrame, id_col: str, name_map: dict,
                  eligible: pd.Series, role: str, weights: pd.Series | None = None
                  ) -> tuple[sparse.csr_matrix, list[str], dict[str, dict]]:
    """Build one sparse block of person-indicator columns."""
    track_index = {t: i for i, t in enumerate(tracks["track_id"])}
    col_index = {e: j for j, e in enumerate(eligible.index)}

    rows, cols, vals = [], [], []
    linked = bridge[bridge[id_col].isin(col_index) & bridge["track_id"].isin(track_index)]
    for _, r in linked.iterrows():
        rows.append(track_index[r["track_id"]])
        cols.append(col_index[r[id_col]])
        vals.append(float(weights.get(r.name, 1.0)) if weights is not None else 1.0)

    block = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(len(track_index), len(col_index))
    )
    names = [f"{role}::{e}" for e in col_index]
    meta = {
        f"{role}::{e}": {
            "role": role,
            "entity_id": e,
            "name": name_map.get(e, str(e)),
            "n_tracks": int(eligible[e]),
        }
        for e in col_index
    }
    return block, names, meta


def build_design_matrix(tracks: pd.DataFrame, track_artists: pd.DataFrame,
                        track_producers: pd.DataFrame, track_songwriters: pd.DataFrame,
                        artist_names: dict | None = None, producer_names: dict | None = None,
                        songwriter_names: dict | None = None,
                        target: str = "composite_success_score") -> DesignMatrix:
    """Assemble the full sparse design matrix.

    Featured artists get FEATURED_BILLING_WEIGHT (0.5) instead of 1.0 so a
    guest verse counts for less than top billing.
    """
    tracks = tracks.reset_index(drop=True)

    # --- eligibility
    elig_artists = eligible_entities(track_artists, "artist_id", tracks, settings.MIN_TRACKS_ARTIST)
    elig_producers = eligible_entities(track_producers, "producer_id", tracks, settings.MIN_TRACKS_PRODUCER)
    elig_writers = eligible_entities(track_songwriters, "songwriter_id", tracks, settings.MIN_TRACKS_SONGWRITER)
    logger.info(
        "Eligible: %d artists, %d producers, %d songwriters",
        len(elig_artists), len(elig_producers), len(elig_writers),
    )

    # --- person blocks (artists get billing weights)
    billing = track_artists["role"].map(
        {"primary_artist": 1.0, "featured_artist": settings.FEATURED_BILLING_WEIGHT}
    ).fillna(1.0)
    a_block, a_names, a_meta = _person_block(
        tracks, track_artists, "artist_id", artist_names or {}, elig_artists, "artist", billing
    )
    p_block, p_names, p_meta = _person_block(
        tracks, track_producers, "producer_id", producer_names or {}, elig_producers, "producer"
    )
    w_block, w_names, w_meta = _person_block(
        tracks, track_songwriters, "songwriter_id", songwriter_names or {}, elig_writers, "songwriter"
    )

    # --- control block (dense one-hots + numerics, converted to sparse)
    cats = [c for c in CONTROL_CATEGORICALS if c in tracks.columns]
    controls = pd.get_dummies(
        tracks[cats].astype(str), prefix=cats, dtype=float
    )
    for col in CONTROL_NUMERICS:
        if col in tracks.columns:
            controls[col] = tracks[col].astype(float)
    c_block = sparse.csr_matrix(controls.to_numpy())

    X = sparse.hstack([a_block, p_block, w_block, c_block], format="csr")
    columns = a_names + p_names + w_names + list(controls.columns)
    person_columns = {**a_meta, **p_meta, **w_meta}
    y = tracks[target].to_numpy(dtype=float)

    logger.info("Design matrix: %s, density %.4f%%", X.shape, 100 * X.nnz / max(1, X.shape[0] * X.shape[1]))
    return DesignMatrix(
        X=X, y=y, track_ids=tracks["track_id"].tolist(),
        columns=columns, person_columns=person_columns,
    )
