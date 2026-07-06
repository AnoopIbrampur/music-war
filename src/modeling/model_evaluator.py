"""Model evaluation: holdout metrics, CV, residual bias checks, and the
people-vs-structure baseline comparison.

The baseline model uses only structural controls (genre, era, audio, crew
size). The gap between baseline and full R² quantifies how much individual
talent explains beyond structural factors — the headline number of the
whole project.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from src.modeling.sparse_matrix_builder import DesignMatrix
from src.modeling.war_calculator import WARResult

logger = logging.getLogger(__name__)


def evaluate(dm: DesignMatrix, result: WARResult,
             tracks: pd.DataFrame | None = None) -> dict:
    """Compute the full evaluation report as a JSON-serialisable dict."""
    train_idx, test_idx = result.split["train_idx"], result.split["test_idx"]
    X_test, y_test = dm.X[test_idx], dm.y[test_idx]

    pred = result.ridge_model.predict(X_test)
    metrics: dict = {
        "ridge_alpha": float(result.ridge_model.alpha_),
        "r2_test": float(r2_score(y_test, pred)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, pred))),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_person_columns": len(dm.person_slice),
        "replacement_level": result.replacement_level,
    }

    # 5-fold CV on the training split
    cv = cross_val_score(
        Ridge(alpha=result.ridge_model.alpha_), dm.X[train_idx], dm.y[train_idx],
        cv=5, scoring="r2",
    )
    metrics["cv_r2_mean"] = float(cv.mean())
    metrics["cv_r2_std"] = float(cv.std())

    # Baseline: controls only — no artist/producer/songwriter columns
    person = set(dm.person_slice)
    control_idx = [i for i in range(dm.X.shape[1]) if i not in person]
    baseline = Ridge(alpha=result.ridge_model.alpha_)
    baseline.fit(dm.X[train_idx][:, control_idx], dm.y[train_idx])
    base_pred = baseline.predict(X_test[:, control_idx])
    metrics["baseline_r2_test"] = float(r2_score(y_test, base_pred))
    metrics["talent_lift_r2"] = metrics["r2_test"] - metrics["baseline_r2_test"]

    # Residuals by genre — systematic bias check
    if tracks is not None and "primary_genre" in tracks.columns:
        genres = tracks.set_index("track_id").loc[
            [dm.track_ids[i] for i in test_idx], "primary_genre"
        ].to_numpy()
        resid = pd.DataFrame({"genre": genres, "residual": y_test - pred})
        by_genre = resid.groupby("genre")["residual"].agg(["mean", "std", "count"])
        metrics["residuals_by_genre"] = {
            g: {"mean": round(float(r["mean"]), 3), "std": round(float(r["std"]), 3),
                "count": int(r["count"])}
            for g, r in by_genre.iterrows()
        }

    logger.info(
        "Eval: R²=%.3f (baseline %.3f, talent lift %.3f), RMSE=%.2f",
        metrics["r2_test"], metrics["baseline_r2_test"],
        metrics["talent_lift_r2"], metrics["rmse_test"],
    )
    return metrics
