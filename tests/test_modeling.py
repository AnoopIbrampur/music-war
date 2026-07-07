"""Tests for the sparse matrix, WAR model, evaluator, and database layer.

The headline test: on synthetic data with *known* planted talent effects,
the ridge WAR estimates must correlate strongly with the ground truth.
"""

import numpy as np
import pandas as pd
import pytest

from src.database.db_manager import DBManager, load_star_schema
from src.database import schema
from src.ingestion import synthetic
from src.modeling.model_evaluator import evaluate
from src.modeling.sparse_matrix_builder import build_design_matrix, eligible_entities
from src.modeling.war_calculator import enrich_artist_war, fit_war_models, interpret
from src.processing.cleaner import clean_tracks
from src.processing.feature_engineer import engineer_collab_features, engineer_track_features
from src.processing.transformer import build_producer_dim, build_songwriter_dim


@pytest.fixture(scope="module")
def dataset():
    return synthetic.generate(n_tracks=1500, n_artists=120, n_producers=50,
                              n_songwriters=60, seed=7)


@pytest.fixture(scope="module")
def prepared(dataset):
    tracks = clean_tracks(dataset.tracks)
    tracks = engineer_track_features(tracks)
    tracks = engineer_collab_features(tracks, dataset.track_artists, dataset.track_producers)
    return tracks


@pytest.fixture(scope="module")
def design(prepared, dataset):
    return build_design_matrix(
        prepared, dataset.track_artists, dataset.track_producers, dataset.track_songwriters
    )


@pytest.fixture(scope="module")
def war_result(design, prepared):
    return fit_war_models(design, prepared)


class TestSyntheticGenerator:
    def test_shapes(self, dataset):
        assert len(dataset.tracks) == 1500
        assert dataset.track_artists["track_id"].nunique() == 1500

    def test_every_track_has_primary_artist(self, dataset):
        primaries = dataset.track_artists[dataset.track_artists["role"] == "primary_artist"]
        assert primaries["track_id"].nunique() == 1500


class TestSparseMatrix:
    def test_row_count_matches_tracks(self, design, prepared):
        assert design.X.shape[0] == len(prepared)

    def test_columns_aligned(self, design):
        assert design.X.shape[1] == len(design.columns)

    def test_matrix_is_sparse(self, design):
        density = design.X.nnz / (design.X.shape[0] * design.X.shape[1])
        assert density < 0.10

    def test_eligibility_threshold(self, dataset, prepared):
        counts = eligible_entities(dataset.track_artists, "artist_id", prepared, 5)
        assert (counts >= 5).all()

    def test_featured_weight_applied(self, design, dataset, prepared):
        # find a track row with a featured artist whose column is eligible
        feats = dataset.track_artists[dataset.track_artists["role"] == "featured_artist"]
        track_pos = {t: i for i, t in enumerate(design.track_ids)}
        col_pos = {design.person_columns[c]["entity_id"]: i
                   for i, c in enumerate(design.columns) if c in design.person_columns
                   and design.person_columns[c]["role"] == "artist"}
        for _, r in feats.iterrows():
            if r["track_id"] in track_pos and r["artist_id"] in col_pos:
                val = design.X[track_pos[r["track_id"]], col_pos[r["artist_id"]]]
                assert val == pytest.approx(0.5)
                return
        pytest.skip("no eligible featured artist in sample")

    def test_person_metadata_counts(self, design):
        for meta in design.person_columns.values():
            assert meta["n_tracks"] >= 5


class TestWARModel:
    def test_recovers_planted_artist_effects(self, war_result, dataset):
        """Estimated WAR must track the ground-truth skills we planted."""
        artists = war_result.war_table.query("role == 'artist'")
        truth = artists["entity_id"].map(dataset.true_effects["artist"])
        corr = np.corrcoef(artists["war_per_track"], truth)[0, 1]
        assert corr > 0.8, f"WAR/truth correlation too low: {corr:.2f}"

    def test_recovers_planted_producer_effects(self, war_result, dataset):
        producers = war_result.war_table.query("role == 'producer'")
        truth = producers["entity_id"].map(dataset.true_effects["producer"])
        corr = np.corrcoef(producers["war_per_track"], truth)[0, 1]
        assert corr > 0.7

    def test_percentiles_within_range(self, war_result):
        assert war_result.war_table["percentile_rank"].between(0, 100).all()

    def test_lasso_sparser_than_ridge(self, war_result):
        survived = war_result.war_table["survives_lasso"].mean()
        assert survived < 1.0  # lasso must zero out someone

    def test_interpret_string(self, war_result):
        row = war_result.war_table.iloc[0]
        text = interpret(row, war_result.replacement_level)
        assert row["name"] in text and "replacement" in text

    def test_enrich_artist_war_adds_columns(self, war_result, prepared, dataset):
        enriched = enrich_artist_war(
            war_result.war_table, prepared, dataset.track_artists, dataset.artists
        )
        for col in ["avg_popularity", "score_std", "overperformance", "genre"]:
            assert col in enriched.columns
        artists = enriched[enriched["role"] == "artist"]
        assert artists["avg_popularity"].notna().all()
        # overperformance is a mean-zero residual around the popularity trend
        assert abs(artists["overperformance"].mean()) < 1.0


class TestEvaluator:
    def test_metrics_present_and_sane(self, design, war_result, prepared):
        metrics = evaluate(design, war_result, prepared)
        assert 0 < metrics["r2_test"] <= 1
        assert metrics["rmse_test"] > 0
        assert metrics["n_train"] > metrics["n_test"]

    def test_people_beat_structure_only_baseline(self, design, war_result, prepared):
        metrics = evaluate(design, war_result, prepared)
        assert metrics["talent_lift_r2"] > 0.05, (
            "adding people columns should explain meaningfully more variance"
        )

    def test_residuals_by_genre_reported(self, design, war_result, prepared):
        metrics = evaluate(design, war_result, prepared)
        assert len(metrics["residuals_by_genre"]) >= 5


class TestDatabase:
    def test_schema_roundtrip(self, prepared, dataset):
        db = DBManager("sqlite:///:memory:")
        producers = build_producer_dim(dataset.track_producers)
        writers = build_songwriter_dim(dataset.track_songwriters)
        keep = set(prepared["track_id"])
        counts = load_star_schema(
            db, prepared, dataset.artists,
            dataset.track_artists[dataset.track_artists["track_id"].isin(keep)],
            producers, writers,
            dataset.track_producers[dataset.track_producers["track_id"].isin(keep)].drop_duplicates(["track_id", "producer_id"]),
            dataset.track_songwriters[dataset.track_songwriters["track_id"].isin(keep)].drop_duplicates(["track_id", "songwriter_id"]),
        )
        assert counts["dim_track"] == len(prepared)
        read_back = db.read_table(schema.DimTrack)
        assert len(read_back) == len(prepared)

    def test_replace_table_is_idempotent(self, prepared):
        db = DBManager("sqlite:///:memory:")
        db.create_schema()
        db.replace_table(schema.DimTrack, prepared)
        db.replace_table(schema.DimTrack, prepared)  # second load must not duplicate
        assert len(db.read_table(schema.DimTrack)) == len(prepared)

    def test_bridge_relationship_join(self, prepared, dataset):
        db = DBManager("sqlite:///:memory:")
        db.create_schema()
        db.replace_table(schema.DimTrack, prepared)
        db.replace_table(schema.DimArtist, dataset.artists)
        keep = set(prepared["track_id"])
        db.replace_table(
            schema.BridgeTrackArtist,
            dataset.track_artists[dataset.track_artists["track_id"].isin(keep)],
        )
        bridge = db.read_table(schema.BridgeTrackArtist)
        assert set(bridge["role"].unique()) <= {"primary_artist", "featured_artist"}
        assert bridge["track_id"].isin(keep).all()
