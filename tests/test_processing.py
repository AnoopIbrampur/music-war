"""Tests for cleaning and feature engineering."""

import pandas as pd
import pytest

from src.processing.cleaner import (
    deduplicate_tracks, flag_special_albums, missing_data_report,
    rollup_genre, standardize_artist_name,
)
from src.processing.feature_engineer import (
    compute_composite_score, duration_bucket, era_bucket, season_released, tempo_bucket,
)


class TestStandardizeArtistName:
    @pytest.mark.parametrize(
        "a,b",
        [
            ("The Weeknd", "Weeknd"),
            ("JAY-Z", "Jay Z"),
            ("Jay-Z", "jay z"),
            ("Beyoncé", "Beyonce"),
            ("  Drake  ", "drake"),
        ],
    )
    def test_variants_collapse(self, a, b):
        assert standardize_artist_name(a) == standardize_artist_name(b)

    def test_distinct_artists_stay_distinct(self):
        assert standardize_artist_name("Drake") != standardize_artist_name("Future")

    def test_non_string_returns_empty(self):
        assert standardize_artist_name(None) == ""


class TestGenreRollup:
    @pytest.mark.parametrize(
        "raw,parent",
        [
            ("canadian hip hop", "hip_hop"),
            ("uk drill", "hip_hop"),
            ("dance pop", "electronic"),
            ("k-pop girl group", "kpop"),
            ("alt rock", "indie"),
            ("classic rock", "rock"),
            ("neo soul", "rnb"),
            ("reggaeton flow", "latin"),
        ],
    )
    def test_rollup(self, raw, parent):
        assert rollup_genre(raw) == parent

    def test_unknown_and_none(self):
        assert rollup_genre("obscure microgenre xyz") == "other"
        assert rollup_genre(None) == "other"


class TestDeduplication:
    def test_exact_id_dupes_dropped(self):
        df = pd.DataFrame(
            {
                "track_id": ["a", "a", "b"],
                "track_name": ["X", "X", "Y"],
                "spotify_popularity": [50, 50, 60],
            }
        )
        assert len(deduplicate_tracks(df)) == 2

    def test_near_dupes_keep_most_popular(self):
        df = pd.DataFrame(
            {
                "track_id": ["a", "b"],
                "track_name": ["Same Song", "same song "],
                "primary_artist_name": ["The Weeknd", "Weeknd"],
                "spotify_popularity": [40, 90],
            }
        )
        out = deduplicate_tracks(df)
        assert len(out) == 1
        assert out.iloc[0]["spotify_popularity"] == 90


class TestSpecialAlbumFlags:
    def test_flags(self):
        df = pd.DataFrame(
            {
                "track_id": ["a", "b", "c"],
                "track_name": ["Song", "Jingle Bells", "Tune"],
                "album_name": ["Greatest Hits Vol 1", "Christmas Party", "Normal Album"],
                "album_type": ["compilation", "album", "album"],
            }
        )
        out = flag_special_albums(df)
        assert out.iloc[0]["is_compilation"]
        assert out.iloc[1]["is_holiday"]
        assert not out.iloc[2][["is_compilation", "is_soundtrack", "is_holiday"]].any()


class TestBuckets:
    def test_era_buckets(self):
        assert era_bucket(1995) == "90s"
        assert era_bucket(2003) == "early_2000s"
        assert era_bucket(2007) == "late_2000s"
        assert era_bucket(2015) == "2010s"
        assert era_bucket(2023) == "early_2020s"
        assert era_bucket(1985) == "pre_90s"

    def test_tempo_buckets(self):
        assert tempo_bucket(85) == "slow"
        assert tempo_bucket(115) == "mid"
        assert tempo_bucket(150) == "fast"
        assert tempo_bucket(None) == "unknown"

    def test_duration_buckets(self):
        assert duration_bucket(150_000) == "short"     # 2.5 min
        assert duration_bucket(200_000) == "standard"  # 3.3 min
        assert duration_bucket(300_000) == "long"      # 5 min

    def test_season(self):
        assert season_released("2020-07-15") == "Q3"
        assert season_released("2020-01-01") == "Q1"
        assert season_released(None) == "unknown"


class TestCompositeScore:
    def _tracks(self):
        return pd.DataFrame(
            {
                "track_id": ["hit", "flop", "uncharted"],
                "spotify_popularity": [90.0, 20.0, 90.0],
                "billboard_peak_position": [1, None, None],
                "billboard_weeks_on_chart": [30, None, None],
                "release_year": [2020, 2020, 2020],
            }
        )

    def test_score_within_bounds(self):
        out = compute_composite_score(self._tracks())
        assert out["composite_success_score"].between(0, 100).all()

    def test_charted_hit_beats_uncharted_twin(self):
        out = compute_composite_score(self._tracks()).set_index("track_id")
        assert (
            out.loc["hit", "composite_success_score"]
            > out.loc["uncharted", "composite_success_score"]
        )

    def test_popularity_ordering_preserved(self):
        out = compute_composite_score(self._tracks()).set_index("track_id")
        assert (
            out.loc["uncharted", "composite_success_score"]
            > out.loc["flop", "composite_success_score"]
        )


class TestMissingReport:
    def test_rates(self):
        df = pd.DataFrame({"full": [1, 2], "half": [1, None]})
        report = missing_data_report(df).set_index("column")
        assert report.loc["half", "missing_rate"] == 50.0
        assert report.loc["full", "missing_count"] == 0
