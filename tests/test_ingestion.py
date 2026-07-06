"""Tests for the ingestion layer: parsing, matching, batching, fallbacks."""

import pandas as pd
import pytest

from src.ingestion.billboard_client import (
    CHART_COLUMNS, load_fallback_csv, match_to_tracks, summarize_chart_runs,
)
from src.ingestion.musicbrainz_client import extract_credits, is_good_match
from src.ingestion.spotify_client import SpotifyClient, batch, parse_featured_artists


class TestParseFeaturedArtists:
    def test_feat_dot(self):
        clean, feats = parse_featured_artists("Sicko Mode (feat. Drake)")
        assert clean == "Sicko Mode"
        assert feats == ["Drake"]

    def test_ft_no_parens(self):
        clean, feats = parse_featured_artists("No Guidance ft. Drake")
        assert clean == "No Guidance"
        assert feats == ["Drake"]

    def test_with_keyword(self):
        clean, feats = parse_featured_artists("Stay (with Justin Bieber)")
        assert clean == "Stay"
        assert feats == ["Justin Bieber"]

    def test_multiple_features(self):
        _, feats = parse_featured_artists("Forever (feat. Drake, Kanye West & Eminem)")
        assert feats == ["Drake", "Kanye West", "Eminem"]

    def test_no_feature(self):
        clean, feats = parse_featured_artists("Bohemian Rhapsody")
        assert clean == "Bohemian Rhapsody"
        assert feats == []

    def test_square_brackets(self):
        clean, feats = parse_featured_artists("Song Title [feat. Someone]")
        assert clean == "Song Title"
        assert feats == ["Someone"]


class TestBatching:
    def test_exact_batches(self):
        chunks = list(batch(list(range(200)), 100))
        assert len(chunks) == 2
        assert all(len(c) == 100 for c in chunks)

    def test_remainder_batch(self):
        chunks = list(batch(list(range(105)), 100))
        assert [len(c) for c in chunks] == [100, 5]

    def test_empty(self):
        assert list(batch([], 100)) == []


class TestTrackToRow:
    def test_flattens_track_object(self):
        track = {
            "id": "abc", "name": "Song (feat. Guest)", "duration_ms": 200000,
            "explicit": True, "popularity": 80,
            "album": {"id": "al1", "name": "Album", "album_type": "album",
                      "release_date": "2020-05-01"},
            "artists": [{"id": "a1", "name": "Main"}, {"id": "a2", "name": "Guest"}],
        }
        row = SpotifyClient.track_to_row(track)
        assert row["track_id"] == "abc"
        assert row["clean_track_name"] == "Song"
        assert row["featured_from_title"] == ["Guest"]
        assert row["artist_ids"] == ["a1", "a2"]
        assert row["explicit"] is True


class TestMusicBrainz:
    def test_extract_producer_and_writer_credits(self):
        recording = {
            "artist-relation-list": [
                {"type": "producer", "artist": {"name": "Rick Rubin", "id": "m1"}},
                {"type": "mix", "artist": {"name": "Some Engineer", "id": "m2"}},
            ],
            "work-relation-list": [
                {"work": {"artist-relation-list": [
                    {"type": "composer", "artist": {"name": "Max Martin", "id": "m3"}},
                ]}},
            ],
        }
        credits = extract_credits(recording)
        roles = {c["name"]: c["role"] for c in credits}
        assert roles["Rick Rubin"] == "producer"
        assert roles["Max Martin"] == "songwriter"
        assert roles["Some Engineer"] == "engineer"

    def test_extract_credits_empty_recording(self):
        assert extract_credits({}) == []

    def test_fuzzy_match_accepts_close_titles(self):
        candidate = {
            "title": "HUMBLE.",
            "artist-credit": [{"artist": {"name": "Kendrick Lamar"}}],
        }
        assert is_good_match("Humble", "Kendrick Lamar", candidate)

    def test_fuzzy_match_rejects_different_artist(self):
        candidate = {
            "title": "Humble",
            "artist-credit": [{"artist": {"name": "Completely Different Band"}}],
        }
        assert not is_good_match("Humble", "Kendrick Lamar", candidate)


class TestBillboard:
    def _weekly(self):
        return pd.DataFrame(
            {
                "chart_date": ["2020-01-04", "2020-01-11", "2020-01-18"],
                "rank": [40, 20, 5],
                "track_name": ["Riser", "Riser", "Riser"],
                "artist": ["Band A", "Band A", "Band A"],
                "peak_position": [40, 20, 5],
                "weeks_on_chart": [1, 2, 3],
            }
        )

    def test_summarize_derives_peak_and_weeks(self):
        runs = summarize_chart_runs(self._weekly())
        assert len(runs) == 1
        assert runs.iloc[0]["peak_position"] == 5
        assert runs.iloc[0]["weeks_on_chart"] == 3

    def test_summarize_trajectory_rising(self):
        runs = summarize_chart_runs(self._weekly())
        assert runs.iloc[0]["trajectory"] == "rising"

    def test_summarize_empty(self):
        runs = summarize_chart_runs(pd.DataFrame(columns=CHART_COLUMNS))
        assert runs.empty

    def test_match_to_tracks_fuzzy(self):
        runs = summarize_chart_runs(self._weekly())
        tracks = pd.DataFrame(
            {
                "track_id": ["t1", "t2"],
                "track_name": ["Riser", "Unrelated Song"],
                "primary_artist_name": ["Band A", "Band B"],
            }
        )
        out = match_to_tracks(runs, tracks)
        assert out.loc[out["track_id"] == "t1", "billboard_peak_position"].iloc[0] == 5
        assert pd.isna(out.loc[out["track_id"] == "t2", "billboard_peak_position"].iloc[0])

    def test_fallback_csv_normalises_columns(self, tmp_path):
        csv = tmp_path / "hot100.csv"
        pd.DataFrame(
            {
                "week_id": ["2020-01-04"], "rank": [1], "song": ["Hit"],
                "performer": ["Star"], "peak_rank": [1], "weeks_on_board": [10],
            }
        ).to_csv(csv, index=False)
        df = load_fallback_csv(str(csv))
        assert list(df.columns) == CHART_COLUMNS

    def test_fallback_csv_rejects_bad_schema(self, tmp_path):
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1]}).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_fallback_csv(str(csv))
