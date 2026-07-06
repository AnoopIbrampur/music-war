"""Spotify ingestion client.

Pulls track metadata, audio features, and artist details via the Spotify
Web API (client-credentials flow). Raw JSON responses are cached under
``data/raw/spotify/`` so re-runs never re-fetch the same object.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from config import settings

logger = logging.getLogger(__name__)

# Patterns used to pull featured artists out of track titles, e.g.
# "Song (feat. Artist)" / "Song ft. Artist" / "Song (with Artist)".
_FEAT_RE = re.compile(
    r"[\(\[]?\s*(?:feat\.?|ft\.?|featuring|with)\s+(?P<names>[^\)\]]+)[\)\]]?",
    re.IGNORECASE,
)
_NAME_SPLIT_RE = re.compile(r"\s*(?:,|&|\band\b|\bx\b)\s*", re.IGNORECASE)


def parse_featured_artists(track_name: str) -> tuple[str, list[str]]:
    """Split a track title into (clean_title, [featured artist names]).

    >>> parse_featured_artists("Sicko Mode (feat. Drake)")
    ('Sicko Mode', ['Drake'])
    """
    match = _FEAT_RE.search(track_name)
    if not match:
        return track_name.strip(), []
    names = [n.strip() for n in _NAME_SPLIT_RE.split(match.group("names")) if n.strip()]
    clean = _FEAT_RE.sub("", track_name).strip(" -–([")
    return clean.strip(), names


def batch(items: list[Any], size: int) -> Iterator[list[Any]]:
    """Yield successive fixed-size chunks (Spotify endpoints cap batch sizes)."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


class SpotifyClient:
    """Thin wrapper over spotipy with caching and 429-aware backoff."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or (settings.RAW_DIR / "spotify")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._sp = None  # lazily constructed so tests never need credentials

    @property
    def sp(self):
        if self._sp is None:
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials

            if not settings.SPOTIFY_CLIENT_ID or not settings.SPOTIFY_CLIENT_SECRET:
                raise RuntimeError(
                    "SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET are not set. "
                    "Copy .env.example to .env and fill them in, or run the "
                    "pipeline with --demo to use synthetic data."
                )
            auth = SpotifyClientCredentials(
                client_id=settings.SPOTIFY_CLIENT_ID,
                client_secret=settings.SPOTIFY_CLIENT_SECRET,
            )
            # spotipy retries 429s itself, honouring the Retry-After header
            self._sp = spotipy.Spotify(
                auth_manager=auth,
                retries=settings.SPOTIFY_MAX_RETRIES,
                status_forcelist=(429, 500, 502, 503, 504),
            )
        return self._sp

    # ------------------------------------------------------------- caching
    def _cache_path(self, kind: str, key: str) -> Path:
        return self.cache_dir / f"{kind}__{key}.json"

    def _cached_call(self, kind: str, key: str, fetch) -> dict:
        """Return the cached JSON for (kind, key) or fetch + persist it."""
        path = self._cache_path(kind, key)
        if path.exists():
            return json.loads(path.read_text())
        result = self._call_with_backoff(fetch)
        path.write_text(json.dumps(result))
        return result

    def _call_with_backoff(self, fetch):
        """Belt-and-braces exponential backoff on top of spotipy's retries."""
        from spotipy.exceptions import SpotifyException

        delay = 1.0
        for attempt in range(settings.SPOTIFY_MAX_RETRIES):
            try:
                return fetch()
            except SpotifyException as exc:
                if exc.http_status != 429:
                    raise
                retry_after = float(exc.headers.get("Retry-After", delay)) if exc.headers else delay
                logger.warning("Spotify 429; sleeping %.1fs (attempt %d)", retry_after, attempt + 1)
                time.sleep(retry_after)
                delay *= 2
        raise RuntimeError("Spotify API kept rate limiting after max retries")

    # --------------------------------------------------------------- fetch
    def get_playlist_tracks(self, playlist_id: str) -> list[dict]:
        """All track objects from a playlist, following pagination."""
        tracks: list[dict] = []
        results = self._cached_call(
            "playlist", playlist_id, lambda: self.sp.playlist_items(playlist_id)
        )
        while True:
            tracks.extend(
                item["track"] for item in results.get("items", []) if item.get("track")
            )
            if not results.get("next"):
                break
            results = self._call_with_backoff(lambda: self.sp.next(results))
        return tracks

    def get_audio_features(self, track_ids: list[str]) -> list[dict]:
        """Audio features for many tracks, batched at the API limit of 100."""
        features: list[dict] = []
        for chunk in batch(track_ids, settings.SPOTIFY_AUDIO_FEATURES_BATCH):
            key = chunk[0] + f"_{len(chunk)}"
            result = self._cached_call(
                "audio_features", key, lambda c=chunk: {"audio_features": self.sp.audio_features(c)}
            )
            features.extend(f for f in result["audio_features"] if f)
        return features

    def get_artists(self, artist_ids: list[str]) -> list[dict]:
        """Full artist objects (genres, followers, popularity), batched at 50."""
        artists: list[dict] = []
        for chunk in batch(artist_ids, 50):
            key = chunk[0] + f"_{len(chunk)}"
            result = self._cached_call(
                "artists", key, lambda c=chunk: self.sp.artists(c)
            )
            artists.extend(a for a in result["artists"] if a)
        return artists

    def search_playlists(self, query: str, limit: int = 10) -> list[dict]:
        """Find curated playlists by name (Today's Top Hits, RapCaviar, ...)."""
        result = self._cached_call(
            "playlist_search", re.sub(r"\W+", "_", query.lower()),
            lambda: self.sp.search(q=query, type="playlist", limit=limit),
        )
        return [p for p in result.get("playlists", {}).get("items", []) if p]

    # ------------------------------------------------------------ normalise
    @staticmethod
    def track_to_row(track: dict) -> dict:
        """Flatten a Spotify track object into a tabular record."""
        album = track.get("album") or {}
        artists = track.get("artists") or []
        clean_name, featured = parse_featured_artists(track.get("name", ""))
        return {
            "track_id": track.get("id"),
            "track_name": track.get("name"),
            "clean_track_name": clean_name,
            "featured_from_title": featured,
            "album_id": album.get("id"),
            "album_name": album.get("name"),
            "album_type": album.get("album_type"),
            "release_date": album.get("release_date"),
            "duration_ms": track.get("duration_ms"),
            "explicit": bool(track.get("explicit")),
            "spotify_popularity": track.get("popularity"),
            "artist_ids": [a.get("id") for a in artists],
            "artist_names": [a.get("name") for a in artists],
        }


# Curated seed playlists spanning genres and eras (names are searched, so
# regional variants still resolve).
SEED_PLAYLIST_QUERIES: tuple[str, ...] = (
    "Today's Top Hits", "RapCaviar", "Rock Classics", "Pop Rising",
    "All Out 2010s", "All Out 2000s", "All Out 90s", "Hot Country",
    "Viva Latino", "mint", "Are & Be", "Beast Mode", "Peaceful Piano",
    "Jazz Classics", "Metal Essentials", "Indie Pop", "K-Pop ON!",
)


def collect_seed_tracks(client: SpotifyClient, queries: Iterable[str] = SEED_PLAYLIST_QUERIES) -> list[dict]:
    """Crawl seed playlists and return flattened, deduplicated track rows."""
    seen: set[str] = set()
    rows: list[dict] = []
    for query in queries:
        try:
            playlists = client.search_playlists(query, limit=3)
        except Exception:
            logger.exception("Playlist search failed for %r; skipping", query)
            continue
        for playlist in playlists:
            for track in client.get_playlist_tracks(playlist["id"]):
                tid = track.get("id")
                if tid and tid not in seen:
                    seen.add(tid)
                    rows.append(client.track_to_row(track))
    logger.info("Collected %d unique tracks from %d seed queries", len(rows), len(list(queries)))
    return rows
