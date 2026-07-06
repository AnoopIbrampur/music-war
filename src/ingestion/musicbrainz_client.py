"""MusicBrainz credit enrichment.

Matches Spotify tracks to MusicBrainz recordings and extracts producer,
songwriter, and engineer relationships. MusicBrainz allows 1 request per
second and requires a descriptive user agent with contact info — both are
enforced here.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from rapidfuzz import fuzz

from config import settings

logger = logging.getLogger(__name__)

# Relationship types we care about, mapped to our role vocabulary.
_RELATION_ROLES = {
    "producer": "producer",
    "composer": "songwriter",
    "writer": "songwriter",
    "lyricist": "songwriter",
    "mix": "engineer",
    "recording": "engineer",
    "mastering": "engineer",
}


def extract_credits(recording: dict) -> list[dict]:
    """Pull (name, role) credits from a MusicBrainz recording's relations.

    Works on both recording-level and work-level relation lists.
    """
    credits: list[dict] = []
    for rel in recording.get("artist-relation-list", []) or []:
        role = _RELATION_ROLES.get(rel.get("type", "").lower())
        artist = rel.get("artist", {})
        if role and artist.get("name"):
            credits.append(
                {"name": artist["name"], "mbid": artist.get("id"), "role": role}
            )
    # Songwriters usually hang off the linked *work*, not the recording
    for work_rel in recording.get("work-relation-list", []) or []:
        work = work_rel.get("work", {})
        for rel in work.get("artist-relation-list", []) or []:
            role = _RELATION_ROLES.get(rel.get("type", "").lower())
            artist = rel.get("artist", {})
            if role and artist.get("name"):
                credits.append(
                    {"name": artist["name"], "mbid": artist.get("id"), "role": role}
                )
    return credits


def is_good_match(track_name: str, artist_name: str, candidate: dict,
                  threshold: int = settings.FUZZY_MATCH_THRESHOLD) -> bool:
    """Fuzzy-match a Spotify (track, artist) pair against an MB recording."""
    title_score = fuzz.token_set_ratio(track_name.lower(), candidate.get("title", "").lower())
    mb_artists = " ".join(
        c.get("artist", {}).get("name", "")
        for c in candidate.get("artist-credit", [])
        if isinstance(c, dict)
    )
    artist_score = fuzz.token_set_ratio(artist_name.lower(), mb_artists.lower())
    return title_score >= threshold and artist_score >= threshold


class MusicBrainzClient:
    """Rate-limited MusicBrainz lookups with on-disk JSON caching."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or (settings.RAW_DIR / "musicbrainz")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request = 0.0
        self._initialised = False

    def _init_api(self) -> None:
        if self._initialised:
            return
        import musicbrainzngs

        if not settings.MUSICBRAINZ_CONTACT:
            raise RuntimeError(
                "MUSICBRAINZ_CONTACT is not set — MusicBrainz policy requires "
                "a user agent with contact info. Fill in .env or use --demo."
            )
        musicbrainzngs.set_useragent(
            settings.MUSICBRAINZ_APP_NAME,
            settings.MUSICBRAINZ_APP_VERSION,
            settings.MUSICBRAINZ_CONTACT,
        )
        self._mb = musicbrainzngs
        self._initialised = True

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        wait = settings.MUSICBRAINZ_RATE_LIMIT_SECONDS - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def get_track_credits(self, track_name: str, artist_name: str) -> list[dict]:
        """Search MB for a recording and return its producer/writer credits.

        Returns [] when no confident match exists — never guesses.
        """
        cache_key = f"{track_name}::{artist_name}".lower().replace("/", "_")[:180]
        cache_path = self.cache_dir / f"credits__{abs(hash(cache_key))}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        self._init_api()
        self._throttle()
        try:
            result = self._mb.search_recordings(
                recording=track_name, artist=artist_name, limit=5
            )
        except Exception:
            logger.exception("MB search failed for %s — %s", track_name, artist_name)
            return []

        credits: list[dict] = []
        for candidate in result.get("recording-list", []):
            if not is_good_match(track_name, artist_name, candidate):
                continue
            self._throttle()
            try:
                full = self._mb.get_recording_by_id(
                    candidate["id"],
                    includes=["artist-rels", "work-rels", "work-level-rels", "label-rels"],
                )["recording"]
            except Exception:
                logger.exception("MB lookup failed for recording %s", candidate["id"])
                continue
            credits = extract_credits(full)
            if credits:
                break

        cache_path.write_text(json.dumps(credits))
        return credits
