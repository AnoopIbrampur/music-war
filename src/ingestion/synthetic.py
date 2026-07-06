"""Synthetic dataset generator (demo mode).

Produces a dataset with the exact shape of the real merged pipeline output,
but with *known* latent artist/producer/songwriter effects baked into the
success scores. This serves two purposes:

1. The full pipeline + dashboard run end-to-end without any API keys.
2. Tests can verify the WAR model actually recovers planted effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

GENRES = [
    "pop", "hip_hop", "rock", "rnb", "country", "electronic", "latin",
    "indie", "metal", "jazz",
]
# Structural genre lift so the model has real confounders to control for.
GENRE_EFFECT = {g: e for g, e in zip(GENRES, [8, 10, 2, 4, 3, 5, 6, 1, -2, -4])}


@dataclass
class SyntheticDataset:
    """Container for all generated frames plus the planted ground truth."""

    tracks: pd.DataFrame
    track_artists: pd.DataFrame
    track_producers: pd.DataFrame
    track_songwriters: pd.DataFrame
    artists: pd.DataFrame
    true_effects: dict[str, dict[str, float]] = field(default_factory=dict)


def generate(n_tracks: int = 4000, n_artists: int = 300, n_producers: int = 120,
             n_songwriters: int = 150, seed: int = 42) -> SyntheticDataset:
    """Generate a correlated, realistic-shaped music dataset."""
    rng = np.random.default_rng(seed)

    artist_ids = [f"AR{i:04d}" for i in range(n_artists)]
    producer_ids = [f"PR{i:04d}" for i in range(n_producers)]
    writer_ids = [f"SW{i:04d}" for i in range(n_songwriters)]

    artist_skill = dict(zip(artist_ids, rng.normal(0, 8, n_artists)))
    producer_skill = dict(zip(producer_ids, rng.normal(0, 5, n_producers)))
    writer_skill = dict(zip(writer_ids, rng.normal(0, 4, n_songwriters)))
    artist_genre = {a: rng.choice(GENRES) for a in artist_ids}

    # Popularity-weighted sampling: a few stars appear on many tracks,
    # mirroring the long tail in real catalogues.
    artist_weights = rng.pareto(1.5, n_artists) + 0.1
    artist_weights /= artist_weights.sum()
    producer_weights = rng.pareto(1.5, n_producers) + 0.1
    producer_weights /= producer_weights.sum()
    writer_weights = rng.pareto(1.5, n_songwriters) + 0.1
    writer_weights /= writer_weights.sum()

    track_rows, ta_rows, tp_rows, tw_rows = [], [], [], []
    for i in range(n_tracks):
        tid = f"TK{i:06d}"
        n_track_artists = rng.choice([1, 1, 1, 2, 2, 3])
        chosen = rng.choice(artist_ids, size=n_track_artists, replace=False, p=artist_weights)
        primary = chosen[0]
        genre = artist_genre[primary]
        year = int(rng.integers(1990, 2026))
        month = int(rng.integers(1, 13))

        producers = rng.choice(
            producer_ids, size=int(rng.choice([1, 1, 2])), replace=False, p=producer_weights
        )
        writers = rng.choice(
            writer_ids, size=int(rng.choice([1, 1, 2])), replace=False, p=writer_weights
        )

        # Ground-truth signal: genre + era + people + noise
        signal = 45.0 + GENRE_EFFECT[genre] + (year - 1990) * 0.15
        signal += artist_skill[primary]
        for feat in chosen[1:]:
            signal += 0.5 * artist_skill[feat]  # featured artists count half
        signal += sum(producer_skill[p] for p in producers)
        signal += sum(writer_skill[w] for w in writers)
        popularity = float(np.clip(signal + rng.normal(0, 6), 0, 100))

        charted = popularity > 62 and rng.random() < 0.8
        peak = int(np.clip(101 - popularity + rng.normal(0, 8), 1, 100)) if charted else None
        weeks = int(np.clip((popularity - 55) / 2 + rng.normal(0, 3), 1, 60)) if charted else None

        track_rows.append(
            {
                "track_id": tid,
                "track_name": f"Track {i}",
                "album_id": f"AL{i // 10:05d}",
                "album_name": f"Album {i // 10}",
                "album_type": "album",
                "release_date": f"{year}-{month:02d}-15",
                "duration_ms": int(rng.normal(215000, 45000)),
                "explicit": bool(rng.random() < 0.3),
                "spotify_popularity": popularity,
                "primary_genre": genre,
                "danceability": float(np.clip(rng.normal(0.6, 0.15), 0, 1)),
                "energy": float(np.clip(rng.normal(0.65, 0.18), 0, 1)),
                "key": int(rng.integers(0, 12)),
                "loudness": float(rng.normal(-7, 3)),
                "mode": int(rng.integers(0, 2)),
                "speechiness": float(np.clip(rng.beta(2, 8), 0, 1)),
                "acousticness": float(np.clip(rng.beta(2, 5), 0, 1)),
                "instrumentalness": float(np.clip(rng.beta(1, 10), 0, 1)),
                "liveness": float(np.clip(rng.beta(2, 8), 0, 1)),
                "valence": float(np.clip(rng.normal(0.5, 0.2), 0, 1)),
                "tempo": float(np.clip(rng.normal(120, 25), 60, 200)),
                "time_signature": 4,
                "billboard_peak_position": peak,
                "billboard_weeks_on_chart": weeks,
            }
        )
        for order, aid in enumerate(chosen, start=1):
            ta_rows.append(
                {
                    "track_id": tid,
                    "artist_id": aid,
                    "role": "primary_artist" if order == 1 else "featured_artist",
                    "billing_order": order,
                }
            )
        tp_rows.extend({"track_id": tid, "producer_id": p, "producer_name": f"Producer {p[2:]}"} for p in producers)
        tw_rows.extend({"track_id": tid, "songwriter_id": w, "songwriter_name": f"Writer {w[2:]}"} for w in writers)

    artists_df = pd.DataFrame(
        {
            "artist_id": artist_ids,
            "artist_name": [f"Artist {a[2:]}" for a in artist_ids],
            "primary_genre": [artist_genre[a] for a in artist_ids],
            "followers": rng.integers(1_000, 50_000_000, n_artists),
            "artist_popularity": rng.integers(10, 100, n_artists),
        }
    )

    return SyntheticDataset(
        tracks=pd.DataFrame(track_rows),
        track_artists=pd.DataFrame(ta_rows),
        track_producers=pd.DataFrame(tp_rows),
        track_songwriters=pd.DataFrame(tw_rows),
        artists=artists_df,
        true_effects={
            "artist": artist_skill,
            "producer": producer_skill,
            "songwriter": writer_skill,
        },
    )
