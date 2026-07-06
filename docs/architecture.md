# Music WAR — Architecture

## Data flow

```
┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│ Spotify API │   │ MusicBrainz  │   │  Billboard  │
│ (spotipy)   │   │ (1 req/sec)  │   │ (billboard. │
│             │   │              │   │  py / CSV)  │
└──────┬──────┘   └──────┬───────┘   └──────┬──────┘
       │ tracks,         │ producer/        │ weekly Hot 100
       │ audio features, │ songwriter       │ snapshots
       │ artists         │ credits          │
       ▼                 ▼                  ▼
┌──────────────────────────────────────────────────┐
│        data_collector.py  (checkpoint/resume)    │
│  raw JSON cache: data/raw/   parquet checkpoints │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│  processing/: cleaner → transformer →            │
│  feature_engineer (composite score, clusters)    │
└───────────┬─────────────────────────┬────────────┘
            ▼                         ▼
┌───────────────────────┐  ┌──────────────────────────┐
│ database/: star schema│  │ modeling/:               │
│ SQLite / PostgreSQL   │  │ sparse_matrix_builder →  │
│ via SQLAlchemy        │  │ war_calculator (ridge +  │
└───────────────────────┘  │ lasso) → model_evaluator │
                           └────────────┬─────────────┘
                                        ▼
                           ┌──────────────────────────┐
                           │ dashboard/app.py         │
                           │ (Streamlit + Plotly)     │
                           └──────────────────────────┘
```

## Star schema

```
                      ┌─────────────────────────┐
                      │ fact_track_performance  │
                      │ track_id (PK/FK)        │
                      │ spotify_popularity      │
                      │ billboard_peak_position │
                      │ billboard_weeks_on_chart│
                      │ composite_success_score │
                      └───────────┬─────────────┘
                                  │ 1:1
                      ┌───────────┴─────────────┐
                      │ dim_track               │
                      │ track_id (PK), name,    │
                      │ release_year, genre,    │
                      │ audio features…         │
                      └───┬─────────┬───────┬───┘
              M:N via     │         │       │      M:N via
     bridge_track_artist  │         │       │  bridge_track_songwriter
          ┌───────────────┘         │       └────────────────┐
          ▼                         ▼ M:N via                ▼
  ┌──────────────┐        bridge_track_producer      ┌────────────────┐
  │ dim_artist   │        ┌──────────────┐           │ dim_songwriter │
  │ artist_id,   │        │ dim_producer │           │ songwriter_id, │
  │ name, genre, │        │ producer_id, │           │ name, tracks   │
  │ followers…   │        │ name, tracks │           └────────────────┘
  └──────────────┘        └──────────────┘
```

`bridge_track_artist` additionally carries `role`
(primary_artist / featured_artist) and `billing_order` — the modeling
layer turns these into billing weights.

## Rate limiting strategy

| Source | Limit | Handling |
|---|---|---|
| Spotify | dynamic, 429 + Retry-After | spotipy retries + wrapper exponential backoff, batched endpoints (100 audio features, 50 artists per call) |
| MusicBrainz | 1 request/second (policy) | monotonic-clock throttle before every call; descriptive user agent with contact email |
| Billboard | unofficial | one request per weekly chart; CSV fallback if blocked |

All raw responses cache to `data/raw/` keyed by request, so reruns are
free and the pipeline is resumable.

## Pipeline orchestration

`DataCollector.run()` executes stages in order — Spotify → MusicBrainz →
Billboard — with a parquet checkpoint after each. On restart, any existing
checkpoint is loaded instead of re-fetching. Database loads are
wipe-and-replace per table (idempotent). The modeling stage is pure
compute and always reruns.

Demo mode (`python -m src.pipeline --demo`) swaps the three API stages for
`src/ingestion/synthetic.py`, which produces identically-shaped frames
with known planted talent effects — used by tests and for keyless demos.
