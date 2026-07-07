# Architecture

## Data sources

The pipeline takes one of three sources, chosen with `--source`.

`bulk` is the default and the one you should use. It pulls a real Spotify export from Hugging Face (`maharshipandya/spotify-tracks-dataset`): 114,000 tracks with real names, popularity, genres, and audio features. This exists because Spotify cut off the live audio-features, popularity, and genre endpoints for apps created after November 2024, so a new API key can't feed the model.

`api` is the original live path: Spotify for tracks and audio, MusicBrainz for producer and songwriter credits, Billboard for chart history. It still works if you have an older app with extended access.

`demo` generates synthetic data with talent effects planted on purpose, used by the tests and for a keyless run.

## Data flow

```
┌────────────────────┐   ┌──────────────┐   ┌─────────────┐
│ Hugging Face bulk  │   │ Spotify API  │   │ MusicBrainz │
│ export (default)   │   │ (spotipy)    │   │  Billboard  │
└─────────┬──────────┘   └──────┬───────┘   └──────┬──────┘
          │ real tracks,        │ tracks, audio    │ credits,
          │ audio, popularity   │ (api path)       │ charts
          ▼                     ▼                  ▼
┌──────────────────────────────────────────────────┐
│  ingestion: bulk_dataset / data_collector        │
│  raw cache in data/raw/, parquet checkpoints     │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│  processing: cleaner → transformer →             │
│  feature_engineer (success score, sound clusters)│
└───────────┬─────────────────────────┬────────────┘
            ▼                         ▼
┌───────────────────────┐  ┌──────────────────────────┐
│ database: star schema │  │ modeling:                │
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

`bridge_track_artist` also carries `role` (primary or featured) and
`billing_order`. The modeling layer turns those into the billing weights.

## Rate limiting on the API path

The bulk export is a single download, so none of this applies to the default run. It matters only when you use `--source api`.

| Source | Limit | Handling |
|---|---|---|
| Spotify | dynamic, 429 with Retry-After | spotipy retries plus a wrapper backoff; endpoints batched at 100 audio features and 50 artists per call |
| MusicBrainz | 1 request per second by policy | a monotonic-clock throttle before every call, and a user agent with a contact email |
| Billboard | unofficial | one request per weekly chart, with a CSV fallback if it's blocked |

Every raw response caches to `data/raw/` keyed by the request, so reruns cost nothing and an interrupted pull picks up where it left off.

## Orchestration

On the API path, `DataCollector.run()` goes Spotify, then MusicBrainz, then Billboard, writing a parquet checkpoint after each stage. Restart it and it loads whatever checkpoints exist instead of re-fetching. Database loads wipe and replace each table, so running twice never doubles the rows. The modeling stage is pure compute and always reruns.

The bulk and demo sources skip the collector entirely. Bulk downloads and reshapes the export in `src/ingestion/bulk_dataset.py`; demo builds synthetic frames in `src/ingestion/synthetic.py`. Both hand the rest of the pipeline the same shape of data the API path produces, so nothing downstream cares which source ran.
