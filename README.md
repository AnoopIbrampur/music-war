# Music WAR — Wins Above Replacement for music

Baseball has a stat called WAR: roughly, how many more games a team wins by playing someone instead of a freely available replacement. The trick is that it measures a player against a baseline and controls for context, so a shortstop in a pitcher's park gets judged fairly.

I wanted to try the same idea on music. When a track does well, how much of that is the artist actually being good, and how much is just the genre being popular, the sound being on-trend, or a bigger name carrying the feature? This project fits a regression that pulls those apart and gives every artist a single number: the points they add to a track's success above a replacement-level artist, holding genre, sound, and collaborators fixed.

The most fun result: WAR and raw popularity only correlate about 0.80. The gap is where it gets interesting. Bad Bunny and Rammstein overdeliver even for how famous they are. Justin Bieber and J Balvin land *negative* WAR despite being huge, because their hits are often collaborations where, once you control for the genre and the other artists, their own marginal contribution trails what a replacement would give you. A plain popularity chart can't tell you that.

## What it runs on

Here's the honest version of the data story. Spotify shut off the API endpoints this project needs (audio features, popularity, genres) for any app created after November 2024. A freshly registered key can authenticate and not much else. So the default path uses a real Spotify export published on Hugging Face instead of the live API: 114,000 tracks across 114 genres, with real names, popularity, and audio features baked in from before the lockdown. After cleaning that's about 80,000 unique tracks and 4,580 artists with enough credits to model.

Since WAR is a look-back analysis and Spotify popularity is a snapshot either way, a pre-lockdown export gives up nothing analytically, and it hands you far more tracks than you could ever pull through a rate-limited API. The live-API code still ships for anyone who has an older app with extended access.

## How it works

Every track gets a 0–100 success score. On the export that score is driven by Spotify popularity; when Billboard chart data is available (the API path), it also folds in peak position, weeks on chart, and a longevity bonus.

The model is a big sparse matrix, one row per track. Most columns are a single on/off flag for each artist with at least five credits, and the rest are controls: genre, the track's sound profile from a k-means clustering of its audio features, tempo and duration buckets, explicit flag, and how many artists are on it. Featured artists count for half a top billing. A ridge regression predicts the success score, and the coefficient on an artist's column is their WAR. Everyone below the credit threshold dissolves into the intercept, which is exactly what "replacement level" should mean. A lasso runs alongside it and zeroes out the weak effects, so anyone who survives both models is a safer bet. Bootstrap resampling gives each artist a 95% confidence interval.

On the real data the model lands at R² 0.46 on held-out tracks, against 0.16 for a version that only knows genre and sound and nothing about who made the track. That jump is the whole point: who is on a record explains far more than what it sounds like.

## Architecture

```
Spotify API (locked for new apps) ─┐
Hugging Face bulk export ──────────┼─► ingestion → cleaner → feature_engineer
MusicBrainz / Billboard (optional) ┘         │
                                             ▼
                                   star schema (SQLAlchemy)
                                             │
                                             ▼
                            sparse matrix → RidgeCV / LassoCV
                                             │
                                             ▼
                              WAR table + bootstrap CIs → Streamlit
```

Diagrams: [docs/architecture.md](docs/architecture.md). Statistical detail: [docs/methodology.md](docs/methodology.md).

## Quickstart

```bash
git clone https://github.com/AnoopIbrampur/music-war && cd music-war
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Real data: ~80k Spotify tracks with real names. Downloads the export, no keys.
python -m src.pipeline --source bulk

# Synthetic data with known planted effects. Runs in ~30s and validates the model.
python -m src.pipeline --demo

# Live Spotify API. Needs an older app with extended access (see the data note).
cp .env.example .env
python -m src.pipeline --source api

streamlit run dashboard/app.py
pytest
```

## What the model found (real data)

1. WAR is not a synonym for popularity. They track together at 0.80, and the leftover is the signal.
2. Overperformers, who beat their own fame: Bad Bunny (WAR 29.8, 95% CI 25.8–34.8), Rammstein, Nicki Minaj, Bring Me The Horizon.
3. Coattail riders, popular but with negative WAR once you control for context: Justin Bieber (−15.3), J Balvin (−15.4).
4. Total-WAR leaders reward being both good and prolific: The Beatles, Arctic Monkeys, Arijit Singh, Bad Bunny.
5. Best in their lane: Doja Cat tops dance, System Of A Down tops metal. WAR lets you compare within a genre instead of across.
6. Instrumentalness is the strongest single audio predictor of popularity, and it's negative (r = −0.19): tracks with vocals win. Every audio feature on its own is weak, which is exactly why the artist columns carry so much of the model.
7. Explicit tracks average about four popularity points higher, and tracks with two or three credited artists beat solo tracks.

The `--demo` run reproduces the same pipeline on synthetic data with effects I planted on purpose, and the model recovers them at r > 0.9. That's how I check the machinery isn't fooling itself before trusting it on real names.

## Repository layout

```
config/settings.py      constants and .env loading
src/ingestion/          spotify_client, musicbrainz_client, billboard_client,
                        bulk_dataset (real export), synthetic (demo), data_collector
src/processing/         cleaner, transformer, feature_engineer
src/modeling/           sparse_matrix_builder, war_calculator, model_evaluator
src/database/           SQLAlchemy star schema + db_manager
src/visualization/      Plotly chart builders
dashboard/app.py        Streamlit app
notebooks/              exploration, feature engineering, modeling walkthroughs
tests/                  67 pytest cases, including a ground-truth recovery test
docs/                   methodology.md, architecture.md
```

## Tech stack

Python, pandas, NumPy, SciPy (sparse matrices), scikit-learn (RidgeCV, LassoCV, KMeans), SQLAlchemy on SQLite or PostgreSQL, spotipy, musicbrainzngs, billboard.py, rapidfuzz, Streamlit, Plotly, pytest.

## Limitations

Spotify popularity is a rolling number, not a historical one, so older catalogues get scored on how they stream today. WAR is a correlation, not proof of cause: stars get first pick of good songs, and the model can't separate talent from that head start. Duos who always record together share one effective coefficient that ridge splits between them. The export has no release dates or production credits, so there's no era analysis and no producer or songwriter WAR unless you enrich with MusicBrainz on the API path.

Things I'd add next: play-count panel data for a real time series instead of a popularity snapshot, per-role adjustments like baseball's positional ones, and credit coverage from Genius or Discogs.

## License

[MIT](LICENSE)
