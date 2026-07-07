# 🎵 Music WAR — Wins Above Replacement for the Music Industry

**Sabermetrics for songs: a sparse-regression model that isolates how many
success-score points each artist, producer, and songwriter adds to a track
above a replacement-level substitute.**

Baseball's WAR asks how many wins a player adds over freely available
replacement talent. Inspired by the "Moneyball for Movies" approach (sparse
matrix regressions on IMDb credits), Music WAR asks the same question of
music: when Metro Boomin produces a track, or Drake jumps on a feature, how
much of the resulting success is *them* — and how much is genre, era, and
sound that any replacement-level person would have delivered anyway?

## How it works

1. **Ingest** — 50k-track target from Spotify (metadata, audio features,
   artists), enriched with producer/songwriter credits from MusicBrainz and
   25 years of Billboard Hot 100 history. Checkpointed, cached, rate-limited.
2. **Score** — every track gets a 0–100 **composite success score**:
   Spotify popularity (40%) + Billboard peak (30%) + weeks on chart (20%) +
   longevity bonus (10%).
3. **Model** — a `scipy.sparse` design matrix with one binary column per
   eligible person (≥5 credits) plus genre/era/sound controls, fed to a
   cross-validated **ridge regression**. Each person's coefficient *is*
   their WAR. A parallel **lasso** flags which effects are robust.
4. **Explore** — a 6-tab Streamlit dashboard: leaderboards, artist deep
   dives, and a Dream Team Builder that predicts the score of your
   hypothetical collab.

## Architecture

```
Spotify ──┐
MusicBrainz ─┼─► data_collector (checkpoint/resume, JSON cache)
Billboard ──┘        │
                     ▼
        cleaner → feature_engineer → star schema (SQLAlchemy)
                     │
                     ▼
   sparse matrix → RidgeCV / LassoCV → WAR table + bootstrap CIs
                     │
                     ▼
            Streamlit dashboard (Plotly)
```

Full diagrams: [docs/architecture.md](docs/architecture.md) ·
Statistical details: [docs/methodology.md](docs/methodology.md)

## Quickstart

```bash
git clone <repo-url> && cd music-war
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Option A: REAL DATA — 80k+ real Spotify tracks, real artist names (~2 min).
# Downloads a public bulk Spotify export; no API keys required.
python -m src.pipeline --source bulk

# Option B: synthetic demo with known planted effects (~30s, validates the model)
python -m src.pipeline --demo

# Option C: live Spotify API (see note below — restricted for new apps)
cp .env.example .env   # add SPOTIFY_CLIENT_ID / SECRET
python -m src.pipeline --source api

# Launch the dashboard
streamlit run dashboard/app.py

# Run the test suite (66 tests)
pytest
```

### A note on data sources

Spotify **deprecated the audio-features, popularity, and genre fields for
apps created after November 2024**. A newly registered app therefore can't
pull the inputs this model needs from the live API — this is a Spotify
platform change, not a limitation of this project. So the default and
recommended path is `--source bulk`, which uses a **real** pre-lockdown
Spotify export (the [Hugging Face `maharshipandya/spotify-tracks-dataset`](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset),
114k tracks / 114 genres, with real names, popularity, and audio features).
Since WAR is a retrospective analysis, a snapshot export is analytically
equivalent to a live pull — and gives far more tracks than rate-limited API
calls would. The `--source api` path is kept intact for anyone with an
extended-quota Spotify app.

## Key findings (real data — 80,393 Spotify tracks, 4,580 artists)

Model: ridge (α=1, 5-fold CV), test **R² = 0.46 vs 0.16** for a structure-only
baseline — *who* is on a track explains **~30% more variance** than genre and
sound alone. That gap is the whole thesis, and the findings below flow from it.

1. **WAR is not just popularity** (they correlate 0.80, not 1.0). The gap is
   the insight: some artists overperform their fame, others ride it.
2. **Overperformers** — deliver more than a replacement even at their fame
   level: **Bad Bunny** (WAR 29.8, 95% CI 25.8–34.8), **Rammstein**,
   **Nicki Minaj**, **Bring Me The Horizon**.
3. **Coattail riders** — popular but *negative* WAR: **Justin Bieber**
   (−15.3) and **J Balvin** (−15.4). Their hits are often collabs where,
   controlling for genre and co-artists, their marginal contribution trails a
   replacement's. A naive popularity ranking misses this entirely.
4. **Total-WAR leaders reward prolific excellence**: **The Beatles**,
   **Arctic Monkeys**, **Arijit Singh**, **Bad Bunny**.
5. **Best-in-genre**: Doja Cat (dance), System Of A Down (metal) — WAR lets
   you compare like-for-like within a genre.
6. **Instrumentalness is the #1 audio popularity-killer** (r = −0.19); every
   audio feature on its own is weak (|r| < 0.2), which is *why* artist
   identity dominates.
7. **Explicit tracks average +4.3 popularity points**, and **collaborations
   beat solo tracks** — features and co-bills lift outcomes.
8. **Bootstrap 95% CIs** accompany every artist's WAR, separating robust
   rankings from small-sample noise; **lasso agreement** flags the most
   defensible names.

*Names are real; `--source demo` reproduces the same structure on synthetic
data with known planted effects (recovered at r > 0.9), which is how the model
itself is validated.*

## Repository layout

```
config/settings.py      all constants + .env loading
src/ingestion/          spotify_client, musicbrainz_client, billboard_client,
                        data_collector (orchestration), synthetic (demo data)
src/processing/         cleaner, transformer, feature_engineer
src/modeling/           sparse_matrix_builder, war_calculator, model_evaluator
src/database/           SQLAlchemy star schema + db_manager
src/visualization/      Plotly chart builders
dashboard/app.py        Streamlit app (6 tabs)
notebooks/              exploration → features → modeling walkthroughs
tests/                  66 pytest cases incl. ground-truth recovery test
docs/                   methodology.md, architecture.md
```

## Tech stack

Python · pandas · NumPy · SciPy (sparse) · scikit-learn (RidgeCV/LassoCV/
KMeans) · SQLAlchemy (SQLite/PostgreSQL) · spotipy · musicbrainzngs ·
billboard.py · rapidfuzz · Streamlit · Plotly · pytest

## Limitations & future work

- Spotify popularity is a **rolling** metric, not historical — older
  catalogues are penalised; a longevity bonus partially compensates.
- WAR is **associational**, not causal: stars may get first pick of songs.
- Perfectly collinear collaborators (always-together duos) split credit.
- Future: play-count panels for a true time-series target, position-style
  adjustments per role, hierarchical shrinkage by genre, credit coverage
  beyond MusicBrainz (Genius, Discogs).

## License

[MIT](LICENSE)
