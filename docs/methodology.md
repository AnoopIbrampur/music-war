# Music WAR — Statistical Methodology

## The idea, borrowed from baseball

In sabermetrics, **Wins Above Replacement (WAR)** answers: *how many more
games does a team win with this player instead of a freely available
replacement-level player?* The key move is defining value **relative to a
baseline**, not in absolute terms, and **controlling for context** (park
effects, era, position).

Music WAR asks the same question: *how many more success-score points does
a track earn with this artist / producer / songwriter on it, instead of a
replacement-level one?* — controlling for genre, era, sound profile, and
everyone else credited on the track.

## The target: composite success score

Each track gets a 0–100 score:

| Component | Weight | Notes |
|---|---|---|
| Spotify popularity | 40% | rolling 0–100 metric |
| Billboard peak (inverse-normalised) | 30% | rank 1 → 100 points |
| Billboard weeks on chart | 20% | capped at 52 weeks |
| Longevity bonus | 10% | old tracks that remain popular |

Tracks that never charted are scored on popularity + longevity alone,
multiplied by a 0.95 penalty. Without the penalty, the model would treat
"never charted" as neutral rather than mildly negative, inflating the
apparent value of artists whose catalogues never reach the charts.

## The design matrix

One row per track. Columns:

- **People indicators** — one binary column per *eligible* artist,
  producer, and songwriter. Eligibility: ≥ 5 credits on tracks with
  popularity > 20. Featured artists enter with weight **0.5** instead of
  1.0 (a guest verse is roughly half a top billing).
- **Controls** — one-hot genre (~20 parents), era (5 buckets), tempo and
  duration buckets, audio-feature K-means cluster ("sound profile"),
  explicit flag, crew size.

Everything is stored as a `scipy.sparse.csr_matrix`: with thousands of
people columns the matrix is >99% zeros, and sparse storage keeps the
regression tractable.

## Replacement level

People below the eligibility threshold get **no column**. Their effect is
absorbed into the intercept and control coefficients — which is exactly
the definition of replacement level: the expected score of a track with
the same genre, era, and sound, made by people too obscure to model
individually. A person's coefficient is therefore their lift *above that
baseline* by construction.

## Why ridge regression

The matrix is wide (thousands of columns) relative to rows, and many
people have only a handful of credits. OLS would assign wild coefficients
to small-sample people (the "actor with two movies" problem). Ridge (L2)
shrinks all coefficients toward zero in proportion to how weakly the data
supports them — a person with 5 tracks needs a much stronger signal to
earn a large WAR than one with 100 tracks. Regularisation strength α is
chosen by cross-validation over {0.01, 0.1, 1, 10, 100}.

A parallel **lasso** (L1) run zeroes out weak contributors entirely.
People with large ridge WAR who also survive lasso are the most robust
findings; ridge-only stars deserve skepticism.

## Uncertainty

Bootstrap CIs: resample tracks with replacement, refit ridge at the chosen
α, repeat (default 100×), take the 2.5th/97.5th percentile of each
coefficient. Wide intervals flag WAR values resting on few tracks.

## Validation against planted ground truth

The test suite generates synthetic data with *known* latent talent effects
and verifies the pipeline recovers them (correlation > 0.8 between
estimated and true artist effects). This guards against wiring bugs that
plain goodness-of-fit metrics would miss.

## The baseline comparison

We also fit a controls-only model (genre + era + audio, no people). The
gap in test R² between the full and baseline models measures how much
individual talent explains beyond structural factors — the project's
headline quantity.

## Assumptions and limitations

1. **Association, not causation.** A high WAR means tracks with this
   person outperform expectations; it cannot separate talent from
   selection (stars get first pick of good songs).
2. **Collinearity.** Duos who always work together share one effective
   coefficient split arbitrarily between them; ridge splits it evenly.
3. **Popularity is a rolling metric.** Spotify popularity reflects
   *current* listening, so older catalogues are systematically penalised;
   the longevity bonus only partially compensates.
4. **Credit coverage is incomplete.** MusicBrainz producer/writer credits
   exist for roughly 60% of tracks; missing credits bias those people's
   WAR toward zero.
5. **Survivorship in playlists.** Seed playlists over-sample successful
   tracks; replacement level is therefore "replacement level among
   playlist-worthy music", not among all recorded music.

## Differences from baseball WAR

Baseball WAR is denominated in wins via a run-to-win conversion and uses
an explicitly defined replacement team (~.294 winning percentage). Music
WAR is denominated in composite-score points, and replacement level is
estimated (the sub-threshold pool) rather than defined. Baseball also has
clean position adjustments; our analogue is the billing weight (primary
vs featured), which is cruder.
