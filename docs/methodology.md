# Methodology

## The idea, borrowed from baseball

WAR in baseball answers one question: how many more games does a team win with this player instead of a freely available replacement? Two things make it work. It measures value against a baseline rather than in a vacuum, and it controls for context like park effects and position.

Music WAR asks the same question about a track. How many more success-score points does a record earn with this artist on it instead of a replacement-level one, once you hold genre, sound, and the other people on the track constant?

## The target: a success score

Every track gets a 0–100 score. What goes into it depends on which data source you run.

On the default bulk export, there are no chart positions, so the score comes from Spotify popularity. When you run the API path with Billboard data, the score blends four pieces:

| Component | Weight | Notes |
|---|---|---|
| Spotify popularity | 40% | rolling 0–100 metric |
| Billboard peak, inverse-normalised | 30% | rank 1 becomes 100 points |
| Billboard weeks on chart | 20% | capped at 52 weeks |
| Longevity bonus | 10% | older tracks that still stream |

Tracks that never charted get scored on popularity and longevity alone, then multiplied by 0.95. Without that small penalty the model would read "never charted" as neutral instead of slightly below average, which would flatter artists whose catalogues never reach the chart.

## The design matrix

One row per track. The columns come in two kinds.

Most of them are a single on/off flag for each eligible artist, producer, and songwriter. Eligible means at least five credits on tracks with popularity above 20. Featured artists enter at weight 0.5 rather than 1.0, since a guest verse is worth roughly half a top billing.

The rest are controls: one-hot genre, era buckets, tempo and duration buckets, a k-means cluster over the audio features that stands in for the track's sound profile, an explicit flag, and how many artists are credited.

With thousands of people columns the matrix is more than 99% zeros, so it lives in a `scipy.sparse.csr_matrix`. That's what keeps the regression tractable.

## Replacement level

Anyone below the five-credit threshold gets no column of their own. Their effect folds into the intercept and the control coefficients. That's the definition of replacement level: the expected score of a track with the same genre, era, and sound, made by people too obscure to model on their own. So an artist's coefficient is already their lift above that baseline, by construction.

## Why ridge

The matrix is wide relative to how many rows it has, and plenty of artists show up on only a handful of tracks. Plain OLS would hand those small-sample artists wild coefficients, the music version of rating an actor off two movies. Ridge shrinks every coefficient toward zero in proportion to how weakly the data backs it, so an artist with five tracks needs a much stronger signal than one with a hundred to earn the same WAR. The penalty strength is picked by cross-validation over {0.01, 0.1, 1, 10, 100}.

A lasso runs next to it and zeroes out the weak effects entirely. Artists with a big ridge WAR who also survive the lasso are the safer calls; ridge-only stars deserve a raised eyebrow.

## Uncertainty

For confidence intervals I resample tracks with replacement, refit ridge at the chosen penalty, and repeat a hundred times, then take the 2.5th and 97.5th percentile of each coefficient. A wide interval is the tell that a WAR is resting on too few tracks.

## Checking the machinery

The test suite builds synthetic data with talent effects I plant on purpose, then confirms the pipeline recovers them (correlation above 0.8 between estimated and true effects). Goodness-of-fit on real data won't catch a wiring bug that quietly swaps two columns; this does.

## The baseline comparison

I also fit a controls-only model that knows genre, era, and sound but nothing about who made the track. The gap in test R² between that and the full model is the headline number: how much the people explain beyond the structural stuff. On the real data it's 0.46 against 0.16.

## Assumptions and limitations

WAR here is a correlation, not proof of cause. A high WAR means tracks with this artist beat expectations; it can't tell whether they're talented or just get first pick of good songs.

Collaborators who always record together share one effective coefficient, and ridge splits it evenly between them whether or not that's fair.

Spotify popularity is a rolling number. It reflects how a track streams now, so older catalogues get judged on today's listening and the longevity bonus only partly makes up for it.

On the API path, MusicBrainz credits cover only about 60% of tracks, which biases the WAR of anyone with missing credits toward zero. The bulk export has no credits at all, so producer and songwriter WAR only exist if you enrich it.

## How this differs from baseball WAR

Baseball WAR is denominated in wins through a runs-to-wins conversion, and its replacement level is a defined team that wins about 29% of its games. Music WAR is denominated in success-score points, and its replacement level is estimated from the pool of sub-threshold artists rather than fixed in advance. Baseball also has clean positional adjustments; the closest thing here is the primary-versus-featured billing weight, which is a blunter instrument.
