"""Music WAR interactive dashboard.

Launch with:  streamlit run dashboard/app.py
Reads pipeline outputs from data/processed/ and data/models/. If they are
missing, offers to generate them from synthetic demo data on the spot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.visualization import charts  # noqa: E402

st.set_page_config(page_title="Music WAR", page_icon="🎵", layout="wide")


@st.cache_data
def load_data():
    tracks = pd.read_parquet(settings.PROCESSED_DIR / "tracks_features.parquet")
    war = pd.read_parquet(settings.MODELS_DIR / "war_results.parquet")
    bridge = pd.read_parquet(settings.PROCESSED_DIR / "bridge_track_artist.parquet")
    artists = pd.read_parquet(settings.PROCESSED_DIR / "dim_artist.parquet")
    metrics = json.loads((settings.MODELS_DIR / "model_metrics.json").read_text())
    return tracks, war, bridge, artists, metrics


def ensure_data() -> bool:
    if (settings.MODELS_DIR / "war_results.parquet").exists():
        return True
    st.warning("No pipeline output found. Run `python -m src.pipeline --demo` first, "
               "or generate demo data now.")
    if st.button("Generate demo data (≈30s)"):
        from src.pipeline import run

        with st.spinner("Running demo pipeline..."):
            run(demo=True)
        st.cache_data.clear()
        st.rerun()
    return False


def leaderboard_tab(war: pd.DataFrame, role: str, title: str) -> None:
    df = war[war["role"] == role].copy()
    st.subheader(title)

    col1, col2, col3 = st.columns(3)
    view = col1.radio("View", ["All", "Top 50 Positive", "Bottom 50 Negative"],
                      horizontal=True, key=f"view_{role}")
    min_tracks = col2.slider("Minimum tracks", 5, 50, 5, key=f"min_{role}")
    search = col3.text_input("Search by name", key=f"search_{role}")

    df = df[df["n_tracks"] >= min_tracks]
    if search:
        df = df[df["name"].str.contains(search, case=False, na=False)]
    if view == "Top 50 Positive":
        df = df.nlargest(50, "war_per_track")
    elif view == "Bottom 50 Negative":
        df = df.nsmallest(50, "war_per_track")

    display_cols = [c for c in ["name", "war_per_track", "total_war", "n_tracks",
                                "percentile_rank", "survives_lasso",
                                "war_ci_low", "war_ci_high"] if c in df.columns]
    st.dataframe(
        df[display_cols].round(2), use_container_width=True, hide_index=True,
        column_config={"war_per_track": st.column_config.NumberColumn("WAR / track")},
    )
    st.download_button(
        f"Download {role} leaderboard (CSV)",
        df[display_cols].to_csv(index=False),
        file_name=f"{role}_war_leaderboard.csv",
        key=f"dl_{role}",
    )
    if not df.empty:
        st.plotly_chart(charts.war_bar(df, title=f"{title}: top & bottom 20"),
                        use_container_width=True)


def main() -> None:
    st.title("🎵 Music WAR — Wins Above Replacement for Music")
    st.caption("How many success-score points does each artist, producer, and "
               "songwriter add above a replacement-level substitute?")

    if not ensure_data():
        return
    tracks, war, bridge, artists, metrics = load_data()

    # Producer WAR only appears when the data source actually has credits
    # (the bulk Spotify export has none); it stays out of the tab bar entirely
    # rather than showing an empty table.
    has_producers = (war["role"] == "producer").any()
    tab_names = ["Overview", "Artist WAR", "Beyond Popularity", "Sound & Success"]
    if has_producers:
        tab_names.append("Producer WAR")
    tab_names += ["Artist Deep Dive", "Dream Team Builder", "Methodology"]
    tab = dict(zip(tab_names, st.tabs(tab_names)))

    # ---------------------------------------------------------------- overview
    with tab["Overview"]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tracks analyzed", f"{len(tracks):,}")
        c2.metric("Eligible artists", f"{(war['role'] == 'artist').sum():,}")
        c3.metric("Genres", f"{tracks['primary_genre'].nunique():,}")
        # Show credit counts only when the data source has them; otherwise a
        # more useful summary stat than a hard-coded zero.
        if has_producers:
            c4.metric("Eligible producers", f"{(war['role'] == 'producer').sum():,}")
        else:
            c4.metric("Median popularity", f"{tracks['spotify_popularity'].median():.0f}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model R² (test)", f"{metrics['r2_test']:.3f}")
        m2.metric("Baseline R² (no people)", f"{metrics['baseline_r2_test']:.3f}")
        m3.metric("Talent lift (ΔR²)", f"{metrics['talent_lift_r2']:.3f}")
        m4.metric("RMSE", f"{metrics['rmse_test']:.2f}")

        left, right = st.columns(2)
        left.plotly_chart(charts.score_distribution(tracks), use_container_width=True)
        right.plotly_chart(charts.genre_breakdown(tracks), use_container_width=True)
        # Era analysis only makes sense when release dates are present; the
        # bulk Spotify export has none, so show genre popularity instead.
        if tracks["era"].nunique() > 1:
            st.plotly_chart(charts.era_breakdown(tracks), use_container_width=True)
        else:
            st.plotly_chart(charts.genre_popularity(tracks), use_container_width=True)

    # ---------------------------------------------------------- leaderboards
    with tab["Artist WAR"]:
        leaderboard_tab(war, "artist", "Artist WAR Leaderboard")

    # ----------------------------------------------------- beyond popularity
    with tab["Beyond Popularity"]:
        aw = war[war["role"] == "artist"].copy()
        st.subheader("Beyond popularity — what WAR sees that a hit-count doesn't")
        st.caption("WAR controls for genre, sound, and collaborators, so it "
                   "separates artists who *drive* success from those who ride it.")

        if "overperformance" not in aw or aw["overperformance"].notna().sum() == 0:
            st.info("Run `python -m src.pipeline --source bulk` to generate the "
                    "enriched artist metrics this tab needs.")
        else:
            corr = aw["war_per_track"].corr(aw["avg_popularity"])
            st.metric("Correlation of WAR with raw popularity", f"{corr:.2f}",
                      help="High but far from 1.0 — WAR is not just fame")
            st.plotly_chart(charts.war_vs_popularity(aw), use_container_width=True)

            min_tracks = st.slider("Minimum tracks", 5, 50, 20, key="bp_min")
            elig = aw[aw["n_tracks"] >= min_tracks]
            cols = ["name", "war_per_track", "avg_popularity", "overperformance",
                    "n_tracks", "genre"]
            cols = [c for c in cols if c in elig.columns]

            left, right = st.columns(2)
            left.markdown("**Overperformers** — high WAR for their fame")
            left.dataframe(
                elig.nlargest(12, "overperformance")[cols].round(1),
                use_container_width=True, hide_index=True,
            )
            right.markdown("**Coattail riders** — popular but low WAR")
            right.dataframe(
                elig.nsmallest(12, "overperformance")[cols].round(1),
                use_container_width=True, hide_index=True,
            )

            st.divider()
            g1, g2 = st.columns(2)
            g1.markdown("**Best artist per genre** (highest WAR, ≥10 tracks)")
            if "genre" in aw.columns:
                per_genre = (
                    aw[aw["n_tracks"] >= 10]
                    .dropna(subset=["genre"])
                    .sort_values("war_per_track", ascending=False)
                    .groupby("genre")
                    .first()
                    .reset_index()[["genre", "name", "war_per_track", "n_tracks"]]
                    .sort_values("war_per_track", ascending=False)
                )
                g1.dataframe(per_genre.round(1), use_container_width=True, hide_index=True)

            g2.markdown("**Most consistent hitmakers** (high WAR, low score variance)")
            if "score_std" in aw.columns:
                reliable = aw[(aw["n_tracks"] >= 20) & (aw["war_per_track"] > 0)].copy()
                reliable = reliable.nsmallest(12, "score_std")[
                    ["name", "war_per_track", "score_std", "n_tracks"]
                ]
                g2.dataframe(reliable.round(1), use_container_width=True, hide_index=True)

    # ------------------------------------------------------- sound & success
    with tab["Sound & Success"]:
        st.subheader("Sound & Success")
        st.caption("What the audio itself says about popularity — the dimension "
                   "this dataset is richest in.")

        inst = tracks["instrumentalness"].corr(tracks["spotify_popularity"])
        exp_lift = (tracks.loc[tracks["is_explicit"] == 1, "spotify_popularity"].mean()
                    - tracks.loc[tracks["is_explicit"] == 0, "spotify_popularity"].mean())
        best_genre = tracks.groupby("primary_genre")["spotify_popularity"].mean().idxmax()
        k1, k2, k3 = st.columns(3)
        k1.metric("Instrumentalness ↔ popularity", f"{inst:+.2f}",
                  help="Strongest single audio correlate — vocals win")
        k2.metric("Explicit popularity lift", f"{exp_lift:+.1f} pts")
        k3.metric("Most popular genre", best_genre)

        st.plotly_chart(charts.audio_popularity_corr(tracks), use_container_width=True)
        st.info("Every audio feature correlates only weakly with popularity "
                "(|r| < 0.2). That's the point: sound alone doesn't make a hit — "
                "which is why *who* is on the track (Artist WAR) explains far more.")

        left, right = st.columns(2)
        left.plotly_chart(charts.sound_profile_clusters(tracks), use_container_width=True)
        right.plotly_chart(
            charts.factor_effect(tracks, "is_explicit", "Explicit vs clean",
                                 labels={0: "clean", 1: "explicit"}),
            use_container_width=True,
        )
        crew = tracks.assign(crew=tracks["num_artists_on_track"].clip(upper=4))
        st.plotly_chart(
            charts.factor_effect(crew, "crew", "Popularity by number of credited artists"),
            use_container_width=True,
        )

    # ---------------------------------------------------------- producer WAR
    if has_producers:
        with tab["Producer WAR"]:
            leaderboard_tab(war, "producer", "Producer WAR Leaderboard")

    # ------------------------------------------------------------ deep dive
    with tab["Artist Deep Dive"]:
        artist_war = war[war["role"] == "artist"]
        pick = st.selectbox("Choose an artist", artist_war["name"].sort_values())
        row = artist_war[artist_war["name"] == pick].iloc[0]
        entity_id = row["entity_id"]

        track_ids = bridge.loc[bridge["artist_id"] == entity_id, "track_id"]
        their_tracks = tracks[tracks["track_id"].isin(track_ids)]

        c1, c2, c3 = st.columns(3)
        c1.metric("WAR per track", f"{row['war_per_track']:+.2f}")
        c2.metric("Tracks in dataset", int(row["n_tracks"]))
        c3.metric("Percentile", f"{row['percentile_rank']:.0f}%")

        left, right = st.columns(2)
        if not their_tracks.empty:
            left.plotly_chart(charts.artist_radar(their_tracks), use_container_width=True)
            right.plotly_chart(charts.popularity_timeline(their_tracks), use_container_width=True)
            st.subheader("Most successful tracks")
            ranked = their_tracks.sort_values("composite_success_score", ascending=False)
            st.dataframe(
                ranked[charts.track_table_columns(ranked)].head(10),
                use_container_width=True, hide_index=True,
            )

    # ----------------------------------------------------------- dream team
    with tab["Dream Team Builder"]:
        st.subheader("Build your dream hit-song team")
        st.caption("Predicted score = replacement-level baseline + the primary "
                   "artist's WAR + half the featured artist's WAR. Producer and "
                   "songwriter terms appear automatically when the data source "
                   "carries credits.")
        a = war[war["role"] == "artist"]
        p = war[war["role"] == "producer"]
        w = war[war["role"] == "songwriter"]

        c1, c2 = st.columns(2)
        primary = c1.selectbox("Primary artist", a["name"].sort_values())
        featured = c2.selectbox("Featured artist", ["(none)"] + a["name"].sort_values().tolist())

        producer = writer = "(none)"
        if not p.empty or not w.empty:
            c3, c4 = st.columns(2)
            if not p.empty:
                producer = c3.selectbox("Producer", ["(none)"] + p["name"].sort_values().tolist())
            if not w.empty:
                writer = c4.selectbox("Songwriter", ["(none)"] + w["name"].sort_values().tolist())

        score = metrics["replacement_level"]
        breakdown = [("Replacement-level baseline", metrics["replacement_level"])]
        score_add = a.loc[a["name"] == primary, "war_per_track"].iloc[0]
        score += score_add
        breakdown.append((f"{primary} (primary)", score_add))
        if featured != "(none)":
            add = settings.FEATURED_BILLING_WEIGHT * a.loc[a["name"] == featured, "war_per_track"].iloc[0]
            score += add
            breakdown.append((f"{featured} (featured, ×{settings.FEATURED_BILLING_WEIGHT})", add))
        if producer != "(none)":
            add = p.loc[p["name"] == producer, "war_per_track"].iloc[0]
            score += add
            breakdown.append((f"{producer} (producer)", add))
        if writer != "(none)":
            add = w.loc[w["name"] == writer, "war_per_track"].iloc[0]
            score += add
            breakdown.append((f"{writer} (songwriter)", add))

        score = max(0.0, min(100.0, score))
        st.metric("Predicted composite success score", f"{score:.1f} / 100")
        st.dataframe(
            pd.DataFrame(breakdown, columns=["Component", "Points"]).round(2),
            use_container_width=True, hide_index=True,
        )
        st.caption(f"For reference, a replacement-level track sits at "
                   f"{metrics['replacement_level']:.1f}; the dataset's most popular "
                   f"tracks reach the mid-80s.")

    # ---------------------------------------------------------- methodology
    with tab["Methodology"]:
        st.markdown(
            f"""
### How Music WAR works

**WAR (Wins Above Replacement)** comes from baseball sabermetrics: a player's
value is measured against a hypothetical *replacement-level* player — freely
available talent. We port the idea to music.

1. Every track gets a **composite success score** (0–100) blending Spotify
   popularity (40%), Billboard peak (30%), weeks on chart (20%), and a
   longevity bonus (10%).
2. We build a **sparse design matrix**: one row per track, one binary column
   per eligible artist / producer / songwriter (≥5 credits), plus controls
   for genre, era, sound profile, tempo, duration, and crew size. Featured
   artists are down-weighted to {settings.FEATURED_BILLING_WEIGHT}.
3. A **ridge regression** (α chosen by cross-validation) predicts the score.
   Each person's coefficient is their WAR: the lift they add above a
   replacement-level person, *controlling for everything else on the track*.
4. A **lasso** run zeroes out weak contributors; people who survive both
   models are the most robust findings.

**Current model:** R² = {metrics['r2_test']:.3f} vs {metrics['baseline_r2_test']:.3f}
for a structure-only baseline — individual talent explains
**{metrics['talent_lift_r2']:.1%}** additional variance.

**Limitations:** popularity is a rolling (not historical) metric; credits
coverage is incomplete; coefficients are associations, not causal effects;
collinear collaborators (people who *always* work together) share credit.

Full write-up: `docs/methodology.md`.
"""
        )


main()
