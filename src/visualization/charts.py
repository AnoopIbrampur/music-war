"""Plotly chart builders shared by the dashboard and notebooks."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_dark"
POSITIVE = "#00CC96"
NEGATIVE = "#EF553B"


def score_distribution(tracks: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        tracks, x="composite_success_score", nbins=50,
        title="Composite Success Score Distribution", template=TEMPLATE,
    )
    fig.update_layout(xaxis_title="Composite success score", yaxis_title="Tracks")
    return fig


def genre_breakdown(tracks: pd.DataFrame) -> go.Figure:
    counts = tracks["primary_genre"].value_counts().reset_index()
    counts.columns = ["genre", "tracks"]
    fig = px.bar(counts, x="genre", y="tracks", title="Tracks by Genre",
                 template=TEMPLATE, color="genre")
    fig.update_layout(showlegend=False)
    return fig


def era_breakdown(tracks: pd.DataFrame) -> go.Figure:
    counts = tracks["era"].value_counts().reset_index()
    counts.columns = ["era", "tracks"]
    fig = px.bar(counts, x="era", y="tracks", title="Tracks by Era",
                 template=TEMPLATE, color="era")
    fig.update_layout(showlegend=False)
    return fig


AUDIO_COLS = [
    "danceability", "energy", "valence", "tempo", "loudness",
    "speechiness", "acousticness", "instrumentalness", "liveness",
]


def genre_popularity(tracks: pd.DataFrame) -> go.Figure:
    """Average popularity per genre — which genres over/under-perform."""
    g = (
        tracks.groupby("primary_genre")
        .agg(popularity=("spotify_popularity", "mean"), tracks=("track_id", "size"))
        .reset_index()
        .sort_values("popularity", ascending=False)
    )
    fig = px.bar(
        g, x="primary_genre", y="popularity", color="popularity",
        title="Average Popularity by Genre", template=TEMPLATE,
        color_continuous_scale="Viridis", hover_data=["tracks"],
    )
    fig.update_layout(xaxis_title="genre", yaxis_title="avg Spotify popularity",
                      coloraxis_showscale=False)
    return fig


def audio_popularity_corr(tracks: pd.DataFrame) -> go.Figure:
    """Correlation of each audio feature with popularity (the 'audio DNA')."""
    corrs = {c: tracks[c].corr(tracks["spotify_popularity"]) for c in AUDIO_COLS if c in tracks}
    df = pd.DataFrame({"feature": list(corrs), "correlation": list(corrs.values())})
    df = df.sort_values("correlation")
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in df["correlation"]]
    fig = go.Figure(go.Bar(x=df["correlation"], y=df["feature"], orientation="h",
                           marker_color=colors))
    fig.update_layout(
        template=TEMPLATE, title="What sonic traits predict popularity?",
        xaxis_title="correlation with Spotify popularity", height=380,
    )
    return fig


def sound_profile_clusters(tracks: pd.DataFrame) -> go.Figure:
    """Bubble chart: each K-means 'sound profile' by danceability, energy,
    and average popularity (bubble size = track count)."""
    c = (
        tracks.groupby("audio_cluster")
        .agg(
            danceability=("danceability", "mean"),
            energy=("energy", "mean"),
            acousticness=("acousticness", "mean"),
            popularity=("spotify_popularity", "mean"),
            tracks=("track_id", "size"),
        )
        .reset_index()
    )
    c["profile"] = "Cluster " + c["audio_cluster"].astype(str)
    fig = px.scatter(
        c, x="danceability", y="energy", size="tracks", color="popularity",
        text="profile", title="Sound profiles: which sounds sell",
        template=TEMPLATE, color_continuous_scale="Viridis", size_max=60,
        hover_data={"acousticness": ":.2f", "popularity": ":.1f", "tracks": True},
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(coloraxis_colorbar_title="avg pop")
    return fig


def factor_effect(tracks: pd.DataFrame, column: str, title: str, labels: dict | None = None) -> go.Figure:
    """Average popularity by a categorical/binary factor (explicit, crew size)."""
    grp = tracks.copy()
    if labels:
        grp[column] = grp[column].map(labels).fillna(grp[column])
    agg = (
        grp.groupby(column)
        .agg(popularity=("spotify_popularity", "mean"), tracks=("track_id", "size"))
        .reset_index()
    )
    fig = px.bar(agg, x=column, y="popularity", title=title, template=TEMPLATE,
                 color="popularity", color_continuous_scale="Viridis", hover_data=["tracks"])
    fig.update_layout(coloraxis_showscale=False, yaxis_title="avg popularity")
    return fig


def war_bar(war: pd.DataFrame, n: int = 20, title: str = "WAR Leaders") -> go.Figure:
    """Top-n and bottom-n WAR, green above replacement, red below."""
    top = war.nlargest(n, "war_per_track")
    bottom = war.nsmallest(n, "war_per_track")
    combined = pd.concat([top, bottom]).drop_duplicates("entity_id")
    combined = combined.sort_values("war_per_track")
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in combined["war_per_track"]]
    fig = go.Figure(
        go.Bar(x=combined["war_per_track"], y=combined["name"], orientation="h",
               marker_color=colors)
    )
    fig.update_layout(
        template=TEMPLATE, title=title, xaxis_title="WAR (score points per track)",
        height=max(500, 22 * len(combined)),
    )
    return fig


def artist_radar(track_rows: pd.DataFrame) -> go.Figure:
    """Radar chart of an artist's average audio profile."""
    features = ["danceability", "energy", "valence", "speechiness", "acousticness", "instrumentalness"]
    available = [f for f in features if f in track_rows.columns]
    values = track_rows[available].mean().tolist()
    fig = go.Figure(
        go.Scatterpolar(r=values + values[:1], theta=available + available[:1], fill="toself")
    )
    fig.update_layout(template=TEMPLATE, title="Audio Feature Profile",
                      polar={"radialaxis": {"range": [0, 1]}})
    return fig


def popularity_timeline(track_rows: pd.DataFrame) -> go.Figure:
    df = track_rows.sort_values("release_date")
    fig = px.scatter(
        df, x="release_date", y="composite_success_score", hover_name="track_name",
        title="Track Success Over Time", template=TEMPLATE,
        color="composite_success_score", color_continuous_scale="Viridis",
    )
    return fig
