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
