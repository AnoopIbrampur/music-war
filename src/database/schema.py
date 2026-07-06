"""Star schema for the Music WAR warehouse (SQLAlchemy 2.0 ORM).

Fact table: track performance. Dimensions: track, artist, producer,
songwriter. Bridge tables handle the many-to-many credit relationships.
Works identically on SQLite (dev) and PostgreSQL (prod) via DATABASE_URL.
"""

from __future__ import annotations

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class FactTrackPerformance(Base):
    __tablename__ = "fact_track_performance"

    track_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("dim_track.track_id"), primary_key=True
    )
    spotify_popularity: Mapped[float | None] = mapped_column(Float)
    billboard_peak_position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    billboard_weeks_on_chart: Mapped[int | None] = mapped_column(Integer, nullable=True)
    composite_success_score: Mapped[float | None] = mapped_column(Float)

    track: Mapped["DimTrack"] = relationship(back_populates="performance")


class DimTrack(Base):
    __tablename__ = "dim_track"

    track_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    track_name: Mapped[str] = mapped_column(String(512))
    album_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    album_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    release_date: Mapped[str | None] = mapped_column(String(16), nullable=True)
    release_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    explicit: Mapped[bool] = mapped_column(Boolean, default=False)
    primary_genre: Mapped[str | None] = mapped_column(String(64), nullable=True)
    danceability: Mapped[float | None] = mapped_column(Float, nullable=True)
    energy: Mapped[float | None] = mapped_column(Float, nullable=True)
    valence: Mapped[float | None] = mapped_column(Float, nullable=True)
    tempo: Mapped[float | None] = mapped_column(Float, nullable=True)
    loudness: Mapped[float | None] = mapped_column(Float, nullable=True)
    speechiness: Mapped[float | None] = mapped_column(Float, nullable=True)
    acousticness: Mapped[float | None] = mapped_column(Float, nullable=True)
    instrumentalness: Mapped[float | None] = mapped_column(Float, nullable=True)
    key: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mode: Mapped[int | None] = mapped_column(Integer, nullable=True)
    time_signature: Mapped[int | None] = mapped_column(Integer, nullable=True)

    performance: Mapped[FactTrackPerformance | None] = relationship(back_populates="track")
    artist_links: Mapped[list["BridgeTrackArtist"]] = relationship(back_populates="track")


class DimArtist(Base):
    __tablename__ = "dim_artist"

    artist_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    artist_name: Mapped[str] = mapped_column(String(256))
    primary_genre: Mapped[str | None] = mapped_column(String(64), nullable=True)
    followers: Mapped[int | None] = mapped_column(Integer, nullable=True)
    artist_popularity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    career_start_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tracks_in_dataset: Mapped[int | None] = mapped_column(Integer, nullable=True)

    track_links: Mapped[list["BridgeTrackArtist"]] = relationship(back_populates="artist")


class DimProducer(Base):
    __tablename__ = "dim_producer"

    producer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    producer_name: Mapped[str] = mapped_column(String(256))
    total_tracks_produced: Mapped[int | None] = mapped_column(Integer, nullable=True)
    primary_genre_produced: Mapped[str | None] = mapped_column(String(64), nullable=True)


class DimSongwriter(Base):
    __tablename__ = "dim_songwriter"

    songwriter_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    songwriter_name: Mapped[str] = mapped_column(String(256))
    total_tracks_written: Mapped[int | None] = mapped_column(Integer, nullable=True)


class BridgeTrackArtist(Base):
    __tablename__ = "bridge_track_artist"

    track_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("dim_track.track_id"), primary_key=True
    )
    artist_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("dim_artist.artist_id"), primary_key=True
    )
    role: Mapped[str] = mapped_column(String(32))  # primary_artist / featured_artist
    billing_order: Mapped[int] = mapped_column(Integer, default=1)

    track: Mapped[DimTrack] = relationship(back_populates="artist_links")
    artist: Mapped[DimArtist] = relationship(back_populates="track_links")


class BridgeTrackProducer(Base):
    __tablename__ = "bridge_track_producer"

    track_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("dim_track.track_id"), primary_key=True
    )
    producer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("dim_producer.producer_id"), primary_key=True
    )


class BridgeTrackSongwriter(Base):
    __tablename__ = "bridge_track_songwriter"

    track_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("dim_track.track_id"), primary_key=True
    )
    songwriter_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("dim_songwriter.songwriter_id"), primary_key=True
    )
