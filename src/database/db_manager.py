"""Database access layer.

Creates the engine from DATABASE_URL (SQLite by default, PostgreSQL in
production), builds the schema, and provides idempotent bulk loaders —
re-running the pipeline upserts rather than duplicating rows.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session, sessionmaker

from config import settings
from src.database import schema

logger = logging.getLogger(__name__)


class DBManager:
    def __init__(self, url: str | None = None) -> None:
        self.engine = create_engine(url or settings.DATABASE_URL)
        self._session_factory = sessionmaker(bind=self.engine)

    def create_schema(self) -> None:
        schema.Base.metadata.create_all(self.engine)

    def drop_schema(self) -> None:
        schema.Base.metadata.drop_all(self.engine)

    @contextmanager
    def session(self):
        s: Session = self._session_factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ------------------------------------------------------------- loading
    def replace_table(self, model, df: pd.DataFrame) -> int:
        """Idempotent load: wipe the table and insert the dataframe's rows.

        Only columns that exist on the model are inserted; extras in the
        dataframe are ignored so callers can pass enriched frames directly.
        """
        columns = {c.name for c in model.__table__.columns}
        records = (
            df[[c for c in df.columns if c in columns]]
            .where(df.notna(), None)
            .to_dict(orient="records")
        )
        with self.session() as s:
            s.execute(delete(model))
            if records:
                s.bulk_insert_mappings(model.__mapper__, records)
        logger.info("Loaded %d rows into %s", len(records), model.__tablename__)
        return len(records)

    def read_table(self, model) -> pd.DataFrame:
        return pd.read_sql_table(model.__tablename__, self.engine)


def load_star_schema(db: DBManager, tracks: pd.DataFrame, artists: pd.DataFrame,
                     track_artists: pd.DataFrame, producers: pd.DataFrame,
                     songwriters: pd.DataFrame, track_producers: pd.DataFrame,
                     track_songwriters: pd.DataFrame) -> dict[str, int]:
    """Load all star-schema tables in FK-safe order. Returns row counts."""
    db.create_schema()
    counts = {
        "dim_track": db.replace_table(schema.DimTrack, tracks),
        "dim_artist": db.replace_table(schema.DimArtist, artists),
        "dim_producer": db.replace_table(schema.DimProducer, producers),
        "dim_songwriter": db.replace_table(schema.DimSongwriter, songwriters),
        "bridge_track_artist": db.replace_table(schema.BridgeTrackArtist, track_artists),
        "bridge_track_producer": db.replace_table(schema.BridgeTrackProducer, track_producers),
        "bridge_track_songwriter": db.replace_table(schema.BridgeTrackSongwriter, track_songwriters),
        "fact_track_performance": db.replace_table(schema.FactTrackPerformance, tracks),
    }
    return counts
