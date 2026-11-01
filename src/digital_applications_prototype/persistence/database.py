"""Async SQLite setup; no model call holds a database session open."""

from pathlib import Path
from typing import Any

from sqlalchemy import URL, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .models import Base


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.engine = create_async_engine(
            URL.create("sqlite+aiosqlite", database=str(self.path)),
            connect_args={"timeout": 30},
        )
        self.sessions = async_sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession,
        )

        @event.listens_for(self.engine.sync_engine, "connect")
        def configure_connection(connection: Any, _record: Any) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            # FULL sync preserves committed work across power loss as well as crashes.
            cursor.execute("PRAGMA synchronous=FULL")
            cursor.close()

    async def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()


async def initialize_database(path: Path = Path("data/derived/stage1.sqlite")) -> Database:
    """Create the parent directory/schema and return an owned database handle."""
    database = Database(path)
    try:
        await database.initialize()
    except BaseException:
        await database.close()
        raise
    return database
