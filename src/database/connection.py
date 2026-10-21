"""SQLite database initialization and connections for the restaurant."""

import sqlite3
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "restaurant.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tables (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    capacity INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS occupancy_sessions (
    id INTEGER PRIMARY KEY,
    table_id INTEGER NOT NULL REFERENCES tables (id),
    started_at REAL NOT NULL,
    ended_at REAL,
    max_people INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS one_active_session_per_table
    ON occupancy_sessions (table_id)
    WHERE ended_at IS NULL;
"""


def get_connection() -> sqlite3.Connection:
    """Open a database connection with foreign-key enforcement enabled.

    ``PRAGMA foreign_keys`` is per-connection in SQLite, so it must be
    enabled on every connection (not only at initialization).
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database() -> None:
    """Create the database file and schema if they do not exist yet."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(get_connection()) as conn:
        with conn:
            conn.executescript(SCHEMA)
