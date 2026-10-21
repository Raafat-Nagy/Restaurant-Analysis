"""Database operations for tables and occupancy sessions."""

from contextlib import closing

from database.connection import get_connection


def sync_tables(tables: list[tuple[str, int]]) -> None:
    """Insert new tables and refresh capacities from the regions config.

    Existing tables are updated in place and nothing is ever deleted, so
    historical occupancy sessions survive re-synchronization.
    """
    with closing(get_connection()) as conn:
        with conn:
            for name, capacity in tables:
                conn.execute(
                    """
                    INSERT INTO tables (name, capacity) VALUES (?, ?)
                    ON CONFLICT (name) DO UPDATE SET capacity = excluded.capacity
                    """,
                    (name, capacity),
                )


def _close_session(conn, name: str, ended_at: float) -> None:
    """Close the table's active session on an existing connection.

    ``ended_at`` is clamped to ``started_at`` so a session can never be
    closed with an end timestamp earlier than its start.
    """
    conn.execute(
        """
        UPDATE occupancy_sessions
        SET ended_at = MAX(started_at, ?)
        WHERE ended_at IS NULL
          AND table_id = (SELECT id FROM tables WHERE name = ?)
        """,
        (ended_at, name),
    )


def _close_all(conn, ended_at: float) -> None:
    """Close every active session on an existing connection (same clamp)."""
    conn.execute(
        """
        UPDATE occupancy_sessions
        SET ended_at = MAX(started_at, ?)
        WHERE ended_at IS NULL
        """,
        (ended_at,),
    )


def start_session(name: str, started_at: float, max_people: int) -> None:
    """Record a 0 -> positive occupancy transition as a new session.

    Timestamps are video-timeline seconds. Each processing run records its
    own rows: a session left active by a previous run (e.g. after a crash)
    is closed first, so old rows can never absorb or block the new run's
    session, and nothing about an old row is ever updated afterwards.
    """
    with closing(get_connection()) as conn:
        with conn:
            _close_session(conn, name, started_at)
            conn.execute(
                """
                INSERT INTO occupancy_sessions (table_id, started_at, max_people)
                SELECT id, ?, ? FROM tables WHERE name = ?
                """,
                (started_at, max_people, name),
            )


def update_session_max(name: str, max_people: int) -> None:
    """Raise ``max_people`` of the active session when the count exceeds it."""
    with closing(get_connection()) as conn:
        with conn:
            conn.execute(
                """
                UPDATE occupancy_sessions
                SET max_people = ?
                WHERE ended_at IS NULL
                  AND max_people < ?
                  AND table_id = (SELECT id FROM tables WHERE name = ?)
                """,
                (max_people, max_people, name),
            )


def close_session(name: str, ended_at: float) -> None:
    """Close the table's active session with a video-timeline end time."""
    with closing(get_connection()) as conn:
        with conn:
            _close_session(conn, name, ended_at)


def close_all_sessions(ended_at: float) -> None:
    """Close every active session at the end of a processing run.

    Called with the last processed video timestamp, so sessions of people
    still seated when the video ends are closed there and never left open.
    """
    with closing(get_connection()) as conn:
        with conn:
            _close_all(conn, ended_at)


def close_leftover_sessions() -> None:
    """Close active sessions left behind by an interrupted processing run.

    Called at the start of every processing run. Each leftover row is
    preserved and closed at its own start time, so ``ended_at`` is never
    earlier than ``started_at``.
    """
    with closing(get_connection()) as conn:
        with conn:
            conn.execute(
                """
                UPDATE occupancy_sessions
                SET ended_at = started_at
                WHERE ended_at IS NULL
                """
            )
