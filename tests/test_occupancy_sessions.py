"""Focused tests for occupancy session lifecycle (database/repository.py).

Run with:

    python3 tests/test_occupancy_sessions.py

Covers: closing all sessions at the end of a processing run with the last
video timestamp, never storing ended_at earlier than started_at, restarts
from the beginning not touching previous runs' rows, history preservation,
and the one-active-session partial unique index.
"""

import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from database import connection as db_connection
from database.repository import (
    close_all_sessions,
    close_leftover_sessions,
    close_session,
    start_session,
    sync_tables,
    update_session_max,
)


class OccupancySessionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_path = db_connection.DATABASE_PATH
        db_connection.DATABASE_PATH = Path(self._tmp.name) / "test.db"
        db_connection.init_database()
        sync_tables([("T-01", 4), ("T-02", 4)])

    def tearDown(self):
        db_connection.DATABASE_PATH = self._old_path
        self._tmp.cleanup()

    def _rows(self, sql, *params):
        with closing(db_connection.get_connection()) as conn:
            return conn.execute(sql, params).fetchall()

    def _sessions(self, name):
        return self._rows(
            """
            SELECT s.started_at, s.ended_at, s.max_people
            FROM occupancy_sessions AS s
            JOIN tables AS t ON t.id = s.table_id
            WHERE t.name = ?
            ORDER BY s.id
            """,
            name,
        )

    def test_end_of_processing_closes_active_sessions(self):
        # People are still seated when processing ends: sessions must close
        # at the last processed video timestamp (12.5), not stay open.
        start_session("T-01", 2.0, 2)
        start_session("T-02", 3.0, 1)
        close_all_sessions(12.5)
        self.assertEqual(self._sessions("T-01"), [(2.0, 12.5, 2)])
        self.assertEqual(self._sessions("T-02"), [(3.0, 12.5, 1)])

    def test_close_is_never_earlier_than_start(self):
        start_session("T-01", 10.0, 2)
        close_session("T-01", 4.0)  # earlier than started_at -> clamped
        self.assertEqual(self._sessions("T-01"), [(10.0, 10.0, 2)])

        start_session("T-02", 50.0, 1)
        close_all_sessions(2.0)  # sweep with an earlier timestamp -> clamped
        self.assertEqual(self._sessions("T-02"), [(50.0, 50.0, 1)])

    def test_restart_records_its_own_rows(self):
        # Run 1: one visit, closed at the end of processing.
        start_session("T-01", 0.2, 2)
        update_session_max("T-01", 4)
        close_all_sessions(1.96)

        # Run 2 (restart from the beginning): a fresh row; run 1's row is
        # left exactly as it was and is never updated again.
        start_session("T-01", 0.2, 1)
        self.assertEqual(
            self._sessions("T-01"),
            [(0.2, 1.96, 4), (0.2, None, 1)],
        )
        update_session_max("T-01", 3)
        self.assertEqual(self._sessions("T-01"), [(0.2, 1.96, 4), (0.2, None, 3)])
        close_all_sessions(1.96)
        self.assertEqual(self._sessions("T-01"), [(0.2, 1.96, 4), (0.2, 1.96, 3)])

    def test_restart_with_leftover_active_session(self):
        # A crashed run left a session open at video t=50. The new run starts
        # at t=1: the leftover is closed (never ended < started) and the new
        # run gets its own active row that later updates affect exclusively.
        start_session("T-01", 50.0, 2)
        start_session("T-01", 1.0, 1)
        self.assertEqual(
            self._sessions("T-01"),
            [(50.0, 50.0, 2), (1.0, None, 1)],
        )
        update_session_max("T-01", 5)
        self.assertEqual(self._sessions("T-01"), [(50.0, 50.0, 2), (1.0, None, 5)])

    def test_run_start_closes_leftover_active_sessions(self):
        # A crashed run left sessions open on two tables. Beginning the new
        # run closes them at their own start times (never ended < started)
        # and preserves every historical row.
        start_session("T-01", 50.0, 2)
        start_session("T-02", 80.0, 3)
        close_leftover_sessions()
        self.assertEqual(self._sessions("T-01"), [(50.0, 50.0, 2)])
        self.assertEqual(self._sessions("T-02"), [(80.0, 80.0, 3)])
        # The new run's sessions start fresh and are unaffected.
        start_session("T-01", 1.0, 1)
        self.assertEqual(self._sessions("T-01"), [(50.0, 50.0, 2), (1.0, None, 1)])

    def test_one_active_session_index_still_enforces(self):
        start_session("T-01", 1.0, 1)
        with closing(db_connection.get_connection()) as conn:
            with conn:
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute(
                        """
                        INSERT INTO occupancy_sessions
                            (table_id, started_at, max_people)
                        VALUES ((SELECT id FROM tables WHERE name = 'T-01'), 2.0, 1)
                        """
                    )
        close_session("T-01", 3.0)
        start_session("T-01", 4.0, 1)  # fine once the previous one is closed
        self.assertEqual(len(self._sessions("T-01")), 2)

    def test_history_is_never_deleted(self):
        start_session("T-01", 1.0, 1)
        close_session("T-01", 2.0)
        start_session("T-01", 3.0, 2)
        close_all_sessions(4.0)
        start_session("T-01", 5.0, 1)
        close_all_sessions(6.0)
        sync_tables([("T-01", 4), ("T-02", 4), ("T-03", 2)])  # resync keeps rows
        self.assertEqual(
            self._sessions("T-01"),
            [(1.0, 2.0, 1), (3.0, 4.0, 2), (5.0, 6.0, 1)],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
