"""Focused tests for OccupancyService (services/occupancy_service.py)."""

import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from database import connection as db_connection
from database.repository import start_session, sync_tables
from services.occupancy_service import OccupancyService


class OccupancyServiceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_path = db_connection.DATABASE_PATH
        db_connection.DATABASE_PATH = Path(self._tmp.name) / "test.db"
        db_connection.init_database()
        sync_tables([("T-01", 4), ("T-02", 4)])
        self.service = OccupancyService(["T-01", "T-02"])

    def tearDown(self):
        db_connection.DATABASE_PATH = self._old_path
        self._tmp.cleanup()

    def _sessions(self, name="T-01"):
        with closing(db_connection.get_connection()) as conn:
            return conn.execute(
                """
                SELECT s.started_at, s.ended_at, s.max_people
                FROM occupancy_sessions AS s
                JOIN tables AS t ON t.id = s.table_id
                WHERE t.name = ?
                ORDER BY s.id
                """,
                (name,),
            ).fetchall()

    def test_start_update_and_close(self):
        self.assertIsNone(self.service.update("T-01", 0, 0.0))
        self.assertEqual(self.service.update("T-01", 2, 0.2), 0.0)  # session starts
        self.assertEqual(self._sessions(), [(0.2, None, 2)])

        self.assertAlmostEqual(self.service.update("T-01", 3, 0.5), 0.3)  # new max
        self.service.update("T-01", 2, 0.9)  # below max -> max unchanged
        self.assertEqual(self._sessions(), [(0.2, None, 3)])

        self.assertIsNone(self.service.update("T-01", 0, 1.3))  # closes
        self.assertEqual(self._sessions(), [(0.2, 1.3, 3)])

    def test_end_run_closes_sessions_still_occupied(self):
        self.service.update("T-01", 2, 0.2)
        self.service.update("T-02", 1, 0.4)
        self.service.end_run(12.5)  # people still seated at the end
        self.assertEqual(self._sessions("T-01"), [(0.2, 12.5, 2)])
        self.assertEqual(self._sessions("T-02"), [(0.4, 12.5, 1)])

    def test_new_run_never_modifies_previous_run_rows(self):
        # Run 1 records a visit and ends.
        self.service.update("T-01", 2, 0.2)
        self.service.update("T-01", 3, 0.6)
        self.service.end_run(1.96)

        # Run 2 (replay) records its own rows; run 1's row stays untouched.
        self.service.begin_run()
        self.service.update("T-01", 1, 0.2)
        self.service.update("T-01", 2, 1.0)
        self.assertEqual(self._sessions(), [(0.2, 1.96, 3), (0.2, None, 2)])
        self.service.end_run(1.96)
        self.assertEqual(self._sessions(), [(0.2, 1.96, 3), (0.2, 1.96, 2)])

    def test_leftover_active_session_is_closed_not_extended(self):
        # A crashed run left a session open at video t=50.
        start_session("T-01", 50.0, 2)
        self.service.update("T-01", 1, 1.0)  # new run starts a session at t=1
        self.assertEqual(self._sessions(), [(50.0, 50.0, 2), (1.0, None, 1)])
        self.service.update("T-01", 5, 2.0)  # updates only the new run's row
        self.assertEqual(self._sessions(), [(50.0, 50.0, 2), (1.0, None, 5)])

    def test_begin_run_closes_leftover_active_sessions(self):
        # A crashed run left sessions open on both tables; beginning the new
        # run closes them at their own start times (never ended < started)
        # and preserves the history before any new session is recorded.
        start_session("T-01", 50.0, 2)
        start_session("T-02", 80.0, 1)
        self.service.begin_run()
        self.assertEqual(self._sessions("T-01"), [(50.0, 50.0, 2)])
        self.assertEqual(self._sessions("T-02"), [(80.0, 80.0, 1)])
        # The new run starts its own sessions.
        self.service.update("T-01", 1, 1.0)
        self.assertEqual(self._sessions(), [(50.0, 50.0, 2), (1.0, None, 1)])

    def test_no_session_is_stored_with_ended_before_started(self):
        start_session("T-01", 50.0, 2)
        self.service.end_run(2.0)  # last video ts is earlier than the leftover
        (started, ended, _), = self._sessions()
        self.assertGreaterEqual(ended, started)


if __name__ == "__main__":
    unittest.main(verbosity=2)
