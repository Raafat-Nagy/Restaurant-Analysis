"""Focused tests for LoggedCount stabilization (counting/logged_count.py)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from counting.logged_count import LoggedCount


def drive(count, raw_values, start, step):
    """Feed readings at start, start+step, ... and return the last output."""
    output = None
    for i, raw in enumerate(raw_values):
        output = count.update(raw, start + i * step)
    return output


class LoggedCountTests(unittest.TestCase):
    def test_increase_confirms_after_configured_delay(self):
        count = LoggedCount()  # increase 2.0s
        # t=0.00 .. 1.96: pending, not yet confirmed
        for i in range(50):
            self.assertEqual(count.update(1, i * 0.04), 0)
        self.assertEqual(count.update(1, 2.0), 1)  # exactly +2.0s

    def test_empty_resets_after_continuous_zeros_only(self):
        count = LoggedCount()  # empty 10.0s
        drive(count, [1] * 51, 0.0, 0.04)  # stable 1 from t=2.0
        self.assertEqual(count.stable, 1)

        # A few zero frames never reset an active visit.
        for t in (2.04, 2.08, 2.12):
            self.assertEqual(count.update(0, t), 1)
        self.assertEqual(count.update(1, 2.16), 1)

        # Continuous zeros confirm at exactly +10s after the first zero (2.24).
        t = 2.24
        while t < 12.24 - 1e-9:
            self.assertEqual(count.update(0, t), 1)
            t = round(t + 0.04, 10)
        self.assertEqual(count.update(0, 12.24), 0)

    def test_decrease_confirms_after_configured_delay(self):
        count = LoggedCount()  # decrease 10.0s
        self.assertEqual(drive(count, [2] * 51, 0.0, 0.04), 2)  # stable 2 from 2.0
        # raw=1 becomes the candidate at t=2.04 (history was cleared on confirm)
        t = 2.04
        while t < 12.04 - 1e-9:
            self.assertEqual(count.update(1, t), 2)
            t = round(t + 0.04, 10)
        self.assertEqual(count.update(1, 12.04), 1)

    def test_custom_delays_are_honored(self):
        count = LoggedCount(increase_seconds=0.5, decrease_seconds=1.0,
                            history_size=10, empty_seconds=1.0)
        self.assertEqual(count.update(1, 0.0), 0)
        self.assertEqual(count.update(1, 0.25), 0)
        self.assertEqual(count.update(1, 0.5), 1)

    def test_majority_tie_prefers_the_stable_count(self):
        count = LoggedCount()
        drive(count, [1] * 51, 0.0, 0.04)  # stable 1
        # Alternating readings tie 5-5 in the window: no change is proposed.
        for i, t in enumerate([2.04 + 0.04 * i for i in range(10)]):
            raw = 2 if i % 2 == 0 else 1
            self.assertEqual(count.update(raw, t), 1)
        # A real majority still confirms after the full delay.
        self.assertEqual(drive(count, [2] * 60, 2.44, 0.04), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
