"""Focused tests for StandingFilter (filters/standing_filter.py)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from filters.standing_filter import StandingFilter


def make_detections(count=1):
    boxes = np.array([[0, 0, 50, 50]] * count, dtype=np.float32)
    return sv.Detections(xyxy=boxes, class_id=np.zeros(count, dtype=int))


def make_keypoints(shoulder_y, hip_y, knee_y, conf=0.9, count=1):
    """One person per row; left side (5/11/13) gets the given y positions."""
    xy = np.full((count, 17, 2), 100.0, dtype=np.float32)
    xy[:, 5] = (100.0, shoulder_y)
    xy[:, 11] = (100.0, hip_y)
    xy[:, 13] = (100.0, knee_y)
    xy[:, 6] = (110.0, shoulder_y)
    xy[:, 12] = (110.0, hip_y)
    xy[:, 14] = (110.0, knee_y)
    scores = np.full((count, 17), conf, dtype=np.float32)
    return xy, scores


class StandingFilterTests(unittest.TestCase):
    def test_seated_person_is_kept_for_counting(self):
        # torso = 30px, thigh = 5px -> ratio 0.167 < 0.4 -> seated
        xy, scores = make_keypoints(100.0, 130.0, 135.0)
        kept = StandingFilter().filter(make_detections(), xy, scores)
        self.assertEqual(len(kept), 1)

    def test_standing_person_is_removed_from_counting(self):
        # torso = 30px, thigh = 50px -> ratio 1.67 >= 0.4 -> standing
        xy, scores = make_keypoints(100.0, 130.0, 180.0)
        kept = StandingFilter().filter(make_detections(), xy, scores)
        self.assertEqual(len(kept), 0)

    def test_missing_pose_data_is_unclassified(self):
        kept_default = StandingFilter().filter(make_detections(), None, None)
        self.assertEqual(len(kept_default), 1)  # keep_unclassified=True

        kept_strict = StandingFilter(keep_unclassified=False).filter(
            make_detections(), None, None
        )
        self.assertEqual(len(kept_strict), 0)

    def test_placeholder_keypoints_are_treated_as_missing(self):
        xy = np.zeros((1, 17, 2), dtype=np.float32)  # (0,0) placeholders
        scores = np.full((1, 17), 0.9, dtype=np.float32)
        kept = StandingFilter().filter(make_detections(), xy, scores)
        self.assertEqual(len(kept), 1)  # unclassified -> kept by default

    def test_low_confidence_joints_are_ignored(self):
        xy, scores = make_keypoints(100.0, 130.0, 180.0)  # standing pose
        scores[0, (11, 12, 13, 14)] = 0.1  # hips/knees below min_keypoint_conf
        kept = StandingFilter().filter(make_detections(), xy, scores)
        self.assertEqual(len(kept), 1)  # no measurable side -> unclassified

    def test_mismatched_keypoint_count_raises(self):
        xy, scores = make_keypoints(100.0, 130.0, 135.0, count=2)
        with self.assertRaises(ValueError):
            StandingFilter().filter(make_detections(count=1), xy, scores)


if __name__ == "__main__":
    unittest.main(verbosity=2)
