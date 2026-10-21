"""Standing-person filtering for person detections.

``StandingFilter`` removes standing people from a ``sv.Detections`` set using
the thigh-to-torso ratio computed from YOLO-Pose (COCO-17) keypoints, so that
only seated (or unclassifiable) people remain for table-occupancy counting.
The caller keeps drawing the original, unfiltered detections.
"""

import numpy as np
import supervision as sv

# COCO-17 keypoint indices as (shoulder, hip, knee) per body side.
_SIDES = ((5, 11, 13), (6, 12, 14))


class StandingFilter:
    """Filter out standing people using the thigh-to-torso ratio.

    ratio = abs(hip_y - knee_y) / abs(shoulder_y - hip_y)

    A standing person's thigh is roughly vertical, so the hip-to-knee vertical
    extent is comparable to the torso's (ratio around 0.8-1.2). A seated
    person's thigh is roughly horizontal, so hip and knee are almost level
    (ratio near 0). Ratios below ``max_seated_ratio`` count as seated (kept);
    people at or above it are treated as standing and removed.

    Parameters
    ----------
    max_seated_ratio:
        Ratio below which a person counts as seated (default 0.4), midway
        between sitting (~0-0.3) and standing (~0.8-1.2). Lower it if standing
        people are miscounted as seated; raise it if seated people are dropped.
    min_keypoint_conf:
        Minimum keypoint confidence for a joint to be used (default 0.3).
        Low-confidence joints are treated as missing so noisy estimates cannot
        swing the classification.
    keep_unclassified:
        What happens when no body side can be measured (e.g. heavy occlusion).
        True (default) keeps the person in the filtered result so occluded
        seated people do not disappear from the count; False drops them so only
        positively-classified seated people remain.
    """

    def __init__(
        self,
        max_seated_ratio: float = 0.4,
        min_keypoint_conf: float = 0.3,
        keep_unclassified: bool = True,
    ):
        self.max_seated_ratio = max_seated_ratio
        self.min_keypoint_conf = min_keypoint_conf
        self.keep_unclassified = keep_unclassified

    def filter(
        self,
        detections: sv.Detections,
        keypoints_xy: np.ndarray | None,
        keypoint_conf: np.ndarray | None = None,
    ) -> sv.Detections:
        """Return only the detections classified as seated.

        ``keypoints_xy`` is (N, K, 2) and ``keypoint_conf`` is (N, K), both in
        the same detection order as ``detections`` (rows are kept aligned).
        Either may be None when pose data is unavailable; every person is then
        treated as unclassified.
        """
        count = len(detections)
        keypoints_xy = self._as_numpy(keypoints_xy)
        keypoint_conf = self._as_numpy(keypoint_conf)

        if keypoints_xy is None:
            keep = np.full(count, self.keep_unclassified, dtype=bool)
            return detections[keep]

        if len(keypoints_xy) != count:
            raise ValueError(
                f"keypoints ({len(keypoints_xy)}) must match detections ({count})"
            )
        if keypoint_conf is not None and len(keypoint_conf) != count:
            raise ValueError(
                f"keypoint confidences ({len(keypoint_conf)}) must match "
                f"detections ({count})"
            )

        keep = np.empty(count, dtype=bool)
        for i in range(count):
            seated = self._is_seated(
                keypoints_xy[i],
                None if keypoint_conf is None else keypoint_conf[i],
            )
            keep[i] = self.keep_unclassified if seated is None else seated

        return detections[keep]

    def _is_seated(
        self,
        person_xy: np.ndarray,
        person_conf: np.ndarray | None,
    ) -> bool | None:
        """Classify one person: True seated, False standing, None unknown."""
        ratios = self._side_ratios(person_xy, person_conf)

        if not ratios:
            return None

        return bool(np.mean(ratios) < self.max_seated_ratio)

    def _side_ratios(
        self,
        person_xy: np.ndarray,
        person_conf: np.ndarray | None,
    ) -> list[float]:
        """Thigh-to-torso ratios for every measurable body side."""
        ratios = []

        for shoulder, hip, knee in _SIDES:
            points = []
            for index in (shoulder, hip, knee):
                point = person_xy[index]
                if person_conf is not None and person_conf[index] < self.min_keypoint_conf:
                    break
                if np.allclose(point, 0):  # missing keypoint placeholder
                    break
                points.append(point)
            else:
                s, h, k = points
                torso = abs(float(s[1] - h[1]))
                if torso < 1e-6:  # division by zero: torso not measurable
                    continue
                ratios.append(abs(float(h[1] - k[1])) / torso)

        return ratios

    @staticmethod
    def _as_numpy(value):
        if value is None:
            return None
        if hasattr(value, "cpu"):  # torch.Tensor
            value = value.cpu().numpy()
        return np.asarray(value)
