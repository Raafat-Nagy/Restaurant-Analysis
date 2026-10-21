"""Per-frame video processing: detection, filtering, counting, drawing."""

import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import REGIONS_CONFIG_PATH, settings
from counting.load_regions import load_zones
from counting.logged_count import LoggedCount
from filters.standing_filter import StandingFilter
from services.occupancy_service import OccupancyService


def _make_zone(polygon_points, triggering_anchors=(sv.Position.CENTER,)):
    return sv.PolygonZone(
        polygon=np.array(polygon_points, dtype=np.int32),
        triggering_anchors=triggering_anchors,
    )


class VideoProcessor:
    """Process individual frames of the restaurant video.

    Owns the YOLO model and the frame-processing dependencies (regions,
    annotators, ``StandingFilter``, one ``LoggedCount`` per table) and
    delegates occupancy-session updates to ``OccupancyService``. Standing
    people are removed from counting only; the original detections are
    always drawn.
    """

    def __init__(self):
        self.model = YOLO(settings.MODEL_PATH)
        self.conf = settings.CONF
        self.iou = settings.IOU
        self.imgsz = settings.IMGSZ

        zones_config = load_zones(REGIONS_CONFIG_PATH)
        self.tables = [
            {"id": config["name"], "max": config["max"]} for config in zones_config
        ]

        self.zones = [
            {
                "name": config["name"],
                "max": config["max"],
                "counting_zone": _make_zone(config["counting_polygon"]),
                "annotator": (
                    sv.PolygonZoneAnnotator(
                        zone=_make_zone(config["display_polygon"]),
                        color=sv.Color.WHITE,
                        thickness=2,
                    )
                    if "display_polygon" in config
                    else None
                ),
            }
            for config in zones_config
        ]

        self.blur_annotator = sv.BlurAnnotator(kernel_size=12)
        self.corner_annotator = sv.BoxCornerAnnotator(color=sv.Color.WHITE)
        self.standing_filter = StandingFilter()
        self.count_stabilizers = {
            config["name"]: LoggedCount(
                increase_seconds=settings.COUNT_INCREASE_DELAY,
                decrease_seconds=settings.COUNT_DECREASE_DELAY,
                history_size=settings.COUNT_HISTORY_SIZE,
                empty_seconds=settings.COUNT_EMPTY_DELAY,
            )
            for config in zones_config
        }
        self.occupancy = OccupancyService([config["name"] for config in zones_config])

    def begin_run(self) -> None:
        """Called once when a processing run starts."""
        self.occupancy.begin_run()

    def end_run(self, last_video_time: float) -> None:
        """Called once when a processing run ends (see ``OccupancyService``)."""
        self.occupancy.end_run(last_video_time)

    def process_frame(self, frame, video_time: float):
        """Process one frame; return the annotated frame and dashboard stats."""
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        keypoints = results.keypoints
        filtered_detections = self.standing_filter.filter(
            detections,
            keypoints.xy if keypoints is not None else None,
            keypoints.conf if keypoints is not None else None,
        )

        # Unfiltered detections are drawn; only counting uses the filtered set.
        annotated_frame = self.blur_annotator.annotate(
            scene=frame,
            detections=detections,
        )
        annotated_frame = self.corner_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
        )

        stats = {
            "table_counts": {},
            "seated_guests": 0,
            "occupancy_seconds": {},
        }

        for zone in self.zones:
            zone["counting_zone"].trigger(filtered_detections)

            raw_count = zone["counting_zone"].current_count
            count = self.count_stabilizers[zone["name"]].update(raw_count, video_time)
            maximum = zone["max"]
            stats["table_counts"][zone["name"]] = count

            stats["occupancy_seconds"][zone["name"]] = self.occupancy.update(
                zone["name"], count, video_time
            )

            if count == 0:
                color = sv.Color.GREEN
            elif count < maximum:
                color = sv.Color.YELLOW
            else:
                color = sv.Color.RED

            if zone["annotator"] is not None:
                zone["annotator"].color = color
                annotated_frame = zone["annotator"].annotate(
                    scene=annotated_frame,
                    label=f"{zone['name']}: {count}/{maximum}",
                )

        # SEATED GUESTS is the sum of the stabilized table counts, so the total
        # always matches the per-table numbers, colors, and occupancy.
        stats["seated_guests"] = sum(stats["table_counts"].values())

        return annotated_frame, stats
