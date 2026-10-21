"""OpenCV dashboard panel for the restaurant analytics stream.

``RestaurantDashboard`` draws a dark-themed statistics panel and displays
per-table occupancy timers. It only receives statistics (per-table person
counts, seated-guest totals, and per-table occupancy elapsed video-seconds)
from the video-processing pipeline: YOLO detection, standing filtering,
occupancy counting and occupancy timing stay in ``video_processing``.

All font sizes and thicknesses are constructor parameters, so the look can
be tuned without touching the layout code. The layout measures every text
with ``cv2.getTextSize`` and stacks sections with a running cursor, so
custom font sizes do not overlap.
"""

import cv2
import numpy as np

# BGR colors (see hex values for the intended design colors)
_BG = (39, 24, 17)          # #111827 dark navy
_GREEN = (153, 211, 52)     # #34D399 free tables + progress bar
_RED = (113, 113, 248)      # #F87171 occupied tables
_GRAY = (219, 213, 209)     # #D1D5DB secondary text and labels
_PRIMARY = (251, 250, 249)  # near-white, important numbers
_TRACK = (81, 65, 55)       # #374151 progress track + separators
_BLUE = (250, 165, 96)      # #60A5FA seated-guests progress bar

_PAD = 15
_TITLE = "RESTAURANT ANALYSIS"


def _format_duration(seconds: float) -> str:
    """MM:SS below one hour, HH:MM:SS from one hour on."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


class RestaurantDashboard:
    """Draw the analytics panel and display per-table occupancy timers.

    ``tables`` carries the configured table IDs and capacities (from the
    regions configuration). A table is occupied while its person count is
    above zero - the same occupancy rule the region colors use. Each
    table's timer value is supplied by the processing pipeline in video
    time (elapsed seconds on the video's timeline), so slow processing
    cannot inflate it. The timer starts when the table becomes occupied,
    keeps running (and is not reset per frame) while it stays occupied and
    is hidden when the table becomes free; the next occupancy starts a new
    timer from zero.

    Parameters
    ----------
    tables:
        Configured tables as ``[{"id": "T-01", "max": 4}, ...]``.
    width:
        Panel width in pixels (default 200).
    title_scale, title_thickness:
        Project title ("RESTAURANT ANALYSIS") at the very top.
    heading_scale, heading_thickness:
        Section headings ("TABLE OCCUPANCY", "SEATED GUESTS", "TABLE STATUS").
    stat_scale, stat_thickness:
        Main statistics (the occupied/total count and the seated-guest count).
    table_id_scale, table_id_thickness:
        Table IDs ("T-01").
    table_count_scale, table_count_thickness:
        Per-table occupancy counts ("2/4") and the word "FREE".
    timer_scale, timer_thickness:
        Occupancy duration on occupied rows ("12:35").
    text_scale, text_thickness:
        General text (percentages and the capacity line).

    Example
    -------
    ``RestaurantDashboard(tables, width=200, stat_scale=0.8, title_thickness=2)``
    """

    def __init__(
        self,
        tables: list[dict],
        width: int = 200,
        *,
        title_scale: float = 0.46,
        title_thickness: int = 2,
        heading_scale: float = 0.28,
        heading_thickness: int = 1,
        stat_scale: float = 0.72,
        stat_thickness: int = 2,
        table_id_scale: float = 0.30,
        table_id_thickness: int = 1,
        table_count_scale: float = 0.32,
        table_count_thickness: int = 1,
        timer_scale: float = 0.28,
        timer_thickness: int = 1,
        text_scale: float = 0.32,
        text_thickness: int = 1,
    ):
        self.tables = list(tables)
        self.width = width

        self.title_scale = title_scale
        self.title_thickness = title_thickness
        self.heading_scale = heading_scale
        self.heading_thickness = heading_thickness
        self.stat_scale = stat_scale
        self.stat_thickness = stat_thickness
        self.table_id_scale = table_id_scale
        self.table_id_thickness = table_id_thickness
        self.table_count_scale = table_count_scale
        self.table_count_thickness = table_count_thickness
        self.timer_scale = timer_scale
        self.timer_thickness = timer_thickness
        self.text_scale = text_scale
        self.text_thickness = text_thickness

        self._occupancy_seconds: dict[str, float | None] = {
            table["id"]: None for table in self.tables
        }
        self._table_counts: dict[str, int] = {}
        self._seated_guests = 0

    def update(
        self,
        table_counts: dict[str, int],
        seated_guests: int,
        occupancy_seconds: dict[str, float | None],
    ) -> None:
        """Store this frame's statistics; timer values come from the pipeline."""
        self._table_counts = dict(table_counts)
        self._seated_guests = seated_guests
        self._occupancy_seconds = dict(occupancy_seconds)

    def render(self, height: int) -> np.ndarray:
        """Draw the dashboard panel with the given height and return it."""
        panel = np.full((height, self.width, 3), _BG, dtype=np.uint8)
        pad = _PAD

        # Project title at the very top (wraps to two lines if it does not fit)
        y = pad
        for line in self._title_lines():
            y = self._draw(panel, line, pad, y, self.title_scale, _PRIMARY, self.title_thickness)
        y += 8
        self._divider(panel, y)
        y += 16

        # Table occupancy and seated guests: matching statistics sections
        occupied = sum(
            1 for table in self.tables if self._table_counts.get(table["id"], 0) > 0
        )
        total = len(self.tables)
        capacity = sum(int(table.get("max", 0)) for table in self.tables)

        y = self._stat_section(panel, y, "TABLE OCCUPANCY", occupied, total, _GREEN)
        y = self._stat_section(panel, y, "SEATED GUESTS", self._seated_guests, capacity, _BLUE)

        # Table status: colored dot, ID, count/max or FREE, timer when occupied
        y = self._draw(panel, "TABLE STATUS", pad, y, self.heading_scale, _GRAY, self.heading_thickness)
        y += 6

        row_count = max(1, len(self.tables))
        row_h = max(12, min(24, (height - y - _PAD) // row_count))

        id_x = pad + 12
        ids_width = max(
            (
                self._measure(table["id"], self.table_id_scale, self.table_id_thickness)[0]
                for table in self.tables
            ),
            default=0,
        )
        count_x = id_x + ids_width + 10

        for i, table in enumerate(self.tables):
            top = y + i * row_h
            baseline = top + int(row_h * 0.72)
            tid = table["id"]
            count = self._table_counts.get(tid, 0)
            is_occupied = count > 0
            color = _RED if is_occupied else _GREEN

            cv2.circle(panel, (pad + 4, top + row_h // 2), 3, color, -1)
            self._text(panel, tid, id_x, baseline, self.table_id_scale, _PRIMARY, self.table_id_thickness)

            if is_occupied:
                self._text(
                    panel, f"{count}/{table['max']}", count_x, baseline,
                    self.table_count_scale, _RED, self.table_count_thickness,
                )
                self._right_text(
                    panel, self._duration_text(tid), baseline,
                    self.timer_scale, _RED, self.timer_thickness,
                )
            else:
                self._text(
                    panel, "FREE", count_x, baseline,
                    self.table_count_scale, _GREEN, self.table_count_thickness,
                )

        return panel

    def _stat_section(self, panel, y, heading, current, total, bar_color):
        """Draw one statistics section: value, heading + percentage, progress bar.

        The current value uses the main-stat font while ``/ total`` uses the
        smaller general font, so the current value stands out. Both the
        occupancy and the seated-guests sections share this layout.
        """
        pad = _PAD
        big = str(current)
        y = self._draw(panel, big, pad, y, self.stat_scale, _PRIMARY, self.stat_thickness)
        big_width = self._measure(big, self.stat_scale, self.stat_thickness)[0]
        self._text(
            panel, f" / {total}", pad + big_width + 4, y,
            self.text_scale, _GRAY, self.text_thickness,
        )
        y += 6

        percent = (current / total * 100) if total else 0.0
        percent_text = f"{percent:.0f}%"
        heading_h = self._measure(heading, self.heading_scale, self.heading_thickness)[1]
        percent_h = self._measure(percent_text, self.text_scale, self.text_thickness)[1]
        baseline = y + max(heading_h, percent_h)
        self._text(panel, heading, pad, baseline, self.heading_scale, _GRAY, self.heading_thickness)
        self._right_text(panel, percent_text, baseline, self.text_scale, _PRIMARY, self.text_thickness)
        y = baseline + 10

        bar_h = 7
        cv2.rectangle(panel, (pad, y), (self.width - pad, y + bar_h), _TRACK, -1)
        fill = int((self.width - 2 * pad) * min(percent, 100.0) / 100)
        if fill > 0:
            cv2.rectangle(panel, (pad, y), (pad + fill, y + bar_h), bar_color, -1)
        y += bar_h + 14
        self._divider(panel, y)
        y += 18
        return y

    def _duration_text(self, tid: str) -> str:
        elapsed = self._occupancy_seconds.get(tid)
        return _format_duration(elapsed if elapsed is not None else 0.0)

    def _title_lines(self) -> list[str]:
        max_width = self.width - 2 * _PAD
        if self._measure(_TITLE, self.title_scale, self.title_thickness)[0] <= max_width:
            return [_TITLE]
        first, _, second = _TITLE.partition(" ")
        return [first, second]

    def _measure(self, text: str, scale: float, thickness: int) -> tuple[int, int]:
        (width, height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        return width, height

    def _draw(self, panel, text, x, y, scale, color, thickness):
        """Draw text with its glyph box top at y; return y just below it."""
        _, height = self._measure(text, scale, thickness)
        self._text(panel, text, x, y + height, scale, color, thickness)
        return y + height

    def _text(self, panel, text, x, y, scale, color, thickness=1):
        cv2.putText(
            panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, thickness, cv2.LINE_AA,
        )

    def _right_text(self, panel, text, y, scale, color, thickness=1):
        width, _ = self._measure(text, scale, thickness)
        self._text(panel, text, self.width - _PAD - width, y, scale, color, thickness)

    def _divider(self, panel, y):
        cv2.line(panel, (_PAD, y), (self.width - _PAD, y), _TRACK, 1)
