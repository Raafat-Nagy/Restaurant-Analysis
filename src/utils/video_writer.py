"""Optional output-video writing via Supervision's VideoSink.

Owns the single output writer for the whole process: concurrent stream
generators never open a second sink or write interleaved frames, and a
finished save is never overwritten. The output directory is created only
when saving is enabled. The writer is opened for the exact frame size
being processed (trimmed by at most one pixel to even dimensions, which
video codecs require).
"""

import contextlib
import threading
from pathlib import Path

import supervision as sv

from config import settings

OUTPUT_VIDEO_PATH = Path(settings.OUTPUT_VIDEO_PATH)

_lock = threading.Lock()
_sink = None  # the single open VideoSink (owned by one stream generator)
_saved = False  # a full save already finished; never overwrite it
_DEFAULT_FPS = 25.0


def open_writer(frame, fps):
    """Open the single output writer; returns a writer handle or None.

    Returns None when saving is disabled, another generator already owns
    the writer, or a full save has already finished.
    """
    global _sink

    if not settings.SAVE_OUTPUT_VIDEO:
        return None

    with _lock:
        if _sink is not None or _saved:
            return None

        OUTPUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Video codecs need even frame dimensions; the writer matches the
        # exact frames written below (odd sizes are trimmed by one pixel).
        height = int(frame.shape[0]) // 2 * 2
        width = int(frame.shape[1]) // 2 * 2
        video_info = sv.VideoInfo(
            width=width,
            height=height,
            fps=fps if fps and fps > 0 else _DEFAULT_FPS,
        )

        stack = contextlib.ExitStack()
        sink = stack.enter_context(
            sv.VideoSink(target_path=str(OUTPUT_VIDEO_PATH), video_info=video_info)
        )
        _sink = sink
        return sink, stack


def write_frame(writer, frame):
    """Write one fully processed frame to the output video."""
    if writer is None:
        return

    sink = writer[0]
    sink.write_frame(frame[: sink.video_info.height, : sink.video_info.width])


def close_writer(writer, completed):
    """Release the writer safely; keep the file once a full save finished."""
    global _sink, _saved

    if writer is None:
        return

    _, stack = writer
    stack.close()

    with _lock:
        _sink = None
        if completed:
            _saved = True
