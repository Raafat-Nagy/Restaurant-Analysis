"""Background video execution and MJPEG streaming.

This module runs the video pipeline in one background thread and streams the
latest processed frames to browsers. It owns thread lifecycle, the shared
frame mailbox, and synchronization only: per-frame detection, filtering, and
counting live in ``services.video_processor.VideoProcessor``, and occupancy
sessions are managed by ``services.occupancy_service.OccupancyService``.
Browsers only read the mailbox -- connecting, refreshing, or disconnecting
never starts a processing run and never touches processing state.
"""

import threading

import cv2
import numpy as np
import supervision as sv

from config import settings
from dashboard import RestaurantDashboard
from services.video_processor import VideoProcessor
from utils.video_utils import encode_frame, resize_frame
from utils.video_writer import close_writer, open_writer, write_frame

VIDEO_PATH = settings.VIDEO_PATH

# --- shared frame state ---
#
# One background thread processes the video exactly once and publishes each
# encoded frame into this thread-safe mailbox; MJPEG clients only read it.

_frame_condition = threading.Condition()
_latest_chunk = None
_frame_seq = 0
_processing_done = False

_processing_thread = None
_thread_lock = threading.Lock()
_stop_event = threading.Event()

_processor = None  # shared VideoProcessor (tests may pass a stand-in)


def _get_processor() -> VideoProcessor:
    global _processor
    if _processor is None:
        _processor = VideoProcessor()
    return _processor


def _publish_chunk(chunk):
    """Store the newest encoded frame and wake every waiting streamer."""
    global _latest_chunk, _frame_seq
    with _frame_condition:
        _latest_chunk = chunk
        _frame_seq += 1
        _frame_condition.notify_all()


def _set_processing_done():
    global _processing_done
    with _frame_condition:
        _processing_done = True
        _frame_condition.notify_all()


def _process_video(processor):
    """Run the processing pipeline once and publish frames for the streamers."""
    cap = None
    writer = None
    writer_tried = False
    completed = False

    frame_index = 0
    use_pts = True
    last_pts_ms = None
    video_time = 0.0

    try:
        # Failures while initializing the run (drawing panel, session setup,
        # video file) still have to run the cleanup below.
        processor.begin_run()

        dashboard = RestaurantDashboard(
            tables=processor.tables,
            width=settings.DASHBOARD_WIDTH,
            title_scale=0.48,
            title_thickness=1,
            heading_scale=0.44,
            heading_thickness=1,
            stat_scale=0.72,
            stat_thickness=1,
            table_id_scale=0.45,
            table_id_thickness=1,
            table_count_scale=0.37,
            table_count_thickness=1,
            timer_scale=0.4,
            timer_thickness=1,
            text_scale=0.5,
            text_thickness=1,
        )

        cap = cv2.VideoCapture(VIDEO_PATH)

        if not cap.isOpened():
            raise RuntimeError(f"Error reading video file: {VIDEO_PATH}")

        fps = sv.VideoInfo.from_video_path(VIDEO_PATH).fps
        # Frame-index timeline fallback when presentation timestamps are unusable.
        timeline_fps = fps if fps and np.isfinite(fps) and fps > 0 else None

        while not _stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                break

            # Video timeline in seconds: the presentation timestamp when
            # reliable (finite, non-negative, strictly increasing),
            # otherwise frame index / the video's actual FPS. Never
            # wall-clock processing time, and never allowed to move
            # backwards. If both sources are unusable, the timeline holds,
            # so a timer can never increase incorrectly.
            if use_pts:
                pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if (
                    np.isfinite(pts_ms)
                    and pts_ms >= 0
                    and (last_pts_ms is None or pts_ms > last_pts_ms)
                ):
                    last_pts_ms = pts_ms
                    video_time = max(video_time, pts_ms / 1000.0)
                else:
                    use_pts = False
            if not use_pts and timeline_fps is not None:
                video_time = max(video_time, frame_index / timeline_fps)

            frame, stats = processor.process_frame(frame, video_time)

            # Resize the video frame only, before the dashboard is added,
            # so the dashboard width stays independent of VIDEO_WIDTH.
            if settings.RESIZE_VIDEO:
                frame = resize_frame(frame, width=settings.VIDEO_WIDTH)

            # Optional dashboard panel. Counting, stabilization, timers and
            # the occupancy database keep running either way; only the panel
            # drawing is skipped.
            if settings.SHOW_DASHBOARD:
                dashboard.update(
                    table_counts=stats["table_counts"],
                    seated_guests=stats["seated_guests"],
                    occupancy_seconds=stats["occupancy_seconds"],
                )
                panel = dashboard.render(height=frame.shape[0])
                frame = np.hstack([panel, frame])

            if not writer_tried:
                writer = open_writer(frame, fps)
                writer_tried = True

            write_frame(writer, frame)

            frame_data = encode_frame(frame)

            if frame_data is not None:
                _publish_chunk(frame_data)

            frame_index += 1
        completed = not _stop_event.is_set()
    finally:
        if cap is not None:
            cap.release()
        close_writer(writer, completed)
        # Processing has ended: close sessions that are still active (people
        # may still be seated) at the last processed video timestamp. A
        # failure there must not leave the stream waiting forever, so the
        # completion flag is always set.
        try:
            processor.end_run(video_time)
        finally:
            _set_processing_done()


def stream_frames():
    """Yield MJPEG chunks with the latest frames from the background thread.

    A client that connects mid-run immediately receives the latest available
    frame and then every new one. Clients never trigger processing, and a
    disconnect only ends this generator. The stream ends after the last frame
    once processing has finished.
    """
    last_seq = 0
    while True:
        with _frame_condition:
            while (
                _frame_seq == last_seq
                and not _processing_done
                and not _stop_event.is_set()
            ):
                _frame_condition.wait(timeout=1.0)
            if _frame_seq == last_seq:
                # Processing finished and there is no newer frame.
                return
            last_seq = _frame_seq
            chunk = _latest_chunk
        yield chunk


def start_processing(processor=None):
    """Start the single background processing thread (idempotent).

    Calling this again while processing is running is a no-op, so only one
    thread can ever process the video. ``processor`` defaults to a shared
    ``VideoProcessor``; tests may pass a lightweight stand-in.
    """
    global _processing_thread, _latest_chunk, _frame_seq, _processing_done

    with _thread_lock:
        if _processing_thread is not None and _processing_thread.is_alive():
            return

        if processor is None:
            processor = _get_processor()

        with _frame_condition:
            _latest_chunk = None
            _frame_seq = 0
            _processing_done = False

        _stop_event.clear()
        thread = threading.Thread(
            target=_process_video,
            args=(processor,),
            name="video-processing",
            daemon=True,
        )
        _processing_thread = thread
        thread.start()


def stop_processing():
    """Stop the background thread and release the video resources.

    The thread reference is cleared only after the thread has actually
    stopped. While it is still running the reference stays visible, so
    ``start_processing`` can never spawn a second processing thread
    alongside it.
    """
    global _processing_thread

    _stop_event.set()
    with _frame_condition:
        _frame_condition.notify_all()

    with _thread_lock:
        thread = _processing_thread
        if thread is not None:
            thread.join(timeout=10.0)
            if not thread.is_alive():
                _processing_thread = None
