"""Focused tests for the background processing/streaming (video_processing.py).

Uses a lightweight processor stand-in and a tiny synthetic video, so no YOLO
model, GPU, or real footage is required.
"""

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import video_processing as vp


class FakeProcessor:
    """Stand-in for VideoProcessor: records calls, no model needed."""

    def __init__(self, delay=0.0):
        self.tables = [{"id": "T-01", "max": 4}]
        self.delay = delay
        self.calls = []
        self.runs = 0
        self.ends = []

    def begin_run(self):
        self.runs += 1

    def process_frame(self, frame, video_time):
        if self.delay:
            time.sleep(self.delay)
        self.calls.append(video_time)
        return frame, {
            "table_counts": {"T-01": 0},
            "seated_guests": 0,
            "occupancy_seconds": {"T-01": None},
        }

    def end_run(self, last_video_time):
        self.ends.append(last_video_time)


def make_video(path, frames=5, fps=25.0):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240)
    )
    assert writer.isOpened()
    for i in range(frames):
        frame = np.full((240, 320, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class BackgroundProcessingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        video_path = Path(self._tmp.name) / "clip.mp4"
        make_video(video_path)

        self._old_video_path = vp.VIDEO_PATH
        self._old_dashboard = vp.settings.SHOW_DASHBOARD
        vp.VIDEO_PATH = str(video_path)
        vp.settings.SHOW_DASHBOARD = False

    def tearDown(self):
        vp.stop_processing()
        vp.VIDEO_PATH = self._old_video_path
        vp.settings.SHOW_DASHBOARD = self._old_dashboard
        self._tmp.cleanup()

    def _join_worker(self):
        thread = vp._processing_thread
        if thread is not None:
            thread.join(timeout=15)
        deadline = time.monotonic() + 15
        while not vp._processing_done and time.monotonic() < deadline:
            time.sleep(0.02)

    def test_only_one_processing_thread_runs(self):
        fake = FakeProcessor(delay=0.05)
        vp.start_processing(fake)
        vp.start_processing(fake)  # must be a no-op while running
        self._join_worker()
        self.assertEqual(fake.runs, 1)          # one processing run ...
        self.assertEqual(len(fake.calls), 5)    # ... processed the video once

    def test_browser_refresh_does_not_restart_processing(self):
        fake = FakeProcessor(delay=0.05)
        vp.start_processing(fake)

        first = vp.stream_frames()
        next(first)
        next(first)
        first.close()  # browser disconnects

        second = vp.stream_frames()
        next(second)  # reconnect: resumes at the latest frame
        next(second)
        second.close()

        self._join_worker()
        self.assertEqual(fake.runs, 1, "refresh must not start a new run")
        self.assertEqual(len(fake.calls), 5, "video must be processed exactly once")
        self.assertGreaterEqual(fake.calls[0], 0.0)

    def test_processing_continues_without_any_browser(self):
        fake = FakeProcessor(delay=0.02)
        vp.start_processing(fake)
        self._join_worker()  # no client ever connected
        self.assertEqual(len(fake.calls), 5)

    def test_end_of_processing_reports_last_video_time(self):
        fake = FakeProcessor()
        vp.start_processing(fake)
        for _ in vp.stream_frames():  # drain until the stream ends
            pass
        self._join_worker()
        # 5 frames at 25 fps -> last processed video timestamp is 4/25 = 0.16
        self.assertEqual(fake.ends, [0.16])

    def test_failed_initialization_still_ends_the_run(self):
        class FailingProcessor(FakeProcessor):
            def begin_run(self):
                self.runs += 1
                raise RuntimeError("model unavailable")

        fake = FailingProcessor()
        vp.start_processing(fake)
        self._join_worker()  # the worker must finish, not hang

        self.assertEqual(fake.runs, 1)
        self.assertEqual(fake.calls, [])          # no frame was processed
        self.assertEqual(len(fake.ends), 1)       # cleanup still ran
        self.assertTrue(vp._processing_done)      # stream can end cleanly
        self.assertEqual(list(vp.stream_frames()), [])  # no frames to stream

    def test_stop_processing_stops_thread_and_closes_sessions(self):
        fake = FakeProcessor(delay=0.05)
        vp.start_processing(fake)
        next(vp.stream_frames())
        vp.stop_processing()
        self.assertIsNone(vp._processing_thread)  # cleared after the join
        self.assertEqual(len(fake.ends), 1)  # end_run called on shutdown too
        self.assertGreaterEqual(fake.ends[0], 0.0)

    def test_stop_processing_keeps_reference_until_thread_stops(self):
        release = threading.Event()
        entered = threading.Event()

        class BlockingProcessor(FakeProcessor):
            def process_frame(self, frame, video_time):
                entered.set()
                release.wait(timeout=15)  # hold the worker inside a frame
                return super().process_frame(frame, video_time)

        fake = BlockingProcessor()
        vp.start_processing(fake)
        self.assertTrue(entered.wait(timeout=15))
        thread = vp._processing_thread

        stopper = threading.Thread(target=vp.stop_processing)
        stopper.start()
        time.sleep(0.3)  # stop_processing is waiting inside thread.join()

        # The thread has not stopped yet, so its reference must still be
        # visible -- clearing it here would let a second thread start.
        self.assertIs(vp._processing_thread, thread)
        self.assertTrue(thread.is_alive())

        # A start requested meanwhile must not run a second processing
        # thread alongside the still-running one.
        starter = threading.Thread(target=vp.start_processing, args=(FakeProcessor(),))
        starter.start()
        time.sleep(0.3)
        alive = [
            t
            for t in threading.enumerate()
            if t.name == "video-processing" and t.is_alive()
        ]
        self.assertEqual(len(alive), 1)
        self.assertIs(alive[0], thread)

        release.set()
        stopper.join(timeout=15)
        starter.join(timeout=15)
        self.assertFalse(thread.is_alive())  # cleared only after it stopped


if __name__ == "__main__":
    unittest.main(verbosity=2)
