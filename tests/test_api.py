"""Focused tests for the FastAPI app (main.py).

The web frontend shows a header and the video stream only, so the app exposes
the page, its static assets and ``/stream``; there is no statistics endpoint.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from main import FRONTEND_DIR, app


class AppRoutesTests(unittest.TestCase):
    def setUp(self):
        self.paths = {route.path for route in app.routes if hasattr(route, "path")}

    def test_page_and_stream_routes_exist(self):
        self.assertIn("/", self.paths)
        self.assertIn("/stream", self.paths)

    def test_no_statistics_endpoint(self):
        self.assertNotIn("/api/stats", self.paths)

    def test_frontend_files_served_by_the_page(self):
        index = FRONTEND_DIR / "index.html"
        self.assertTrue(index.is_file())

        html = index.read_text()
        self.assertIn("/static/style.css", html)
        self.assertIn('rel="icon" type="image/svg+xml" href="/static/favicon.svg"', html)
        self.assertIn('src="/static/favicon.svg"', html)
        self.assertIn("/static/app.js", html)
        self.assertTrue((FRONTEND_DIR / "favicon.svg").is_file())
        self.assertTrue((FRONTEND_DIR / "style.css").is_file())
        self.assertTrue((FRONTEND_DIR / "app.js").is_file())

    def test_page_has_no_statistics_ui(self):
        html = (FRONTEND_DIR / "index.html").read_text()

        self.assertIn("Restaurant Analysis", html)
        self.assertIn("Seated-guest counting &amp; table occupancy", html)
        for removed in ("Seated guests", "Occupied tables", "Available tables"):
            self.assertNotIn(removed, html)
        self.assertNotIn("/api/stats", html)

    def test_frontend_script_requests_only_the_stream(self):
        script = (FRONTEND_DIR / "app.js").read_text()

        self.assertIn("/stream", script)
        self.assertNotIn("/api/stats", script)
        self.assertNotIn("fetch(", script)
        self.assertIn("Processed video unavailable — retrying…", script)


if __name__ == "__main__":
    unittest.main(verbosity=2)
