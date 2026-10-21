# Restaurant Analysis

Analyze recorded restaurant video to estimate seated guests and table occupancy. YOLO26 Pose detects
people frame by frame; the pipeline filters standing people for counting, stabilizes counts per
table, updates occupancy sessions, annotates the video, and streams processed frames as MJPEG.

## Demo Video

Watch the full demonstration on YouTube:

[![Watch the video](https://img.youtube.com/vi/iWFZjKku5C4/hqdefault.jpg)](https://youtu.be/iWFZjKku5C4)

## Features

- **Standing-person filtering** — `StandingFilter` excludes standing people from occupancy counts.
  The original detections remain visible in the processed video.
- **Table-based guest counting** — `config/regions.json` defines each table's counting polygon and
  capacity. A `display_polygon` can optionally provide a separate outline for drawing.
- **Count stabilization** — `LoggedCount` smooths noisy frame-by-frame counts using recent readings
  and configurable increase, decrease, and empty delays.
- **Occupancy sessions and timers** — SQLite records visits, their maximum guest count, and
  video-timeline start/end values. The on-video dashboard displays table occupancy, seated guests,
  and elapsed visit time.
- **Processed video stream** — the page at `/` shows the header and video from `/stream`. Counts,
  table outlines, detection annotations, and the optional dashboard are included in the frames; the
  page has no separate statistics display.
- **Background processing** — one background thread processes the input video independently of
  browser connections. Opening or refreshing the page does not restart processing.
- **Optional output and resizing** — save an annotated copy with `SAVE_OUTPUT_VIDEO` (the input is
  unchanged), or downscale frames with `RESIZE_VIDEO` and `VIDEO_WIDTH` (no upscaling).

## Tech Stack

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Ultralytics YOLO26 Pose
- OpenCV, NumPy, and Supervision
- FastAPI and Uvicorn
- Pydantic Settings and SQLite

## Project Structure

```
Restaurant_Analysis/
├── .env.example
├── .gitignore
├── .python-version
├── Makefile
├── README.md
├── config/
│   └── regions.json
├── data/
│   └── restaurant.db
├── frontend/
│   ├── app.js
│   ├── favicon.svg
│   ├── index.html
│   └── style.css
├── models/
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dashboard.py
│   ├── main.py
│   ├── video_processing.py
│   ├── counting/
│   │   ├── __init__.py
│   │   ├── load_regions.py
│   │   └── logged_count.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   └── repository.py
│   ├── filters/
│   │   ├── __init__.py
│   │   └── standing_filter.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── occupancy_service.py
│   │   └── video_processor.py
│   └── utils/
│       ├── __init__.py
│       ├── video_utils.py
│       └── video_writer.py
├── tests/
│   ├── test_api.py
│   ├── test_logged_count.py
│   ├── test_occupancy_service.py
│   ├── test_occupancy_sessions.py
│   ├── test_regions.py
│   ├── test_standing_filter.py
│   └── test_video_processing.py
├── uv.lock
└── videos/
    ├── output/
    └── restaurant-demo.mp4
```

## Installation

Install [uv](https://docs.astral.sh/uv/) and Python 3.12. On Linux, `opencv-python` also requires
the OpenGL library (`libgl1` on Debian/Ubuntu).

```bash
git clone https://github.com/RFN-AI/Restaurant_Analysis.git
cd Restaurant_Analysis
uv sync
make download-video
cp .env.example .env
```

Place YOLO26 pose weights in `models/`. The default `MODEL_PATH` is `models/yolo26s-pose.pt`;
provide that file or set `MODEL_PATH` in `.env` to the weights you have.

## Configuration

Settings are listed in `.env.example`; values can also be supplied through environment variables.
Run commands from the project root so relative video and model paths resolve as expected.

- **Input files:** `VIDEO_PATH` (default `videos/restaurant-demo.mp4`) and `MODEL_PATH` (default
  `models/yolo26s-pose.pt`).
- **Inference:** `CONF`, `IOU`, and `IMGSZ`.
- **Count stabilization:** `COUNT_INCREASE_DELAY`, `COUNT_DECREASE_DELAY`, `COUNT_EMPTY_DELAY`, and
  `COUNT_HISTORY_SIZE`.
- **Output video:** `SAVE_OUTPUT_VIDEO` (off by default) and `OUTPUT_VIDEO_PATH`.
- **Resizing:** `RESIZE_VIDEO` (off by default) and `VIDEO_WIDTH`.
- **On-video dashboard:** `SHOW_DASHBOARD` (on by default) and `DASHBOARD_WIDTH`.

Edit the `zones` list in `config/regions.json` to set table names, capacities, and polygons. Each
zone requires `counting_polygon`; `display_polygon` is optional and, when present, is used to draw
the table outline.

## Usage

```bash
uv run python src/main.py
```

Alternatively, run `make run`. Open [http://localhost:5000/](http://localhost:5000/) to view the
page; the processed MJPEG stream is available at `/stream`. Processing starts with the application
and replays the input from the beginning on the next application start. Occupancy timers use video
time, not processing or browser time.

## Testing

```bash
uv run python -m unittest discover -s tests -v
```

The `unittest` suite covers count stabilization, standing-person filtering, region validation,
occupancy-session behavior, web routes, and background streaming. It uses temporary SQLite databases
and synthetic video; no YOLO model or GPU is required.

## Database

SQLite stores configured table capacities in `tables` and visits in `occupancy_sessions`
(`started_at`, `ended_at`, and `max_people`). Each processing run records separate session rows;
history is preserved, only one session per table may be active, and sessions left by an interrupted
run are closed when the next run starts. Browser connections do not create sessions.

## Acknowledgments

Inspired by the [Restaurant Analysis using Computer Vision – Full Course](https://youtu.be/KqL5ymW3foI), which introduced key concepts in person detection, table-based counting, and occupancy monitoring. This project extends these concepts with count stabilization, standing-person filtering, and SQLite-based occupancy tracking.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
