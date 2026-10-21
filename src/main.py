"""FastAPI app: the web page and the MJPEG video stream."""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import PROJECT_ROOT, REGIONS_CONFIG_PATH
from counting.load_regions import load_zones
from database.connection import init_database
from database.repository import sync_tables
from video_processing import start_processing, stop_processing, stream_frames

FRONTEND_DIR = PROJECT_ROOT / "frontend"

logger = logging.getLogger(__name__)


def _prepare_database():
    """Create the schema and synchronize the tables before processing starts."""
    init_database()
    sync_tables([(zone["name"], zone["max"]) for zone in load_zones(REGIONS_CONFIG_PATH)])


class ConnectionCloseMiddleware:
    """Advertise single-use connections: every response ends its connection.

    ``uvicorn.run`` below sets ``timeout_keep_alive=0``, so the server closes
    each connection as soon as its response completes. Without this header a
    client or proxy assumes HTTP keep-alive and may write its next request
    into a connection that is already closing; that request is lost or
    answered with uvicorn's protocol-error 400. The page, its static assets
    and the repeated ``/stream`` connections all go through this path.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_close(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"connection", b"close"))
                message = dict(message, headers=headers)
            await send(message)

        await self.app(scope, receive, send_with_close)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One background processing thread for the whole server lifetime:
    # browsers only read the frames it publishes, they never start it.
    _prepare_database()
    try:
        start_processing()
    except Exception:
        # A missing model or unreadable video must not abort start-up: the
        # page still loads and reports that the stream is unavailable.
        logger.exception("Video processing could not start")
        stop_processing()
    try:
        yield
    finally:
        stop_processing()


app = FastAPI(lifespan=lifespan)

# Single-use connections: clients are told the connection ends with each
# response, so no request can be written into a closing connection.
app.add_middleware(ConnectionCloseMiddleware)

# Static assets for the dashboard (HTML/CSS/JS in frontend/).
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/stream")
def stream():
    return StreamingResponse(
        stream_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, timeout_keep_alive=0)
