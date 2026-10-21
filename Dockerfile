# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# System libraries required by opencv-python (no GUI is used at runtime).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install locked dependencies first for layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Application code, the default region configuration, and the web frontend
# (index.html + static assets are served by src/main.py at runtime).
COPY src ./src
COPY config ./config
COPY frontend ./frontend

# Runtime mounts: models/, videos/ (input + output), and data/ (SQLite).
RUN mkdir -p data videos/output models
VOLUME ["/app/data", "/app/models", "/app/videos", "/app/config"]

EXPOSE 5000

# Runtime configuration comes from environment variables; .env is never
# copied into the image. Override these at `docker run` as needed.
ENV VIDEO_PATH=videos/restaurant-demo.mp4 \
    MODEL_PATH=models/yolo26s-pose.pt

CMD ["uv", "run", "--no-sync", "python", "src/main.py"]
