.DEFAULT_GOAL := help

# Upstream demo clip; it is downloaded under the project's own filename.
VIDEO_URL := https://cvzone-storage.sfo3.cdn.digitaloceanspaces.com/courses/restaurat-analysis/videos/Cctv-Resturant-Demo.mp4

.PHONY: help install run test download-video

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies with uv
	uv sync

run: ## Run the project
	uv run python src/main.py

test: ## Run the tests
	uv run python -m unittest discover -s tests -v

download-video: ## Download the demo video (videos/restaurant-demo.mp4)
	mkdir -p videos
	wget -O videos/restaurant-demo.mp4 "$(VIDEO_URL)"
