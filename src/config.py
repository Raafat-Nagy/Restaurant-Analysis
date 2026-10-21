from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGIONS_CONFIG_PATH = PROJECT_ROOT / "config" / "regions.json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(PROJECT_ROOT / ".env"))

    VIDEO_PATH: str = "videos/restaurant-demo.mp4"
    MODEL_PATH: str = "models/yolo26s-pose.pt"
    CONF: float = 0.40
    IOU: float = 0.50
    IMGSZ: int = 640
    COUNT_INCREASE_DELAY: float = 2.0
    COUNT_DECREASE_DELAY: float = 10.0
    COUNT_EMPTY_DELAY: float = 10.0
    COUNT_HISTORY_SIZE: int = 10
    SAVE_OUTPUT_VIDEO: bool = False
    OUTPUT_VIDEO_PATH: str = "videos/output/restaurant_analysis_output.mp4"
    RESIZE_VIDEO: bool = False
    VIDEO_WIDTH: int = Field(1280, gt=0)
    SHOW_DASHBOARD: bool = True
    DASHBOARD_WIDTH: int = Field(200, gt=0)

    @field_validator("OUTPUT_VIDEO_PATH")
    @classmethod
    def _resolve_output_path(cls, value: str) -> str:
        """Relative output paths resolve against the project root, not the CWD."""
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path)


settings = Settings()
