from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    app_name: str = "State Sales Forecasting Service"
    app_version: str = "1.0.0"
    project_root: Path = PROJECT_ROOT
    data_path: Path = PROJECT_ROOT / "data" / "raw" / "Forecasting Case- Study.xlsx"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    forecast_horizon_weeks: int = 8
    validation_weeks: int = 8
    weekly_frequency: str = "W-SUN"
    lstm_epochs: int = 25
    random_seed: int = 42


def get_settings() -> Settings:
    return Settings(
        data_path=Path(os.getenv("DATA_PATH", str(Settings.data_path))),
        artifacts_dir=Path(os.getenv("ARTIFACTS_DIR", str(Settings.artifacts_dir))),
        forecast_horizon_weeks=int(os.getenv("FORECAST_HORIZON_WEEKS", "8")),
        validation_weeks=int(os.getenv("VALIDATION_WEEKS", "8")),
        weekly_frequency=os.getenv("WEEKLY_FREQUENCY", "W-SUN"),
        lstm_epochs=int(os.getenv("LSTM_EPOCHS", "25")),
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
    )

