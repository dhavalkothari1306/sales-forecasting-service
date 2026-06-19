from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.forecasting.persistence import read_json


class ArtifactNotReadyError(RuntimeError):
    pass


class ForecastStore:
    def __init__(self, artifacts_dir: str | Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.forecasts = pd.DataFrame()
        self.leaderboard = pd.DataFrame()
        self.metadata: dict[str, object] = {}
        self.reload()

    def reload(self) -> None:
        forecasts_path = self.artifacts_dir / "forecasts.csv"
        leaderboard_path = self.artifacts_dir / "leaderboard.csv"
        metadata_path = self.artifacts_dir / "model_selection.json"

        if forecasts_path.exists():
            self.forecasts = pd.read_csv(forecasts_path)
        else:
            self.forecasts = pd.DataFrame()

        if leaderboard_path.exists():
            self.leaderboard = pd.read_csv(leaderboard_path)
        else:
            self.leaderboard = pd.DataFrame()

        self.metadata = read_json(metadata_path) if metadata_path.exists() else {}

    @property
    def ready(self) -> bool:
        return not self.forecasts.empty and bool(self.metadata)

    def require_ready(self) -> None:
        if not self.ready:
            raise ArtifactNotReadyError(
                "Forecast artifacts are not available. Run `python scripts/train.py` first."
            )

    def states(self) -> list[str]:
        self.require_ready()
        return sorted(self.forecasts["state"].unique().tolist())

    def max_horizon(self) -> int:
        self.require_ready()
        return int(self.forecasts["horizon_week"].max())

    def predictions(self, states: list[str] | None = None, horizon_weeks: int = 8) -> pd.DataFrame:
        self.require_ready()
        if horizon_weeks < 1:
            raise ValueError("horizon_weeks must be at least 1.")
        max_horizon = self.max_horizon()
        if horizon_weeks > max_horizon:
            raise ValueError(f"horizon_weeks cannot exceed trained artifact horizon {max_horizon}.")

        frame = self.forecasts[self.forecasts["horizon_week"] <= horizon_weeks].copy()
        if states:
            available = set(self.states())
            unknown = sorted(set(states) - available)
            if unknown:
                raise ValueError(f"Unknown states requested: {unknown}")
            frame = frame[frame["state"].isin(states)]
        return frame.sort_values(["state", "horizon_week"]).reset_index(drop=True)

    def selection_summary(self) -> list[dict[str, object]]:
        self.require_ready()
        selected = self.metadata.get("selected_models", {})
        return [
            {"state": state, **details}
            for state, details in sorted(selected.items(), key=lambda item: item[0])
        ]

