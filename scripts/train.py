from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.forecasting.trainer import TrainingConfig, train_all_states


def _csv_tuple(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train state-level sales forecasting models.")
    parser.add_argument("--data-path", type=Path, default=settings.data_path)
    parser.add_argument("--artifacts-dir", type=Path, default=settings.artifacts_dir)
    parser.add_argument("--horizon", type=int, default=settings.forecast_horizon_weeks)
    parser.add_argument("--validation-weeks", type=int, default=settings.validation_weeks)
    parser.add_argument("--freq", default=settings.weekly_frequency)
    parser.add_argument("--states", default=None, help="Comma-separated states, e.g. Alabama,California")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated models from sarima,prophet,xgboost,lstm.",
    )
    parser.add_argument("--lstm-epochs", type=int, default=settings.lstm_epochs)
    parser.add_argument("--random-seed", type=int, default=settings.random_seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        data_path=args.data_path,
        artifacts_dir=args.artifacts_dir,
        forecast_horizon_weeks=args.horizon,
        validation_weeks=args.validation_weeks,
        freq=args.freq,
        states=_csv_tuple(args.states),
        models=_csv_tuple(args.models),
        lstm_epochs=args.lstm_epochs,
        random_seed=args.random_seed,
    )
    metadata = train_all_states(config)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
