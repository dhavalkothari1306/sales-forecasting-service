from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.forecasting.data import load_raw_sales, summarize_raw_data, to_weekly_state_sales
from app.forecasting.metrics import regression_metrics
from app.forecasting.models import SeasonalNaiveModel, model_factories
from app.forecasting.persistence import ensure_dir, slugify, write_json


@dataclass(frozen=True)
class TrainingConfig:
    data_path: Path
    artifacts_dir: Path
    forecast_horizon_weeks: int = 8
    validation_weeks: int = 8
    freq: str = "W-SUN"
    states: tuple[str, ...] | None = None
    models: tuple[str, ...] | None = None
    lstm_epochs: int = 25
    random_seed: int = 42


def time_series_split(series_df: pd.DataFrame, validation_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = series_df.sort_values("date").reset_index(drop=True)
    if len(ordered) <= validation_weeks + 30:
        raise ValueError(
            f"Not enough observations ({len(ordered)}) for validation_weeks={validation_weeks}."
        )
    return ordered.iloc[:-validation_weeks].copy(), ordered.iloc[-validation_weeks:].copy()


def _candidate_status(error: Exception) -> str:
    return "dependency_missing" if isinstance(error, ImportError) else "failed"


def _clean_model_dir(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for child in model_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def train_all_states(config: TrainingConfig) -> dict[str, object]:
    started_at = datetime.now(timezone.utc)
    artifacts_dir = ensure_dir(config.artifacts_dir)
    processed_dir = ensure_dir(artifacts_dir / "processed")
    model_root = ensure_dir(artifacts_dir / "models")

    raw = load_raw_sales(config.data_path)
    raw_summary = summarize_raw_data(raw)
    bundle = to_weekly_state_sales(raw, freq=config.freq)
    weekly = bundle.weekly
    weekly.to_csv(processed_dir / "weekly_state_sales.csv", index=False)
    bundle.quality_report.to_csv(artifacts_dir / "data_quality.csv", index=False)

    available_states = tuple(sorted(weekly["state"].unique()))
    selected_states = config.states or available_states
    unknown_states = sorted(set(selected_states) - set(available_states))
    if unknown_states:
        raise ValueError(f"Unknown states requested: {unknown_states}")

    factories = model_factories(
        freq=config.freq,
        lstm_epochs=config.lstm_epochs,
        random_seed=config.random_seed,
    )
    selected_model_names = config.models or tuple(factories.keys())
    unknown_models = sorted(set(selected_model_names) - set(factories.keys()))
    if unknown_models:
        raise ValueError(f"Unknown models requested: {unknown_models}")

    leaderboard_rows: list[dict[str, object]] = []
    forecast_rows: list[dict[str, object]] = []
    selected_models: dict[str, dict[str, object]] = {}
    generated_at = datetime.now(timezone.utc).isoformat()

    for state in selected_states:
        state_series = weekly[weekly["state"] == state].sort_values("date").reset_index(drop=True)
        train_df, validation_df = time_series_split(state_series, config.validation_weeks)
        y_validation = validation_df["y"].to_numpy(dtype=float)
        candidate_rows: list[dict[str, object]] = []

        for model_name in selected_model_names:
            model_started = time.perf_counter()
            row: dict[str, object] = {
                "state": state,
                "model_name": model_name,
                "status": "ok",
                "error": "",
                "train_rows": int(len(train_df)),
                "validation_rows": int(len(validation_df)),
                "runtime_seconds": 0.0,
            }
            try:
                model = factories[model_name]()
                model.fit(train_df)
                validation_pred = model.predict(len(validation_df), train_df)
                metric_values = regression_metrics(y_validation, validation_pred)
                row.update(metric_values)
            except Exception as exc:
                row["status"] = _candidate_status(exc)
                row["error"] = str(exc)
                row.update({"mae": np.nan, "rmse": np.nan, "mape": np.nan, "smape": np.nan})
            finally:
                row["runtime_seconds"] = round(time.perf_counter() - model_started, 3)
            leaderboard_rows.append(row)
            candidate_rows.append(row)

        successful = [row for row in candidate_rows if row["status"] == "ok"]
        used_fallback = False
        if not successful:
            fallback_started = time.perf_counter()
            fallback = SeasonalNaiveModel(freq=config.freq)
            fallback.fit(train_df)
            validation_pred = fallback.predict(len(validation_df), train_df)
            metric_values = regression_metrics(y_validation, validation_pred)
            fallback_row: dict[str, object] = {
                "state": state,
                "model_name": fallback.name,
                "status": "ok",
                "error": "Fallback used because all mandatory model candidates failed.",
                "train_rows": int(len(train_df)),
                "validation_rows": int(len(validation_df)),
                "runtime_seconds": round(time.perf_counter() - fallback_started, 3),
                **metric_values,
            }
            leaderboard_rows.append(fallback_row)
            successful = [fallback_row]
            used_fallback = True

        best_row = min(successful, key=lambda item: float(item["smape"]))
        best_model_name = str(best_row["model_name"])
        best_factory = (
            (lambda: SeasonalNaiveModel(freq=config.freq))
            if best_model_name == "seasonal_naive"
            else factories[best_model_name]
        )
        full_model = best_factory()
        full_model.fit(state_series)
        future_predictions = full_model.predict(config.forecast_horizon_weeks, state_series)
        last_date = pd.Timestamp(state_series["date"].max())
        future_dates = pd.date_range(last_date, periods=config.forecast_horizon_weeks + 1, freq=config.freq)[1:]

        state_slug = slugify(state)
        model_dir = model_root / state_slug / best_model_name
        _clean_model_dir(model_dir)
        full_model.save(model_dir)

        selected_models[state] = {
            "model_name": best_model_name,
            "validation_smape": float(best_row["smape"]),
            "validation_rmse": float(best_row["rmse"]),
            "validation_mae": float(best_row["mae"]),
            "model_artifact_dir": str(model_dir.relative_to(artifacts_dir)),
            "used_fallback": used_fallback,
        }

        for horizon_step, (future_date, forecast_value) in enumerate(
            zip(future_dates, future_predictions, strict=True),
            start=1,
        ):
            forecast_rows.append(
                {
                    "state": state,
                    "forecast_date": future_date.date().isoformat(),
                    "horizon_week": horizon_step,
                    "forecast": float(max(0.0, forecast_value)),
                    "model_name": best_model_name,
                    "validation_smape": float(best_row["smape"]),
                    "validation_rmse": float(best_row["rmse"]),
                    "validation_mae": float(best_row["mae"]),
                    "last_observed_week": last_date.date().isoformat(),
                    "generated_at": generated_at,
                }
            )

    leaderboard = pd.DataFrame(leaderboard_rows)
    forecasts = pd.DataFrame(forecast_rows)
    leaderboard.to_csv(artifacts_dir / "leaderboard.csv", index=False)
    forecasts.to_csv(artifacts_dir / "forecasts.csv", index=False)

    metadata = {
        "generated_at": generated_at,
        "duration_seconds": round((datetime.now(timezone.utc) - started_at).total_seconds(), 3),
        "data_path": str(config.data_path),
        "raw_data": raw_summary,
        "weekly_frequency": config.freq,
        "forecast_horizon_weeks": config.forecast_horizon_weeks,
        "validation_weeks": config.validation_weeks,
        "states_trained": len(selected_states),
        "available_states": list(available_states),
        "models_requested": list(selected_model_names),
        "selected_models": selected_models,
    }
    write_json(artifacts_dir / "model_selection.json", metadata)
    return metadata
