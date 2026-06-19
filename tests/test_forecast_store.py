from __future__ import annotations

import json

import pandas as pd

from app.services.forecast_store import ForecastStore


def test_forecast_store_filters_state_and_horizon(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "state": "A",
                "forecast_date": "2024-01-14",
                "horizon_week": 1,
                "forecast": 10.0,
                "model_name": "xgboost",
                "validation_smape": 1.0,
                "validation_rmse": 2.0,
                "validation_mae": 3.0,
                "last_observed_week": "2024-01-07",
                "generated_at": "2024-01-01T00:00:00+00:00",
            },
            {
                "state": "A",
                "forecast_date": "2024-01-21",
                "horizon_week": 2,
                "forecast": 11.0,
                "model_name": "xgboost",
                "validation_smape": 1.0,
                "validation_rmse": 2.0,
                "validation_mae": 3.0,
                "last_observed_week": "2024-01-07",
                "generated_at": "2024-01-01T00:00:00+00:00",
            },
            {
                "state": "B",
                "forecast_date": "2024-01-14",
                "horizon_week": 1,
                "forecast": 20.0,
                "model_name": "lstm",
                "validation_smape": 4.0,
                "validation_rmse": 5.0,
                "validation_mae": 6.0,
                "last_observed_week": "2024-01-07",
                "generated_at": "2024-01-01T00:00:00+00:00",
            },
        ]
    ).to_csv(tmp_path / "forecasts.csv", index=False)
    pd.DataFrame([{"state": "A", "model_name": "xgboost", "status": "ok"}]).to_csv(
        tmp_path / "leaderboard.csv",
        index=False,
    )
    (tmp_path / "model_selection.json").write_text(
        json.dumps({"selected_models": {"A": {"model_name": "xgboost"}}}),
        encoding="utf-8",
    )

    store = ForecastStore(tmp_path)
    result = store.predictions(states=["A"], horizon_weeks=1)

    assert result["state"].tolist() == ["A"]
    assert result["horizon_week"].tolist() == [1]

