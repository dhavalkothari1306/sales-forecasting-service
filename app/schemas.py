from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    artifacts_ready: bool
    version: str


class PredictionRequest(BaseModel):
    states: list[str] | None = Field(
        default=None,
        description="Optional list of states. When omitted, forecasts for all states are returned.",
    )
    horizon_weeks: int = Field(default=8, ge=1, le=52)


class ForecastPoint(BaseModel):
    forecast_date: str
    horizon_week: int
    forecast: float


class StateForecast(BaseModel):
    state: str
    model_name: str
    validation_smape: float
    validation_rmse: float
    validation_mae: float
    last_observed_week: str
    points: list[ForecastPoint]


class PredictionResponse(BaseModel):
    horizon_weeks: int
    generated_at: str
    forecasts: list[StateForecast]


class MetadataResponse(BaseModel):
    metadata: dict[str, Any]


class LeaderboardResponse(BaseModel):
    rows: list[dict[str, Any]]

