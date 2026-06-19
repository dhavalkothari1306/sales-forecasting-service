from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Query

from app import __version__
from app.config import get_settings
from app.schemas import (
    HealthResponse,
    LeaderboardResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
    StateForecast,
)
from app.services.forecast_store import ArtifactNotReadyError, ForecastStore


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Forecast next 8 weeks of state-level sales using automatically selected models.",
)


@lru_cache(maxsize=1)
def get_store() -> ForecastStore:
    return ForecastStore(settings.artifacts_dir)


def _artifact_error(exc: Exception) -> HTTPException:
    status_code = 503 if isinstance(exc, ArtifactNotReadyError) else 400
    return HTTPException(status_code=status_code, detail=str(exc))


def _response_from_frame(frame, horizon_weeks: int) -> PredictionResponse:
    forecasts: list[StateForecast] = []
    generated_at = ""
    for state, group in frame.groupby("state", sort=True):
        group = group.sort_values("horizon_week")
        generated_at = str(group["generated_at"].iloc[0])
        forecasts.append(
            StateForecast(
                state=str(state),
                model_name=str(group["model_name"].iloc[0]),
                validation_smape=float(group["validation_smape"].iloc[0]),
                validation_rmse=float(group["validation_rmse"].iloc[0]),
                validation_mae=float(group["validation_mae"].iloc[0]),
                last_observed_week=str(group["last_observed_week"].iloc[0]),
                points=[
                    {
                        "forecast_date": str(row.forecast_date),
                        "horizon_week": int(row.horizon_week),
                        "forecast": float(row.forecast),
                    }
                    for row in group.itertuples(index=False)
                ],
            )
        )
    return PredictionResponse(
        horizon_weeks=horizon_weeks,
        generated_at=generated_at,
        forecasts=forecasts,
    )


@app.get("/health", response_model=HealthResponse)
def health(store: ForecastStore = Depends(get_store)) -> HealthResponse:
    return HealthResponse(
        status="ready" if store.ready else "artifacts_missing",
        artifacts_ready=store.ready,
        version=__version__,
    )


@app.post("/reload", response_model=HealthResponse)
def reload_artifacts(store: ForecastStore = Depends(get_store)) -> HealthResponse:
    store.reload()
    return health(store)


@app.get("/states", response_model=list[str])
def states(store: ForecastStore = Depends(get_store)) -> list[str]:
    try:
        return store.states()
    except Exception as exc:
        raise _artifact_error(exc) from exc


@app.get("/models/selection", response_model=MetadataResponse)
def model_selection(store: ForecastStore = Depends(get_store)) -> MetadataResponse:
    try:
        store.require_ready()
        return MetadataResponse(metadata=store.metadata)
    except Exception as exc:
        raise _artifact_error(exc) from exc


@app.get("/models/leaderboard", response_model=LeaderboardResponse)
def leaderboard(store: ForecastStore = Depends(get_store)) -> LeaderboardResponse:
    try:
        store.require_ready()
        rows = store.leaderboard.where(store.leaderboard.notna(), None).to_dict(orient="records")
        return LeaderboardResponse(rows=rows)
    except Exception as exc:
        raise _artifact_error(exc) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    store: ForecastStore = Depends(get_store),
) -> PredictionResponse:
    try:
        frame = store.predictions(states=request.states, horizon_weeks=request.horizon_weeks)
        return _response_from_frame(frame, request.horizon_weeks)
    except Exception as exc:
        raise _artifact_error(exc) from exc


@app.get("/predict/{state}", response_model=PredictionResponse)
def predict_state(
    state: str,
    horizon_weeks: int = Query(default=8, ge=1, le=52),
    store: ForecastStore = Depends(get_store),
) -> PredictionResponse:
    try:
        frame = store.predictions(states=[state], horizon_weeks=horizon_weeks)
        return _response_from_frame(frame, horizon_weeks)
    except Exception as exc:
        raise _artifact_error(exc) from exc

