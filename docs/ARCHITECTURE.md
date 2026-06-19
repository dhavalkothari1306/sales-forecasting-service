# Architecture

## Goal

The service forecasts the next 8 weekly sales values for each state from the assignment workbook. It is split into an offline training pipeline and an online API.

## Offline Training Flow

1. `scripts/train.py` reads `data/raw/Forecasting Case- Study.xlsx`.
2. `app.forecasting.data` normalizes columns, parses dates, aggregates sales by state/date, and resamples to weekly `W-SUN` series.
3. Missing weekly observations are interpolated and logged in `artifacts/data_quality.csv`.
4. Each state is split with time-series logic: all weeks except the last validation window are training data; the final weeks are validation data.
5. The trainer fits:
   - SARIMA
   - Prophet
   - XGBoost with lag and rolling features
   - LSTM
6. Validation metrics are written to `artifacts/leaderboard.csv`.
7. The lowest validation sMAPE model is refit on the full state history.
8. The selected model forecasts the next 8 weeks.
9. Forecasts and model metadata are written to `artifacts/forecasts.csv` and `artifacts/model_selection.json`.

## Online API Flow

The FastAPI service does not retrain on request. It loads the latest training artifacts and serves deterministic forecasts.

Key endpoints:

- `GET /health`
- `GET /states`
- `GET /models/leaderboard`
- `GET /models/selection`
- `POST /predict`
- `GET /predict/{state}`

## Leakage Controls

The validation period is the last `N` weeks for each state. The XGBoost and LSTM models forecast validation recursively from training history, matching the production setup where future target values are unknown.

Feature engineering uses prior values only:

- `lag_1`, `lag_7`, `lag_30`
- Rolling statistics from `y.shift(1)`
- Calendar features from the target date
- Holiday flag for any US federal holiday inside the weekly window

## Failure Handling

Model dependencies are isolated inside wrappers. If one model cannot train, the leaderboard records `dependency_missing` or `failed` and training continues for the other candidates. A seasonal naive fallback is used only if every mandatory model fails for a state, so the API can still produce a response while making the fallback explicit.

