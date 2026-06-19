# State Sales Forecasting Service

Production-style forecasting system for the assignment:

- Forecast next 8 weeks of sales for each state.
- Train and compare SARIMA, Prophet, XGBoost with lag features, and LSTM.
- Use time-series validation with no target leakage.
- Persist the selected model and forecast artifacts.
- Serve predictions with a FastAPI REST API.

## Project Layout

```text
app/
  forecasting/        Data loading, feature engineering, models, trainer
  services/           Artifact loading for API use
  main.py             FastAPI application
data/raw/             Source workbook
artifacts/            Generated training outputs
scripts/              CLI utilities
tests/                Unit tests
docs/                 Architecture, API, video script
```

## Setup

Python 3.11 is recommended because Prophet and TensorFlow publish the most reliable wheels there.

```powershell
cd D:\Projects\files\sales-forecasting-service
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Train Models

Full assignment run:

```powershell
python scripts\train.py
```

Faster local smoke run on two states with fewer LSTM epochs:

```powershell
python scripts\train.py --states Alabama,California --lstm-epochs 3
```

Training outputs:

- `artifacts/leaderboard.csv` - model metrics for every state/model candidate.
- `artifacts/model_selection.json` - selected model and metadata per state.
- `artifacts/forecasts.csv` - next 8 weekly forecasts served by the API.
- `artifacts/data_quality.csv` - weekly missing-date handling report.
- `artifacts/models/` - selected model artifacts.

## Run API

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

Example request:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/predict `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"states":["California","Texas"],"horizon_weeks":8}'
```

## Data Handling

The raw workbook has irregular dates, so preprocessing aggregates to weekly state-level sales using `W-SUN`. Missing weekly slots are interpolated, then forward/back filled if needed. The pipeline records all filled weeks in `artifacts/data_quality.csv`.

Lag and rolling features are built only from prior values:

- Lags: `t-1`, `t-7`, `t-30`
- Rolling: mean and standard deviation over 4, 8, and 12 prior weeks
- Calendar: day of week, week of year, month, quarter, year, month boundaries, US holiday flag

## Tests

```powershell
pytest
python -m compileall app scripts
```

## Notes

If an optional modeling dependency is missing, that model is marked as `dependency_missing` in the leaderboard instead of stopping the training job. In a clean Python 3.11 environment with `requirements.txt` installed, all four mandatory models are trained and compared.

