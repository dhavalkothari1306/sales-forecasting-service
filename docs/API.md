# 🔌 API Reference

The State Sales Forecasting Service uses a **FastAPI** REST interface. It loads pre-computed predictions and model metrics into memory upon start.

---

## 🏃 Starting the Server

Launch the API using `uvicorn`:
```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- Interactive Swagger docs are accessible at: `http://127.0.0.1:8000/docs`
- Redoc alternative UI is accessible at: `http://127.0.0.1:8000/redoc`

---

## 🛠️ Endpoint Summary

### 1. Service Health
Check the status of the API and verify that prediction artifacts are loaded and ready.

*   **URL:** `/health`
*   **Method:** `GET`
*   **Response Schema (`HealthResponse`):**
    ```json
    {
      "status": "ready",
      "artifacts_ready": true,
      "version": "1.0.0"
    }
    ```

---

### 2. Reload Artifacts
Manually reload predictions and metadata from files in the `artifacts/` folder without restarting the application process.

*   **URL:** `/reload`
*   **Method:** `POST`
*   **Response Schema (`HealthResponse`):**
    ```json
    {
      "status": "ready",
      "artifacts_ready": true,
      "version": "1.0.0"
    }
    ```

---

### 3. List States
Retrieve all unique state names present in the trained dataset.

*   **URL:** `/states`
*   **Method:** `GET`
*   **Response Example:**
    ```json
    [
      "Alabama",
      "California",
      "Texas",
      "Washington"
    ]
    ```

---

### 4. Leaderboard Metrics
Retrieve the validation performance metrics (sMAPE, MAE, RMSE) and execution runtimes for all candidate models across all states.

*   **URL:** `/models/leaderboard`
*   **Method:** `GET`
*   **Response Example (`LeaderboardResponse`):**
    ```json
    {
      "rows": [
        {
          "state": "California",
          "model_name": "prophet",
          "status": "ok",
          "error": "",
          "train_rows": 240,
          "validation_rows": 8,
          "runtime_seconds": 1.15,
          "mae": 1520.4,
          "rmse": 1845.2,
          "mape": 0.054,
          "smape": 5.25
        },
        {
          "state": "California",
          "model_name": "xgboost",
          "status": "ok",
          "error": "",
          "train_rows": 240,
          "validation_rows": 8,
          "runtime_seconds": 0.45,
          "mae": 1950.1,
          "rmse": 2105.8,
          "mape": 0.068,
          "smape": 6.75
        }
      ]
    }
    ```

---

### 5. Model Selection Metadata
Get the detailed configurations, dataset counts, and the selected model chosen for each state.

*   **URL:** `/models/selection`
*   **Method:** `GET`
*   **Response Example (`MetadataResponse`):**
    ```json
    {
      "metadata": {
        "generated_at": "2026-06-20T12:00:00Z",
        "duration_seconds": 12.35,
        "weekly_frequency": "W-SUN",
        "forecast_horizon_weeks": 8,
        "validation_weeks": 8,
        "states_trained": 43,
        "selected_models": {
          "California": {
            "model_name": "prophet",
            "validation_smape": 5.25,
            "validation_rmse": 1845.2,
            "validation_mae": 1520.4,
            "model_artifact_dir": "models/california/prophet",
            "used_fallback": false
          }
        }
      }
    }
    ```

---

### 6. Batch Predict (POST)
Generate forecasts for one or more states over a specific horizon.

*   **URL:** `/predict`
*   **Method:** `POST`
*   **Request Headers:** `Content-Type: application/json`
*   **Request Schema (`PredictionRequest`):**
    ```json
    {
      "states": ["California", "Texas"],
      "horizon_weeks": 8
    }
    ```
    *Note: `states` is optional. If left null or omitted, predictions for all available states are returned. `horizon_weeks` must be between 1 and 52.*

*   **Response Schema (`PredictionResponse`):**
    ```json
    {
      "horizon_weeks": 8,
      "generated_at": "2026-06-20T12:00:00Z",
      "forecasts": [
        {
          "state": "California",
          "model_name": "prophet",
          "validation_smape": 5.25,
          "validation_rmse": 1845.2,
          "validation_mae": 1520.4,
          "last_observed_week": "2026-06-14",
          "points": [
            {
              "forecast_date": "2026-06-21",
              "horizon_week": 1,
              "forecast": 32000.5
            },
            {
              "forecast_date": "2026-06-28",
              "horizon_week": 2,
              "forecast": 31500.2
            }
          ]
        }
      ]
    }
    ```

---

### 7. State Predict (GET)
Generate forecasts for a single state via URL parameter path.

*   **URL:** `/predict/{state}`
*   **Method:** `GET`
*   **Parameters:**
    *   `state` (Path, string, Required)
    *   `horizon_weeks` (Query, integer, Optional, default=8, min=1, max=52)
*   **Response Schema (`PredictionResponse`):** Same schema as `/predict` response format.

---

## ⚠️ Error and Exception Handling

*   **`503 Service Unavailable`**: Returned if the offline model tournament has not yet run. You must run `python scripts/train.py` to generate predictions and serializations first.
    ```json
    {
      "detail": "Forecast artifacts are not available. Run `python scripts/train.py` first."
    }
    ```
*   **`422 Unprocessable Entity`**: Returned if validation failed (e.g. `horizon_weeks` is less than 1 or greater than 52).
*   **`400 Bad Request`**: Returned if requesting prediction for an unknown state or if the horizon exceeds the trained model horizon.

---

## 💻 Query Examples

### cURL (Bash)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"states":["California"], "horizon_weeks":4}'
```

### PowerShell
```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"states":["California", "Texas"],"horizon_weeks":8}'
```
