# 📊 State Sales Forecasting Service

> **Production-grade backend forecasting pipeline and REST API.**

This service aggregates and clean-resamples state-level sales data to generate robust, 8-week future sales forecasts. By implementing a automated tournament-style selection framework, it evaluates multiple modeling algorithms—**SARIMA, Prophet, XGBoost with lag features, and LSTM**—to choose and serve the best-performing model dynamically for each state.

---

## 🚀 Key Features

*   **Resilient Data Preprocessing:** Linear interpolation of missing dates, automatic aggregation to weekly `W-SUN` frequency, and isolated Excel subprocess parsing to avoid file locks.
*   **Leakage-Controlled Feature Engineering:** Generates lag values ($t-1, t-7, t-30$), rolling features from shifted targets (`y.shift(1)`), calendar coordinates, and custom US federal holiday mappings.
*   **Automated Model Tournaments:** Splits data using time-series cross-validation, trains four diverse algorithms, refits the winner (lowest validation sMAPE) on complete history, and includes an automated fallback to Seasonal Naive.
*   **Sub-millisecond Serving:** High-performance, stateless FastAPI REST API that serves forecasts from pre-calculated artifacts.
*   **Enterprise Tooling:** Complete test coverage with `pytest`, data quality reporting, and flexible configuration via environment files or CLI parameters.

---

## 📁 Repository Structure

*   [`app/`](file:///d:/Projects/files/sales-forecasting-service/app) - Core application package.
    *   [`forecasting/`](file:///d:/Projects/files/sales-forecasting-service/app/forecasting) - Core ML modules (data loading, features, models, trainer, persistence).
    *   [`services/`](file:///d:/Projects/files/sales-forecasting-service/app/services) - Service layer for artifact caching and loading.
    *   [`main.py`](file:///d:/Projects/files/sales-forecasting-service/app/main.py) - FastAPI API application.
*   [`data/`](file:///d:/Projects/files/sales-forecasting-service/data) - Raw input Excel workbooks.
*   [`artifacts/`](file:///d:/Projects/files/sales-forecasting-service/artifacts) - Generated forecasts, leaderboards, data quality logs, and serialized model files.
*   [`scripts/`](file:///d:/Projects/files/sales-forecasting-service/scripts) - Command-line interfaces for training and data profiling.
*   [`tests/`](file:///d:/Projects/files/sales-forecasting-service/tests) - Unit and integration testing suites.
*   [`docs/`](file:///d:/Projects/files/sales-forecasting-service/docs) - Technical specification files.

---

## 📖 Documentation Index

To explore specific sections of the service design and operation, refer to the guides below:

1.  **[Setup & Installation Guide](file:///d:/Projects/files/sales-forecasting-service/docs/SETUP_GUIDE.md)**
    *   Step-by-step instructions for Python 3.11 virtual environment configuration.
    *   Environment variables, CLI flags, profiling tools, and how to execute the testing suite.
2.  **[System Architecture](file:///d:/Projects/files/sales-forecasting-service/docs/ARCHITECTURE.md)**
    *   Detailed explanation of the offline data processing pipeline, feature engineering, recursive predictions, evaluation metrics, and the tournament workflow.
    *   System architecture flow diagram (Mermaid).
3.  **[API Reference](file:///d:/Projects/files/sales-forecasting-service/docs/API.md)**
    *   Detailed HTTP endpoints documentation for `/health`, `/states`, `/models/selection`, `/models/leaderboard`, `/predict`, and `/predict/{state}`.
    *   Request/response JSON schemas, descriptions, and cURL commands.
4.  **[Submission Checklist](file:///d:/Projects/files/sales-forecasting-service/docs/SUBMISSION_CHECKLIST.md)**
    *   Summary of files, data configurations, and evaluation targets.
5.  **[Presentation Video Script](file:///d:/Projects/files/sales-forecasting-service/docs/VIDEO_SCRIPT.md)**
    *   A structured 3-5 minute video outline explaining the system design, tournament selections, and API response speed.

---

## ⚡ Quick Start

### 1. Configure the Environment
Ensure Python 3.11 is installed, then run:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run automated Model Tournament
Run the training command to preprocess the Excel sheets, train the candidate models, and output validation metrics:
```powershell
# Full run on all 43 states
python scripts/train.py

# Faster test run on selected states
python scripts/train.py --states Alabama,California --lstm-epochs 3
```

### 3. Launch the API Service
Start the FastAPI development server:
```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to view the interactive Swagger API documentation.

### 4. Query predictions
Get the 8-week forecast for California:
```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/predict `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"states":["California"],"horizon_weeks":8}'
```
