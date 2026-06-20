# ⚙️ Setup & Installation Guide

This guide describes how to configure your development environment, manage settings using environment variables, execute CLI scripts for model training, and run the testing suite.

---

## 🐍 Python Installation Requirements

*   **Version Recommendation**: **Python 3.11** is recommended. Prophet and TensorFlow publish pre-compiled wheels for Python 3.11, ensuring a smooth installation process on Windows.
*   **Virtual Environment**: Keep project dependencies isolated from your system Python.

---

## 🛠️ Step-by-Step Local Setup

### 1. Initialize and Activate Virtual Environment
Open a PowerShell terminal, navigate to the project directory, and initialize a virtual environment:
```powershell
# Create venv
python -m venv .venv

# Activate venv on Windows (PowerShell)
.\.venv\Scripts\activate

# Or on macOS/Linux
source .venv/bin/activate
```

### 2. Install Project Dependencies
Install the required scientific, machine learning, and web framework libraries:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```
*Note: Installing packages like TensorFlow and Prophet may take several minutes on some systems.*

---

## ⚙️ Configuration & Environment Variables

You can customize the runtime settings of the service by exporting environment variables or placing a `.env` file at the root of the project. The configuration is parsed by [`app/config.py`](file:///d:/Projects/files/sales-forecasting-service/app/config.py):

| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| `DATA_PATH` | `data/raw/Forecasting Case- Study.xlsx` | File path to the raw workbook input. |
| `ARTIFACTS_DIR` | `artifacts/` | Directory where output files, reports, and models are written. |
| `FORECAST_HORIZON_WEEKS` | `8` | Number of weeks to forecast. |
| `VALIDATION_WEEKS` | `8` | Size of the validation set window (weeks) for sMAPE comparison. |
| `WEEKLY_FREQUENCY` | `W-SUN` | Resampling frequency (defaults to weekly, Sunday ending). |
| `LSTM_EPOCHS` | `25` | Training epochs for the LSTM model. |
| `RANDOM_SEED` | `42` | Seed to initialize model weights and numpy operations. |

---

## 📊 Run Data Profiling Utility

Use `profile_data.py` to examine the raw Excel sheet and check for missing dates and aggregation counts before running full training.

```powershell
python scripts/profile_data.py
```

*   **Output Example**:
    ```json
    {
      "rows": 8084,
      "states": 43,
      "first_date": "2024-01-07",
      "last_date": "2026-06-14",
      "missing_sales": 0,
      "negative_sales": 0,
      "weekly_rows": 11008,
      "weekly_states": 43,
      "weeks_per_state": 256,
      "filled_missing_weeks_total": 3913
    }
    ```

---

## 🏋️ Run automated Model Training CLI

The training script [`scripts/train.py`](file:///d:/Projects/files/sales-forecasting-service/scripts/train.py) provides a flexible command-line interface to configure tournament parameters.

### CLI Options

| Flag | Parameter | Description |
| :--- | :--- | :--- |
| `--data-path` | `Path` | Path override for raw data source. |
| `--artifacts-dir` | `Path` | Directory where artifacts will be saved. |
| `--horizon` | `int` | Length of forecast predictions (weeks). |
| `--validation-weeks`| `int` | Size of the validation window. |
| `--freq` | `str` | Series week frequency (e.g. `W-SUN`, `W-MON`). |
| `--states` | `str` | Comma-separated list of states to train (defaults to all). |
| `--models` | `str` | Comma-separated list of models: `sarima,prophet,xgboost,lstm`. |
| `--lstm-epochs` | `int` | Number of training epochs for the LSTM network. |
| `--random-seed` | `int` | Random seed number. |

### Commands Examples

*   **Full Production Run** (evaluates all 43 states across all models):
    ```powershell
    python scripts/train.py
    ```

*   **Smoke Test Run** (evaluates selected states with fewer LSTM training epochs):
    ```powershell
    python scripts/train.py --states Alabama,California --lstm-epochs 3
    ```

*   **Algorithm Specific Run** (evaluates only Prophet and XGBoost):
    ```powershell
    python scripts/train.py --models prophet,xgboost
    ```

---

## 🧪 Running the Test Suite

Verify system stability and feature engineering calculations with `pytest`:

```powershell
# Run all tests (specifying a local basetemp avoids Windows AppData permission issues)
pytest --basetemp=tests/tmp

# Compile validation
python -m compileall app scripts
```
The test suite validates:
1.  **Data Resampling**: Correct interpolation of missing weeks and aggregation correctness.
2.  **Lag features**: Ensures rolling mean calculations match their expected shapes.
3.  **Forecast Store**: Checks state lists, horizon lengths, and invalid state queries.
