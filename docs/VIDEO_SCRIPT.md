# Presentation Video Script (Production-Ready Selection)

**Target length**: 3 to 5 minutes.
**Goal**: Show the evaluator that this is a professional, production-ready backend service rather than a simple machine learning script.

---

## Intro: The Production-Grade Design (30 seconds)

*Show slide/code editor.*

**Talking Points**:
> "This isn't just a model script—it is built like a real, resilient backend service. 
> To guarantee that the application is production-ready, we separated **offline model training** from **online prediction serving**. 
> The API serves forecasts deterministically from serialized artifacts in sub-milliseconds, rather than training on the fly. 
> Furthermore, the pipeline is built for robustness: Excel workbook reading is isolated in a separate subprocess to prevent Windows file-locking, and we implement automated fallback strategies if models fail."

---

## 1. Automated Data Cleaning & Resampling (45 seconds)

*Open the Excel sheet briefly, then show terminal.*

**Talking Points**:
> "Our raw dataset contains irregular observations across 43 states. 
> To prepare it for forecasting, the pipeline resamples the raw data into a complete weekly series per state. 
> We linearly interpolate missing dates and track the filled weeks in a dedicated data quality report to ensure transparency."

**Run command**:
```powershell
python scripts\profile_data.py
```

*Show JSON output on screen.*

> "Here we can see the results: 8,084 raw rows resampled into 11,008 clean weekly observations across all 43 states, with 3,913 missing weeks successfully filled and logged."

---

## 2. Advanced Feature Engineering & Leakage Prevention (45 seconds)

*Open `app/forecasting/features.py`.*

**Talking Points**:
> "For supervised models, we generate lags ($t-1$, $t-7$, $t-30$), rolling averages, rolling standard deviations, and calendar coordinates. 
> Crucially, rolling features are built using shifted target values (`y.shift(1)`), ensuring **zero target leakage** during training and recursive forecasting.
> We also overlay a custom US Federal Holiday flag, tracking whether a holiday occurred in the 7-day window ending on each weekly data point."

---

## 3. Automated Model Tournament & Fallback (60 seconds)

*Open `app/forecasting/trainer.py`.*

**Talking Points**:
> "To select the best algorithm, we run an automated tournament for each state, holding out the final 8 weeks for validation. 
> We train four distinct architectures: **SARIMA**, **Prophet**, **XGBoost**, and **LSTM**. 
> The framework compares validation metrics and automatically refits the winning model (lowest validation sMAPE) on the complete state history before generating the final 8-week horizon forecasts.
> If all candidate models fail due to library or system issues, the pipeline automatically defaults to a Seasonal Naive baseline to prevent system crash."

**Run quick-test command (Alabama & California)**:
```powershell
python scripts\train.py --states Alabama,California --lstm-epochs 3
```

*(Mention that the final submission was trained on all 43 states by running: `python scripts\train.py`)*

---

## 4. Persisted Artifacts (30 seconds)

*Expand `artifacts/` folder.*

**Talking Points**:
> "The tournament outputs four main artifacts:
> 1. `data_quality.csv` – tracks missing and filled data per state.
> 2. `leaderboard.csv` – metrics for all model candidates.
> 3. `model_selection.json` – chosen models and execution metadata.
> 4. `forecasts.csv` – final 8-week predictions.
> This makes our API stateless and deployment-friendly."

---

## 5. API Demonstration (60 seconds)

*Start the server in terminal.*

**Start Command**:
```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

*Open Swagger UI at `http://127.0.0.1:8000/docs`.*

**Talking Points**:
> "We expose our results through a clean, self-documenting FastAPI service. 
> The `/health` endpoint checks artifact readiness, `/states` lists available states, and `/models/leaderboard` returns validation ranks."

*Execute test predict call.*

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/predict `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"states":["California"],"horizon_weeks":8}'
```

*Show response.*

> "A simple POST request to `/predict` returns California's 8-week forecast in sub-milliseconds.
> With automated tournaments, robust fallbacks, strict data leakage prevention, and sub-millisecond API serving, this system is ready for production."
