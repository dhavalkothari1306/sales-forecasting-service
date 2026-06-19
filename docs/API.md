# API Reference

Start the service:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Health

```http
GET /health
```

Returns whether forecast artifacts are available.

## States

```http
GET /states
```

Returns all states available in the trained artifact.

## Leaderboard

```http
GET /models/leaderboard
```

Returns validation metrics and statuses for each state/model pair.

## Selection

```http
GET /models/selection
```

Returns training metadata and the selected model for each state.

## Predict

```http
POST /predict
Content-Type: application/json

{
  "states": ["California", "Texas"],
  "horizon_weeks": 8
}
```

`states` is optional. If it is omitted, forecasts for all states are returned.

## State Predict

```http
GET /predict/California?horizon_weeks=8
```

Returns the selected model forecast for one state.

## Error Behavior

- `503`: Training artifacts are missing. Run `python scripts/train.py`.
- `400`: Unknown state or requested horizon exceeds the trained artifact horizon.

