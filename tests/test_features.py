from __future__ import annotations

import pandas as pd

from app.forecasting.features import build_future_feature_row, create_supervised_frame


def test_lag_features_use_prior_targets_only() -> None:
    series = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-07", periods=40, freq="W-SUN"),
            "y": [float(value) for value in range(40)],
        }
    )

    supervised = create_supervised_frame(series)
    first = supervised.iloc[0]

    assert first["y"] == 30.0
    assert first["lag_1"] == 29.0
    assert first["lag_7"] == 23.0
    assert first["lag_30"] == 0.0
    assert first["rolling_mean_4"] == (26.0 + 27.0 + 28.0 + 29.0) / 4.0


def test_future_feature_row_uses_history_tail() -> None:
    history = [float(value) for value in range(1, 41)]
    row = build_future_feature_row(history, pd.Timestamp("2024-10-13"))

    assert row["lag_1"].iloc[0] == 40.0
    assert row["lag_7"].iloc[0] == 34.0
    assert row["lag_30"].iloc[0] == 11.0
    assert row["rolling_mean_4"].iloc[0] == (37.0 + 38.0 + 39.0 + 40.0) / 4.0

