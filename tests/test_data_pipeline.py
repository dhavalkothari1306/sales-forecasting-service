from __future__ import annotations

import pandas as pd

from app.forecasting.data import to_weekly_state_sales


def test_weekly_resampling_fills_missing_weeks() -> None:
    raw = pd.DataFrame(
        {
            "state": ["A", "A", "A"],
            "date": pd.to_datetime(["2024-01-07", "2024-01-21", "2024-01-28"]),
            "sales": [100.0, 300.0, 500.0],
        }
    )

    bundle = to_weekly_state_sales(raw, freq="W-SUN")

    assert len(bundle.weekly) == 4
    assert bundle.quality_report.loc[0, "filled_missing_weeks"] == 1
    filled_week = bundle.weekly.loc[bundle.weekly["date"] == pd.Timestamp("2024-01-14")]
    assert filled_week["was_missing"].iloc[0]
    assert filled_week["y"].iloc[0] == 200.0

