from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


LAG_FEATURES: tuple[int, ...] = (1, 7, 30)
ROLLING_WINDOWS: tuple[int, ...] = (4, 8, 12)


def _holiday_flags_for_weeks(dates: pd.Series) -> pd.Series:
    normalized = pd.to_datetime(dates).dt.normalize()
    if normalized.empty:
        return pd.Series(dtype=int)

    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(
        start=normalized.min() - pd.Timedelta(days=7),
        end=normalized.max() + pd.Timedelta(days=1),
    )
    holiday_values = holidays.normalize().to_numpy(dtype="datetime64[D]")

    flags: list[int] = []
    for date_value in normalized:
        start = (date_value - pd.Timedelta(days=6)).to_datetime64().astype("datetime64[D]")
        end = date_value.to_datetime64().astype("datetime64[D]")
        flags.append(int(((holiday_values >= start) & (holiday_values <= end)).any()))
    return pd.Series(flags, index=dates.index, dtype=int)


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    enriched = df.copy()
    dates = pd.to_datetime(enriched[date_col])
    iso_calendar = dates.dt.isocalendar()

    enriched["day_of_week"] = dates.dt.dayofweek.astype(int)
    enriched["week_of_year"] = iso_calendar.week.astype(int)
    enriched["month"] = dates.dt.month.astype(int)
    enriched["quarter"] = dates.dt.quarter.astype(int)
    enriched["year"] = dates.dt.year.astype(int)
    enriched["is_month_start"] = dates.dt.is_month_start.astype(int)
    enriched["is_month_end"] = dates.dt.is_month_end.astype(int)
    enriched["holiday_flag"] = _holiday_flags_for_weeks(dates)
    return enriched


def feature_columns() -> list[str]:
    lag_columns = [f"lag_{lag}" for lag in LAG_FEATURES]
    rolling_columns: list[str] = []
    for window in ROLLING_WINDOWS:
        rolling_columns.extend([f"rolling_mean_{window}", f"rolling_std_{window}"])
    calendar_columns = [
        "day_of_week",
        "week_of_year",
        "month",
        "quarter",
        "year",
        "is_month_start",
        "is_month_end",
        "holiday_flag",
    ]
    return lag_columns + rolling_columns + calendar_columns


def create_supervised_frame(
    series_df: pd.DataFrame,
    target_col: str = "y",
    drop_missing: bool = True,
) -> pd.DataFrame:
    """Create lag, rolling, and calendar features without using current target leakage."""
    df = series_df[["date", target_col]].copy().sort_values("date").reset_index(drop=True)
    df = df.rename(columns={target_col: "y"})

    for lag in LAG_FEATURES:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    shifted_target = df["y"].shift(1)
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = shifted_target.rolling(window=window, min_periods=1).mean()
        df[f"rolling_std_{window}"] = (
            shifted_target.rolling(window=window, min_periods=2).std(ddof=0).fillna(0.0)
        )

    df = add_calendar_features(df, date_col="date")
    if drop_missing:
        df = df.dropna(subset=[f"lag_{lag}" for lag in LAG_FEATURES]).reset_index(drop=True)
    return df


def build_future_feature_row(history: Sequence[float], future_date: pd.Timestamp) -> pd.DataFrame:
    values = pd.Series(list(history), dtype=float)
    if values.empty:
        raise ValueError("History is required to build recursive forecast features.")

    row: dict[str, float | pd.Timestamp] = {"date": pd.Timestamp(future_date)}
    for lag in LAG_FEATURES:
        row[f"lag_{lag}"] = float(values.iloc[-lag]) if len(values) >= lag else float(values.iloc[0])

    for window in ROLLING_WINDOWS:
        tail = values.tail(window)
        row[f"rolling_mean_{window}"] = float(tail.mean())
        row[f"rolling_std_{window}"] = float(tail.std(ddof=0)) if len(tail) > 1 else 0.0

    frame = pd.DataFrame([row])
    frame = add_calendar_features(frame, date_col="date")
    return frame[feature_columns()]


def make_lstm_sequences(values: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[np.ndarray] = []
    y_values: list[float] = []
    for idx in range(sequence_length, len(values)):
        x_values.append(values[idx - sequence_length : idx])
        y_values.append(values[idx])
    return np.array(x_values), np.array(y_values)

