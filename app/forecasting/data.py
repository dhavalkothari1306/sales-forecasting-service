from __future__ import annotations

import io
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"state", "date", "total"}


@dataclass(frozen=True)
class DataBundle:
    weekly: pd.DataFrame
    quality_report: pd.DataFrame


def load_raw_sales(path: str | Path) -> pd.DataFrame:
    """Load the assignment workbook and normalize the schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    raw = _read_source_table(path)
    raw.columns = [str(column).strip() for column in raw.columns]
    lookup = {column.lower().strip(): column for column in raw.columns}
    missing = REQUIRED_COLUMNS - set(lookup)
    if missing:
        raise ValueError(f"Missing required workbook columns: {sorted(missing)}")

    df = raw.rename(
        columns={
            lookup["state"]: "state",
            lookup["date"]: "date",
            lookup["total"]: "sales",
            lookup.get("category", "category"): "category",
        }
    )
    keep_columns = [column for column in ["state", "date", "sales", "category"] if column in df]
    df = df[keep_columns].copy()

    df["state"] = df["state"].astype(str).str.strip()
    df["date"] = _parse_mixed_dates(df["date"])
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["state", "date"])
    df = df[df["state"] != ""]
    df = df.sort_values(["state", "date"]).reset_index(drop=True)
    return df


def _read_source_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return _read_excel_isolated(path)
    raise ValueError(f"Unsupported data file type: {path.suffix}")


def _parse_mixed_dates(values: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    except TypeError:
        parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        fallback = values[parsed.isna()].apply(lambda value: pd.to_datetime(value, errors="coerce"))
        parsed.loc[parsed.isna()] = fallback
    return parsed


def _read_excel_isolated(path: Path) -> pd.DataFrame:
    """Read Excel in a child process so workbook handles cannot affect artifact writes."""
    code = (
        "import pandas as pd, sys\n"
        "df = pd.read_excel(sys.argv[1])\n"
        "df.to_csv(sys.stdout, index=False)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return pd.read_csv(io.StringIO(completed.stdout))


def to_weekly_state_sales(df: pd.DataFrame, freq: str = "W-SUN") -> DataBundle:
    """Aggregate irregular observations into complete weekly state-level series."""
    if df.empty:
        raise ValueError("No rows available after raw data cleaning.")

    daily = (
        df.groupby(["state", "date"], as_index=False)["sales"]
        .sum(min_count=1)
        .sort_values(["state", "date"])
    )

    pieces: list[pd.DataFrame] = []
    quality_rows: list[dict[str, object]] = []

    for state, state_df in daily.groupby("state", sort=True):
        series = state_df.set_index("date")["sales"].sort_index()
        weekly_observed = series.resample(freq).sum(min_count=1)
        missing_mask = weekly_observed.isna()
        filled = (
            weekly_observed.interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
            .fillna(0.0)
            .clip(lower=0.0)
        )

        state_weekly = pd.DataFrame(
            {
                "state": state,
                "date": filled.index,
                "y": filled.astype(float).values,
                "was_missing": missing_mask.values,
            }
        )
        pieces.append(state_weekly)
        quality_rows.append(
            {
                "state": state,
                "first_week": filled.index.min().date().isoformat(),
                "last_week": filled.index.max().date().isoformat(),
                "weeks": int(len(filled)),
                "filled_missing_weeks": int(missing_mask.sum()),
                "observed_weeks": int((~missing_mask).sum()),
            }
        )

    weekly = pd.concat(pieces, ignore_index=True)
    quality_report = pd.DataFrame(quality_rows).sort_values("state").reset_index(drop=True)
    return DataBundle(weekly=weekly, quality_report=quality_report)


def load_weekly_sales(path: str | Path, freq: str = "W-SUN") -> DataBundle:
    return to_weekly_state_sales(load_raw_sales(path), freq=freq)


def summarize_raw_data(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "states": int(df["state"].nunique()),
        "first_date": df["date"].min().date().isoformat(),
        "last_date": df["date"].max().date().isoformat(),
        "missing_sales": int(df["sales"].isna().sum()),
        "negative_sales": int((df["sales"] < 0).sum()),
        "categories": sorted(df["category"].dropna().unique().tolist()) if "category" in df else [],
    }
