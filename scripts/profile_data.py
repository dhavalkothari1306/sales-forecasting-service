from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.forecasting.data import load_raw_sales, summarize_raw_data, to_weekly_state_sales


def main() -> None:
    settings = get_settings()
    raw = load_raw_sales(settings.data_path)
    bundle = to_weekly_state_sales(raw, freq=settings.weekly_frequency)
    summary = summarize_raw_data(raw)
    summary["weekly_rows"] = int(len(bundle.weekly))
    summary["weekly_states"] = int(bundle.weekly["state"].nunique())
    summary["weeks_per_state"] = int(bundle.weekly.groupby("state")["date"].nunique().iloc[0])
    summary["filled_missing_weeks_total"] = int(bundle.quality_report["filled_missing_weeks"].sum())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
