import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.preprocessing import (
    aggregate_electricity_load,
    clean_hourly_load,
)


def test_clean_hourly_load_converts_mw_to_gw() -> None:
    input_data = pd.DataFrame(
        {
            "timestamp": [
                "2015-01-01 00:00:00+00:00",
                "2015-01-01 01:00:00+00:00",
            ],
            "load_mw": [50_000.0, 52_000.0],
        }
    )

    result = clean_hourly_load(input_data)

    assert result["load_gw"].tolist() == [50.0, 52.0]


def test_aggregation_creates_daily_and_weekly_data() -> None:
    timestamp_index = pd.date_range(
        start="2015-01-01",
        periods=24 * 14,
        freq="h",
        tz="UTC",
    )

    hourly_data = pd.DataFrame(
        {"load_gw": 50.0},
        index=timestamp_index,
    )

    daily_data, weekly_data = aggregate_electricity_load(hourly_data)

    assert len(daily_data) == 14
    assert len(weekly_data) >= 2
    assert daily_data["load_gw"].isna().sum() == 0
    assert weekly_data["load_gw"].isna().sum() == 0