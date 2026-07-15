import sys
from pathlib import Path

import pandas as pd

from electricity_demand.features import (
    add_rolling_features,
    add_target_lag_features,
    create_feature_dataset,
)


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

def test_target_lag_uses_only_previous_values() -> None:
    index = pd.date_range(
        start="2015-01-04",
        periods=5,
        freq="W-SUN",
    )

    data = pd.DataFrame(
        {
            "load_gw": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
            ]
        },
        index=index,
    )

    result = add_target_lag_features(
        data,
        lags=[1],
    )

    assert pd.isna(
        result["load_lag_1"].iloc[0]
    )

    assert (
        result["load_lag_1"].iloc[1]
        == 10.0
    )

    assert (
        result["load_lag_1"].iloc[4]
        == 40.0
    )


def test_rolling_mean_excludes_current_target() -> None:
    index = pd.date_range(
        start="2015-01-04",
        periods=5,
        freq="W-SUN",
    )

    data = pd.DataFrame(
        {
            "load_gw": [
                10.0,
                20.0,
                30.0,
                40.0,
                1000.0,
            ]
        },
        index=index,
    )

    result = add_rolling_features(
        data,
        windows=[2],
    )

    assert (
        result["load_rolling_mean_2"].iloc[4]
        == 35.0
    )


def test_feature_dataset_contains_no_missing_values() -> None:
    index = pd.date_range(
        start="2015-01-04",
        periods=120,
        freq="W-SUN",
        tz="UTC",
    )

    data = pd.DataFrame(
        {
            "load_gw": range(120),
            "temp_mean": 10.0,
            "temp_min": 5.0,
            "temp_max": 15.0,
            "heating_degree_days": 20.0,
            "cooling_degree_days": 0.0,
        },
        index=index,
    )

    result = create_feature_dataset(data)

    assert not result.empty
    assert result.isna().sum().sum() == 0