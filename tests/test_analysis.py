import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.analysis import (
    calculate_summary_statistics,
    create_differenced_series,
    run_adf_test,
)


def test_summary_statistics_include_missing_count() -> None:
    index = pd.date_range(
        start="2015-01-04",
        periods=5,
        freq="W-SUN",
    )

    data = pd.DataFrame(
        {"load_gw": [50.0, 51.0, np.nan, 49.0, 52.0]},
        index=index,
    )

    summary = calculate_summary_statistics(data)

    assert float(summary.loc["missing", "load_gw"]) == 1.0


def test_create_differenced_series_adds_expected_columns() -> None:
    index = pd.date_range(
        start="2015-01-04",
        periods=60,
        freq="W-SUN",
    )

    data = pd.DataFrame(
        {"load_gw": np.arange(60, dtype=float)},
        index=index,
    )

    result = create_differenced_series(data)

    expected_columns = {
        "load_gw",
        "first_difference",
        "seasonal_difference",
        "first_seasonal_difference",
    }

    assert expected_columns.issubset(result.columns)


def test_adf_result_contains_required_fields() -> None:
    rng = np.random.default_rng(42)

    stationary_series = pd.Series(
        rng.normal(size=200)
    )

    result = run_adf_test(
        stationary_series,
        "Synthetic stationary series",
    )

    required_fields = {
        "series",
        "adf_statistic",
        "p_value",
        "used_lags",
        "observations",
        "interpretation",
    }

    assert required_fields.issubset(result.keys())
    assert 0.0 <= result["p_value"] <= 1.0