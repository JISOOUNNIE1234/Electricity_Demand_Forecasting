import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.evaluation import (
    calculate_bias,
    calculate_mae,
    calculate_mase,
    calculate_rmse,
)


def test_perfect_forecast_has_zero_errors() -> None:
    actual = pd.Series([1.0, 2.0, 3.0])
    forecast = pd.Series([1.0, 2.0, 3.0])

    assert calculate_mae(actual, forecast) == 0.0
    assert calculate_rmse(actual, forecast) == 0.0
    assert calculate_bias(actual, forecast) == 0.0


def test_bias_is_positive_for_over_forecast() -> None:
    actual = pd.Series([1.0, 2.0, 3.0])
    forecast = pd.Series([2.0, 3.0, 4.0])

    assert calculate_bias(actual, forecast) == 1.0


def test_perfect_forecast_has_zero_mase() -> None:
    training = pd.Series(
        np.arange(120, dtype=float)
    )
    actual = pd.Series([120.0, 121.0, 122.0])
    forecast = actual.copy()

    mase = calculate_mase(
        actual=actual,
        forecast=forecast,
        training_series=training,
        seasonal_period=52,
    )

    assert mase == 0.0