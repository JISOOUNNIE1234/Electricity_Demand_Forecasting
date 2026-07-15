import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.benchmarks import (
    drift_forecast,
    generate_benchmark_forecasts,
    mean_forecast,
    naive_forecast,
    seasonal_naive_forecast,
    split_train_test,
)


def create_test_series(
    periods: int = 200,
) -> pd.Series:
    index = pd.date_range(
        start="2015-01-04",
        periods=periods,
        freq="W-SUN",
    )

    values = np.arange(
        periods,
        dtype=float,
    )

    return pd.Series(
        values,
        index=index,
        name="load_gw",
    )


def test_split_train_test_uses_final_observations() -> None:
    series = create_test_series(200)

    train, test = split_train_test(
        series,
        test_size=104,
    )

    assert len(train) == 96
    assert len(test) == 104
    assert train.index.max() < test.index.min()


def test_mean_forecast_has_correct_length() -> None:
    series = create_test_series(100)
    forecast_index = pd.date_range(
        start="2017-01-01",
        periods=10,
        freq="W-SUN",
    )

    forecast = mean_forecast(
        series,
        forecast_index,
    )

    assert len(forecast) == 10
    assert forecast.nunique() == 1


def test_naive_forecast_repeats_last_value() -> None:
    series = create_test_series(100)
    forecast_index = pd.date_range(
        start="2017-01-01",
        periods=10,
        freq="W-SUN",
    )

    forecast = naive_forecast(
        series,
        forecast_index,
    )

    assert np.allclose(
        forecast.to_numpy(),
        series.iloc[-1],
    )


def test_seasonal_naive_repeats_recent_cycle() -> None:
    series = create_test_series(104)
    forecast_index = pd.date_range(
        start="2017-01-01",
        periods=104,
        freq="W-SUN",
    )

    forecast = seasonal_naive_forecast(
        series,
        forecast_index,
        seasonal_period=52,
    )

    expected = np.tile(
        series.iloc[-52:].to_numpy(),
        2,
    )

    assert np.allclose(
        forecast.to_numpy(),
        expected,
    )


def test_drift_forecast_continues_linear_trend() -> None:
    series = create_test_series(100)
    forecast_index = pd.date_range(
        start="2017-01-01",
        periods=3,
        freq="W-SUN",
    )

    forecast = drift_forecast(
        series,
        forecast_index,
    )

    assert np.allclose(
        forecast.to_numpy(),
        [100.0, 101.0, 102.0],
    )


def test_all_benchmark_forecasts_match_test_length() -> None:
    series = create_test_series(220)

    train, test = split_train_test(
        series,
        test_size=104,
    )

    forecasts = generate_benchmark_forecasts(
        training_series=train,
        test_series=test,
        seasonal_period=52,
    )

    assert len(forecasts) == 104
    assert forecasts.isna().sum().sum() == 0