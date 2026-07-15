import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.benchmarks import (
    generate_benchmark_forecasts,
    split_train_test,
)
from electricity_demand.config import (
    BENCHMARK_FORECAST_FILE,
    BENCHMARK_METRICS_FILE,
    FORECASTS_DIR,
    METRICS_DIR,
    TEST_HORIZON_WEEKS,
    WEEKLY_PROCESSED_FILE,
    WEEKLY_SEASONAL_PERIOD,
)
from electricity_demand.evaluation import (
    evaluate_multiple_forecasts,
)
from electricity_demand.plotting import (
    plot_benchmark_errors,
    plot_benchmark_forecasts,
)


def load_weekly_data() -> pd.DataFrame:
    """Load the processed weekly electricity-demand data."""
    if not WEEKLY_PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"Weekly dataset not found: "
            f"{WEEKLY_PROCESSED_FILE}. "
            "Run: python scripts/make_features.py"
        )

    return pd.read_csv(
        WEEKLY_PROCESSED_FILE,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )


def main() -> None:
    """Generate and evaluate the benchmark forecasts."""
    weekly_data = load_weekly_data()

    train, test = split_train_test(
        weekly_data["load_gw"],
        test_size=TEST_HORIZON_WEEKS,
    )

    forecasts = generate_benchmark_forecasts(
        training_series=train,
        test_series=test,
        seasonal_period=WEEKLY_SEASONAL_PERIOD,
    )

    metrics = evaluate_multiple_forecasts(
        forecasts=forecasts,
        training_series=train,
        seasonal_period=WEEKLY_SEASONAL_PERIOD,
    )

    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    forecasts.to_csv(BENCHMARK_FORECAST_FILE)
    metrics.to_csv(
        BENCHMARK_METRICS_FILE,
        index=False,
    )

    forecast_figure = plot_benchmark_forecasts(
        training_series=train,
        forecasts=forecasts,
    )

    error_figure = plot_benchmark_errors(
        metrics=metrics,
    )

    print("\nBenchmark analysis completed.")
    print("-----------------------------")
    print(f"Training observations: {len(train)}")
    print(f"Test observations:     {len(test)}")
    print(f"Training period: {train.index.min()} to {train.index.max()}")
    print(f"Test period:     {test.index.min()} to {test.index.max()}")

    print("\nBenchmark metrics:")
    print(metrics.to_string(index=False))

    print(f"\nForecasts saved to: {BENCHMARK_FORECAST_FILE}")
    print(f"Metrics saved to:   {BENCHMARK_METRICS_FILE}")
    print(f"Forecast plot:      {forecast_figure}")
    print(f"RMSE plot:          {error_figure}")


if __name__ == "__main__":
    main()