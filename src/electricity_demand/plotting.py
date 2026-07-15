from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from electricity_demand.config import FIGURES_DIR


def plot_benchmark_forecasts(
    training_series: pd.Series,
    forecasts: pd.DataFrame,
) -> Path:
    """
    Plot the training tail, actual test values and benchmark forecasts.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        FIGURES_DIR / "benchmark_forecast_comparison.png"
    )

    figure, axis = plt.subplots(figsize=(13, 6))

    training_tail = training_series.iloc[-104:]

    axis.plot(
        training_tail.index,
        training_tail,
        label="Training data",
        linewidth=1.5,
    )

    axis.plot(
        forecasts.index,
        forecasts["actual"],
        label="Actual test data",
        linewidth=2.2,
    )

    for column in [
        "mean",
        "naive",
        "seasonal_naive",
        "drift",
    ]:
        axis.plot(
            forecasts.index,
            forecasts[column],
            label=column.replace("_", " ").title(),
            linewidth=1.3,
        )

    axis.axvline(
        forecasts.index[0],
        linestyle="--",
        linewidth=1,
        label="Forecast origin",
    )

    axis.set_title(
        "Benchmark Forecasts for Weekly German Electricity Demand"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Electricity demand (GW)")
    axis.legend()
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path


def plot_benchmark_errors(
    metrics: pd.DataFrame,
) -> Path:
    """
    Plot benchmark RMSE values.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        FIGURES_DIR / "benchmark_rmse_comparison.png"
    )

    sorted_metrics = metrics.sort_values(
        "RMSE",
        ascending=True,
    )

    figure, axis = plt.subplots(figsize=(9, 5))

    axis.barh(
        sorted_metrics["model"],
        sorted_metrics["RMSE"],
    )

    axis.set_title("Benchmark Model RMSE Comparison")
    axis.set_xlabel("RMSE (GW)")
    axis.set_ylabel("Model")
    axis.grid(axis="x", alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path