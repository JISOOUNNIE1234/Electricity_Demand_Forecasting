from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from electricity_demand.config import FIGURES_DIR

import numpy as np
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf


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

def plot_sarima_forecast(
    training_series: pd.Series,
    forecast_data: pd.DataFrame,
) -> Path:
    """
    Plot the SARIMA forecast against the actual test data.
    """
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURES_DIR / "sarima_forecast.png"
    )

    figure, axis = plt.subplots(
        figsize=(13, 6)
    )

    training_tail = training_series.iloc[-104:]

    axis.plot(
        training_tail.index,
        training_tail,
        label="Training data",
        linewidth=1.5,
    )

    axis.plot(
        forecast_data.index,
        forecast_data["actual"],
        label="Actual test data",
        linewidth=2,
    )

    axis.plot(
        forecast_data.index,
        forecast_data["sarima"],
        label="SARIMA forecast",
        linewidth=1.8,
    )

    axis.fill_between(
        forecast_data.index,
        forecast_data["lower_95"],
        forecast_data["upper_95"],
        alpha=0.2,
        label="95% confidence interval",
    )

    axis.axvline(
        forecast_data.index[0],
        linestyle="--",
        linewidth=1,
        label="Forecast origin",
    )

    axis.set_title(
        "SARIMA Forecast for Weekly German Electricity Demand"
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


def plot_sarima_residual_diagnostics(
    residuals: pd.Series,
) -> Path:
    """
    Plot SARIMA residual time series, histogram, Q-Q plot and ACF.
    """
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURES_DIR
        / "sarima_residual_diagnostics.png"
    )

    clean_residuals = pd.Series(
        residuals
    ).dropna()

    clean_residuals = clean_residuals.loc[
        np.isfinite(clean_residuals)
    ]

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(13, 9),
    )

    axes[0, 0].plot(
        clean_residuals.index,
        clean_residuals,
    )
    axes[0, 0].axhline(
        0,
        linestyle="--",
        linewidth=1,
    )
    axes[0, 0].set_title("Residuals over time")
    axes[0, 0].set_xlabel("Observation")
    axes[0, 0].set_ylabel("Residual")

    axes[0, 1].hist(
        clean_residuals,
        bins=25,
        edgecolor="black",
    )
    axes[0, 1].set_title(
        "Residual distribution"
    )
    axes[0, 1].set_xlabel("Residual")
    axes[0, 1].set_ylabel("Frequency")

    stats.probplot(
        clean_residuals,
        dist="norm",
        plot=axes[1, 0],
    )
    axes[1, 0].set_title(
        "Normal Q-Q plot"
    )

    maximum_lags = min(
        60,
        len(clean_residuals) - 1,
    )

    plot_acf(
        clean_residuals,
        lags=maximum_lags,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title(
        "Residual autocorrelation"
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path

def plot_sarimax_forecast(
    training_series: pd.Series,
    forecast_data: pd.DataFrame,
) -> Path:
    """Plot SARIMAX conditional forecast."""
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURES_DIR / "sarimax_forecast.png"
    )

    figure, axis = plt.subplots(
        figsize=(13, 6)
    )

    axis.plot(
        training_series.iloc[-104:].index,
        training_series.iloc[-104:],
        label="Training data",
    )

    axis.plot(
        forecast_data.index,
        forecast_data["actual"],
        label="Actual test data",
        linewidth=2,
    )

    axis.plot(
        forecast_data.index,
        forecast_data["sarimax"],
        label="SARIMAX conditional forecast",
        linewidth=1.8,
    )

    axis.fill_between(
        forecast_data.index,
        forecast_data["lower_95"],
        forecast_data["upper_95"],
        alpha=0.2,
        label="95% confidence interval",
    )

    axis.axvline(
        forecast_data.index[0],
        linestyle="--",
        label="Forecast origin",
    )

    axis.set_title(
        "SARIMAX Conditional Forecast with Berlin Temperature"
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

def plot_feature_model_forecasts(
    training_series: pd.Series,
    forecast_data: pd.DataFrame,
) -> Path:
    """
    Plot feature-based model forecasts.
    """
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURES_DIR
        / "feature_model_forecasts.png"
    )

    figure, axis = plt.subplots(
        figsize=(13, 6)
    )

    axis.plot(
        training_series.iloc[-104:].index,
        training_series.iloc[-104:],
        label="Training data",
    )

    axis.plot(
        forecast_data.index,
        forecast_data["actual"],
        label="Actual test data",
        linewidth=2.2,
    )

    axis.plot(
        forecast_data.index,
        forecast_data["random_forest"],
        label="Random Forest",
    )

    axis.plot(
        forecast_data.index,
        forecast_data["gradient_boosting"],
        label="Gradient Boosting",
    )

    axis.axvline(
        forecast_data.index[0],
        linestyle="--",
        label="Forecast origin",
    )

    axis.set_title(
        "Feature-Based Forecasts for Weekly Electricity Demand"
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


def plot_feature_importance(
    importance_data: pd.DataFrame,
    top_number: int = 15,
) -> Path:
    """
    Plot the highest-ranking feature importances.
    """
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURES_DIR / "feature_importance.png"
    )

    top_features = (
        importance_data
        .sort_values(
            "importance",
            ascending=False,
        )
        .head(top_number)
        .sort_values("importance")
    )

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    axis.barh(
        top_features["feature"],
        top_features["importance"],
    )

    axis.set_title(
        f"Top {top_number} Random Forest Feature Importances"
    )
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    axis.grid(axis="x", alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path