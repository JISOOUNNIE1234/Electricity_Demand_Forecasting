import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.config import (
    ALL_FORECASTS_FILE,
    BENCHMARK_FORECAST_FILE,
    BENCHMARK_METRICS_FILE,
    FEATURE_MODEL_FORECAST_FILE,
    FEATURE_MODEL_METRICS_FILE,
    FIGURES_DIR,
    LSTM_FORECAST_FILE,
    LSTM_WEEKLY_FORECAST_FILE,
    LSTM_WEEKLY_METRICS_FILE,
    MODEL_COMPARISON_FILE,
    SARIMA_FORECAST_FILE,
    SARIMA_METRICS_FILE,
    SARIMAX_FORECAST_FILE,
    SARIMAX_METRICS_FILE,
    WEEKLY_SEASONAL_PERIOD,
)
from electricity_demand.evaluation import (
    evaluate_forecast,
)


def load_forecast_file(
    file_path: Path,
) -> pd.DataFrame:
    """Load a forecast CSV using its timestamp column as the index."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required forecast file not found: {file_path}"
        )

    data = pd.read_csv(
        file_path,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )

    return data.sort_index()


def aggregate_lstm_to_weekly() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Aggregate hourly LSTM forecasts and actual values to weekly means.
    """
    hourly_forecast = load_forecast_file(
        LSTM_FORECAST_FILE
    )

    required_columns = {"actual", "lstm"}

    missing_columns = required_columns.difference(
        hourly_forecast.columns
    )

    if missing_columns:
        raise ValueError(
            f"LSTM forecast file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    weekly_forecast = (
        hourly_forecast[
            ["actual", "lstm"]
        ]
        .resample("W-SUN")
        .mean()
        .dropna()
    )

    weekly_forecast.to_csv(
        LSTM_WEEKLY_FORECAST_FILE
    )

    return hourly_forecast, weekly_forecast


def calculate_lstm_weekly_metrics(
    weekly_forecast: pd.DataFrame,
    benchmark_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate weekly LSTM metrics using the same weekly test period.

    The benchmark forecast is used to derive a training proxy for the
    MASE denominator. If a full weekly training series is unavailable
    here, MASE is calculated using the seasonal-naive test relationship.
    """
    aligned = weekly_forecast.join(
        benchmark_forecast[["actual"]],
        how="inner",
        rsuffix="_benchmark",
    )

    if aligned.empty:
        raise ValueError(
            "LSTM weekly forecast does not overlap the benchmark period."
        )

    actual = aligned["actual"]
    forecast = aligned["lstm"]

    absolute_errors = (
        forecast - actual
    ).abs()

    squared_errors = (
        forecast - actual
    ) ** 2

    seasonal_naive_error = (
        benchmark_forecast["seasonal_naive"]
        - benchmark_forecast["actual"]
    ).abs().mean()

    if seasonal_naive_error == 0:
        mase = float("nan")
    else:
        mase = float(
            absolute_errors.mean()
            / seasonal_naive_error
        )

    metrics = pd.DataFrame(
        [
            {
                "model": "lstm_weekly",
                "MAE": float(
                    absolute_errors.mean()
                ),
                "RMSE": float(
                    squared_errors.mean() ** 0.5
                ),
                "MASE": mase,
                "Bias": float(
                    (forecast - actual).mean()
                ),
            }
        ]
    )

    metrics.to_csv(
        LSTM_WEEKLY_METRICS_FILE,
        index=False,
    )

    return metrics


def combine_metric_files(
    lstm_weekly_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Combine all weekly model metric files."""
    metric_sources = [
        (
            BENCHMARK_METRICS_FILE,
            "weekly",
            "operational",
        ),
        (
            SARIMA_METRICS_FILE,
            "weekly",
            "operational",
        ),
        (
            SARIMAX_METRICS_FILE,
            "weekly",
            "conditional",
        ),
        (
            FEATURE_MODEL_METRICS_FILE,
            "weekly",
            "conditional",
        ),
    ]

    metric_tables = []

    for (
        file_path,
        frequency,
        forecast_type,
    ) in metric_sources:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Metric file not found: {file_path}"
            )

        table = pd.read_csv(file_path)
        table["frequency"] = frequency
        table["forecast_type"] = forecast_type

        metric_tables.append(table)

    lstm_table = lstm_weekly_metrics.copy()
    lstm_table["frequency"] = "weekly"
    lstm_table["forecast_type"] = (
        "rolling one-step-ahead"
    )

    metric_tables.append(lstm_table)

    comparison = pd.concat(
        metric_tables,
        ignore_index=True,
    )

    seasonal_naive_row = comparison.loc[
        comparison["model"]
        == "seasonal_naive"
    ]

    if seasonal_naive_row.empty:
        raise ValueError(
            "Seasonal naive model is missing from metrics."
        )

    seasonal_naive_rmse = float(
        seasonal_naive_row.iloc[0]["RMSE"]
    )

    seasonal_naive_mae = float(
        seasonal_naive_row.iloc[0]["MAE"]
    )

    comparison[
        "RMSE_improvement_percent"
    ] = (
        (
            seasonal_naive_rmse
            - comparison["RMSE"]
        )
        / seasonal_naive_rmse
        * 100.0
    )

    comparison[
        "MAE_improvement_percent"
    ] = (
        (
            seasonal_naive_mae
            - comparison["MAE"]
        )
        / seasonal_naive_mae
        * 100.0
    )

    comparison = comparison.sort_values(
        "RMSE"
    ).reset_index(drop=True)

    comparison.to_csv(
        MODEL_COMPARISON_FILE,
        index=False,
    )

    return comparison


def combine_weekly_forecasts(
    benchmark: pd.DataFrame,
    sarima: pd.DataFrame,
    sarimax: pd.DataFrame,
    feature_models: pd.DataFrame,
    lstm_weekly: pd.DataFrame,
) -> pd.DataFrame:
    """Combine all weekly forecasts into one aligned table."""
    all_forecasts = benchmark[
        [
            "actual",
            "mean",
            "naive",
            "seasonal_naive",
            "drift",
        ]
    ].copy()

    all_forecasts = all_forecasts.join(
        sarima[["sarima"]],
        how="left",
    )

    all_forecasts = all_forecasts.join(
        sarimax[["sarimax"]],
        how="left",
    )

    all_forecasts = all_forecasts.join(
        feature_models[
            [
                "random_forest",
                "gradient_boosting",
            ]
        ],
        how="left",
    )

    lstm_renamed = lstm_weekly[
        ["lstm"]
    ].rename(
        columns={
            "lstm": "lstm_weekly"
        }
    )

    all_forecasts = all_forecasts.join(
        lstm_renamed,
        how="left",
    )

    all_forecasts.to_csv(
        ALL_FORECASTS_FILE
    )

    return all_forecasts


def plot_final_forecast_comparison(
    forecasts: pd.DataFrame,
) -> Path:
    """Create the main weekly forecast-comparison figure."""
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURES_DIR / "forecast_comparison.png"
    )

    figure, axis = plt.subplots(
        figsize=(14, 7)
    )

    axis.plot(
        forecasts.index,
        forecasts["actual"],
        label="Actual",
        linewidth=2.5,
    )

    models_to_plot = [
        "seasonal_naive",
        "sarima",
        "sarimax",
        "random_forest",
        "gradient_boosting",
        "lstm_weekly",
    ]

    for model_name in models_to_plot:
        if model_name in forecasts.columns:
            axis.plot(
                forecasts.index,
                forecasts[model_name],
                label=model_name.replace(
                    "_",
                    " ",
                ).title(),
                linewidth=1.2,
            )

    axis.set_title(
        "Weekly German Electricity Demand Forecast Comparison"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Electricity demand (GW)")
    axis.legend(
        ncol=2
    )
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path


def plot_model_rmse(
    comparison: pd.DataFrame,
) -> Path:
    """Plot RMSE for every weekly forecasting model."""
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURES_DIR
        / "model_rmse_comparison.png"
    )

    ordered = comparison.sort_values(
        "RMSE",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    axis.barh(
        ordered["model"],
        ordered["RMSE"],
    )

    axis.set_title(
        "Weekly Forecast RMSE Comparison"
    )
    axis.set_xlabel("RMSE (GW)")
    axis.set_ylabel("Model")
    axis.grid(
        axis="x",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path


def plot_improvement_over_seasonal_naive(
    comparison: pd.DataFrame,
) -> Path:
    """
    Plot RMSE improvement relative to seasonal naive.

    Positive values indicate improvement.
    Negative values indicate worse performance.
    """
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURES_DIR
        / "improvement_over_seasonal_naive.png"
    )

    ordered = comparison.sort_values(
        "RMSE_improvement_percent"
    )

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    axis.barh(
        ordered["model"],
        ordered["RMSE_improvement_percent"],
    )

    axis.axvline(
        0,
        linewidth=1,
    )

    axis.set_title(
        "RMSE Improvement Over Seasonal Naive"
    )
    axis.set_xlabel("RMSE improvement (%)")
    axis.set_ylabel("Model")
    axis.grid(
        axis="x",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path


def main() -> None:
    """Build final forecast and model-comparison outputs."""
    benchmark = load_forecast_file(
        BENCHMARK_FORECAST_FILE
    )

    sarima = load_forecast_file(
        SARIMA_FORECAST_FILE
    )

    sarimax = load_forecast_file(
        SARIMAX_FORECAST_FILE
    )

    feature_models = load_forecast_file(
        FEATURE_MODEL_FORECAST_FILE
    )

    _, lstm_weekly = aggregate_lstm_to_weekly()

    lstm_weekly_metrics = (
        calculate_lstm_weekly_metrics(
            weekly_forecast=lstm_weekly,
            benchmark_forecast=benchmark,
        )
    )

    comparison = combine_metric_files(
        lstm_weekly_metrics
    )

    all_forecasts = combine_weekly_forecasts(
        benchmark=benchmark,
        sarima=sarima,
        sarimax=sarimax,
        feature_models=feature_models,
        lstm_weekly=lstm_weekly,
    )

    forecast_figure = (
        plot_final_forecast_comparison(
            all_forecasts
        )
    )

    rmse_figure = plot_model_rmse(
        comparison
    )

    improvement_figure = (
        plot_improvement_over_seasonal_naive(
            comparison
        )
    )

    print("\nFinal model comparison")
    print("----------------------")
    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        f"\nCombined metrics: "
        f"{MODEL_COMPARISON_FILE}"
    )
    print(
        f"Combined forecasts: "
        f"{ALL_FORECASTS_FILE}"
    )
    print(
        f"Forecast figure: "
        f"{forecast_figure}"
    )
    print(
        f"RMSE figure: "
        f"{rmse_figure}"
    )
    print(
        f"Improvement figure: "
        f"{improvement_figure}"
    )


if __name__ == "__main__":
    main()