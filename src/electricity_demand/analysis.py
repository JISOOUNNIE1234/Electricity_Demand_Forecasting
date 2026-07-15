from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from electricity_demand.config import FIGURES_DIR, METRICS_DIR


def calculate_summary_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate descriptive statistics for the electricity demand series.

    Parameters
    ----------
    data:
        DataFrame containing a load_gw column.

    Returns
    -------
    pandas.DataFrame
        Summary statistics table.
    """
    if "load_gw" not in data.columns:
        raise ValueError("Expected a 'load_gw' column.")

    summary = data["load_gw"].describe().to_frame(name="load_gw")

    summary.loc["missing"] = data["load_gw"].isna().sum()
    summary.loc["start_date"] = str(data.index.min())
    summary.loc["end_date"] = str(data.index.max())

    return summary


def run_adf_test(series: pd.Series, series_name: str) -> dict[str, float | str]:
    """
    Run the Augmented Dickey-Fuller stationarity test.

    Parameters
    ----------
    series:
        Time series to test.
    series_name:
        Descriptive name used in the saved output.

    Returns
    -------
    dict
        ADF statistic, p-value, lag count and interpretation.
    """
    clean_series = series.dropna()

    result = adfuller(clean_series, autolag="AIC")

    adf_statistic = float(result[0])
    p_value = float(result[1])
    used_lags = int(result[2])
    observations = int(result[3])

    interpretation = (
        "Stationary: reject the null hypothesis of a unit root."
        if p_value < 0.05
        else "Non-stationary: fail to reject the null hypothesis of a unit root."
    )

    return {
        "series": series_name,
        "adf_statistic": adf_statistic,
        "p_value": p_value,
        "used_lags": used_lags,
        "observations": observations,
        "interpretation": interpretation,
    }


def plot_time_series(
    data: pd.DataFrame,
    title: str,
    filename: str,
) -> Path:
    """
    Plot and save an electricity demand time series.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / filename

    figure, axis = plt.subplots(figsize=(12, 5))

    axis.plot(data.index, data["load_gw"])
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Electricity demand (GW)")
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


def plot_weekly_seasonal_decomposition(
    weekly_data: pd.DataFrame,
    period: int = 52,
) -> Path:
    """
    Decompose weekly electricity demand into time-series components.

    Parameters
    ----------
    weekly_data:
        Weekly electricity demand.
    period:
        Seasonal cycle length. A value of 52 represents yearly
        seasonality in weekly observations.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / "weekly_seasonal_decomposition.png"

    decomposition = seasonal_decompose(
        weekly_data["load_gw"].dropna(),
        model="additive",
        period=period,
        extrapolate_trend="freq",
    )

    figure = decomposition.plot()
    figure.set_size_inches(12, 9)
    figure.suptitle(
        "Weekly German Electricity Demand: Seasonal Decomposition",
        y=1.02,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return output_path


def plot_acf_pacf(
    series: pd.Series,
    series_name: str,
    filename: str,
    lags: int = 60,
) -> Path:
    """
    Save separate ACF and PACF figures for a time series.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / filename
    clean_series = series.dropna()

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 8),
    )

    plot_acf(
        clean_series,
        lags=min(lags, len(clean_series) - 1),
        ax=axes[0],
    )
    axes[0].set_title(f"ACF: {series_name}")

    pacf_lags = min(lags, max(1, len(clean_series) // 2 - 1))

    plot_pacf(
        clean_series,
        lags=pacf_lags,
        method="ywm",
        ax=axes[1],
    )
    axes[1].set_title(f"PACF: {series_name}")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


def create_differenced_series(
    weekly_data: pd.DataFrame,
    seasonal_period: int = 52,
) -> pd.DataFrame:
    """
    Create first-differenced and seasonally differenced weekly series.
    """
    result = weekly_data.copy()

    result["first_difference"] = result["load_gw"].diff()
    result["seasonal_difference"] = result["load_gw"].diff(
        seasonal_period
    )
    result["first_seasonal_difference"] = (
        result["load_gw"]
        .diff()
        .diff(seasonal_period)
    )

    return result


def plot_differenced_series(
    differenced_data: pd.DataFrame,
) -> Path:
    """
    Plot original, first-differenced and seasonally differenced series.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / "weekly_differencing_comparison.png"

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 10),
        sharex=True,
    )

    axes[0].plot(
        differenced_data.index,
        differenced_data["load_gw"],
    )
    axes[0].set_title("Original weekly electricity demand")
    axes[0].set_ylabel("GW")

    axes[1].plot(
        differenced_data.index,
        differenced_data["first_difference"],
    )
    axes[1].set_title("First-differenced weekly demand")
    axes[1].set_ylabel("Difference")

    axes[2].plot(
        differenced_data.index,
        differenced_data["seasonal_difference"],
    )
    axes[2].set_title("Seasonally differenced weekly demand, lag 52")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Difference")

    for axis in axes:
        axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


def run_part1_analysis(
    hourly_data: pd.DataFrame,
    daily_data: pd.DataFrame,
    weekly_data: pd.DataFrame,
) -> None:
    """
    Run Part 1 exploratory and stationarity analysis.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    hourly_summary = calculate_summary_statistics(hourly_data)
    daily_summary = calculate_summary_statistics(daily_data)
    weekly_summary = calculate_summary_statistics(weekly_data)

    hourly_summary.to_csv(
        METRICS_DIR / "hourly_summary_statistics.csv"
    )
    daily_summary.to_csv(
        METRICS_DIR / "daily_summary_statistics.csv"
    )
    weekly_summary.to_csv(
        METRICS_DIR / "weekly_summary_statistics.csv"
    )

    plot_time_series(
        data=hourly_data,
        title="Hourly German Electricity Demand",
        filename="hourly_electricity_demand.png",
    )

    plot_time_series(
        data=daily_data,
        title="Daily Average German Electricity Demand",
        filename="daily_electricity_demand.png",
    )

    plot_time_series(
        data=weekly_data,
        title="Weekly Average German Electricity Demand",
        filename="weekly_electricity_demand.png",
    )

    plot_weekly_seasonal_decomposition(
        weekly_data=weekly_data,
        period=52,
    )

    plot_acf_pacf(
        series=weekly_data["load_gw"],
        series_name="Original weekly electricity demand",
        filename="weekly_original_acf_pacf.png",
        lags=60,
    )

    differenced_data = create_differenced_series(
        weekly_data=weekly_data,
        seasonal_period=52,
    )

    plot_differenced_series(differenced_data)

    plot_acf_pacf(
        series=differenced_data["first_difference"],
        series_name="First-differenced weekly electricity demand",
        filename="weekly_first_difference_acf_pacf.png",
        lags=60,
    )

    plot_acf_pacf(
        series=differenced_data["seasonal_difference"],
        series_name="Seasonally differenced weekly electricity demand",
        filename="weekly_seasonal_difference_acf_pacf.png",
        lags=60,
    )

    adf_results = [
        run_adf_test(
            weekly_data["load_gw"],
            "Original weekly series",
        ),
        run_adf_test(
            differenced_data["first_difference"],
            "First-differenced weekly series",
        ),
        run_adf_test(
            differenced_data["seasonal_difference"],
            "Seasonally differenced weekly series",
        ),
        run_adf_test(
            differenced_data["first_seasonal_difference"],
            "First and seasonally differenced weekly series",
        ),
    ]

    adf_results_table = pd.DataFrame(adf_results)

    adf_results_table.to_csv(
        METRICS_DIR / "adf_stationarity_results.csv",
        index=False,
    )

    differenced_data.to_csv(
        METRICS_DIR / "weekly_differenced_series.csv"
    )

    print("\nPart 1 analysis completed.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Statistics saved to: {METRICS_DIR}")
    print("\nADF results:")
    print(adf_results_table.to_string(index=False))