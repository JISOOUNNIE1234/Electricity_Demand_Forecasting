import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_mae(
    actual: pd.Series,
    forecast: pd.Series,
) -> float:
    """Calculate mean absolute error."""
    return float(mean_absolute_error(actual, forecast))


def calculate_rmse(
    actual: pd.Series,
    forecast: pd.Series,
) -> float:
    """Calculate root mean squared error."""
    return float(
        np.sqrt(mean_squared_error(actual, forecast))
    )


def calculate_bias(
    actual: pd.Series,
    forecast: pd.Series,
) -> float:
    """
    Calculate mean forecast error.

    Positive values indicate average over-forecasting.
    Negative values indicate average under-forecasting.
    """
    return float(np.mean(forecast - actual))


def calculate_mase(
    actual: pd.Series,
    forecast: pd.Series,
    training_series: pd.Series,
    seasonal_period: int = 52,
) -> float:
    """
    Calculate mean absolute scaled error.

    The scaling denominator is the in-sample seasonal-naive error.
    A MASE below 1 means the model outperforms the seasonal-naive
    scaling benchmark on average.
    """
    actual_values = np.asarray(actual, dtype=float)
    forecast_values = np.asarray(forecast, dtype=float)
    training_values = np.asarray(training_series, dtype=float)

    if len(training_values) <= seasonal_period:
        raise ValueError(
            "Training series must contain more observations than "
            "the seasonal period."
        )

    seasonal_errors = np.abs(
        training_values[seasonal_period:]
        - training_values[:-seasonal_period]
    )

    scale = float(np.mean(seasonal_errors))

    if np.isclose(scale, 0.0):
        raise ValueError(
            "Cannot calculate MASE because the scaling error is zero."
        )

    model_mae = float(
        np.mean(np.abs(actual_values - forecast_values))
    )

    return model_mae / scale


def evaluate_forecast(
    actual: pd.Series,
    forecast: pd.Series,
    training_series: pd.Series,
    model_name: str,
    seasonal_period: int = 52,
) -> dict[str, float | str]:
    """
    Evaluate one forecast using common forecasting metrics.
    """
    if len(actual) != len(forecast):
        raise ValueError(
            "Actual and forecast series must have equal lengths."
        )

    actual_aligned = actual.astype(float)
    forecast_aligned = forecast.astype(float)

    return {
        "model": model_name,
        "MAE": calculate_mae(
            actual_aligned,
            forecast_aligned,
        ),
        "RMSE": calculate_rmse(
            actual_aligned,
            forecast_aligned,
        ),
        "MASE": calculate_mase(
            actual=actual_aligned,
            forecast=forecast_aligned,
            training_series=training_series,
            seasonal_period=seasonal_period,
        ),
        "Bias": calculate_bias(
            actual_aligned,
            forecast_aligned,
        ),
    }


def evaluate_multiple_forecasts(
    forecasts: pd.DataFrame,
    training_series: pd.Series,
    actual_column: str = "actual",
    seasonal_period: int = 52,
) -> pd.DataFrame:
    """
    Evaluate every forecast column against the actual series.
    """
    if actual_column not in forecasts.columns:
        raise ValueError(
            f"Missing actual column: {actual_column}"
        )

    actual = forecasts[actual_column]

    results = []

    for column in forecasts.columns:
        if column == actual_column:
            continue

        result = evaluate_forecast(
            actual=actual,
            forecast=forecasts[column],
            training_series=training_series,
            model_name=column,
            seasonal_period=seasonal_period,
        )

        results.append(result)

    metrics = pd.DataFrame(results)

    return metrics.sort_values("RMSE").reset_index(drop=True)