import numpy as np
import pandas as pd


def split_train_test(
    series: pd.Series,
    test_size: int = 104,
) -> tuple[pd.Series, pd.Series]:
    """
    Split a time series chronologically.

    The final test_size observations are used as the test set.
    """
    clean_series = series.dropna().sort_index()

    if len(clean_series) <= test_size:
        raise ValueError(
            "The series must contain more observations than "
            "the requested test size."
        )

    train = clean_series.iloc[:-test_size].copy()
    test = clean_series.iloc[-test_size:].copy()

    return train, test


def mean_forecast(
    training_series: pd.Series,
    forecast_index: pd.Index,
) -> pd.Series:
    """Forecast every future value using the training mean."""
    forecast_value = float(training_series.mean())

    return pd.Series(
        forecast_value,
        index=forecast_index,
        name="mean",
        dtype=float,
    )


def naive_forecast(
    training_series: pd.Series,
    forecast_index: pd.Index,
) -> pd.Series:
    """Forecast every future value using the final training value."""
    forecast_value = float(training_series.iloc[-1])

    return pd.Series(
        forecast_value,
        index=forecast_index,
        name="naive",
        dtype=float,
    )


def seasonal_naive_forecast(
    training_series: pd.Series,
    forecast_index: pd.Index,
    seasonal_period: int = 52,
) -> pd.Series:
    """
    Repeat the most recent seasonal cycle.

    For weekly data, seasonal_period=52 represents yearly seasonality.
    """
    if len(training_series) < seasonal_period:
        raise ValueError(
            "Training series is shorter than the seasonal period."
        )

    seasonal_pattern = training_series.iloc[
        -seasonal_period:
    ].to_numpy(dtype=float)

    repetitions = int(
        np.ceil(len(forecast_index) / seasonal_period)
    )

    forecast_values = np.tile(
        seasonal_pattern,
        repetitions,
    )[:len(forecast_index)]

    return pd.Series(
        forecast_values,
        index=forecast_index,
        name="seasonal_naive",
        dtype=float,
    )


def drift_forecast(
    training_series: pd.Series,
    forecast_index: pd.Index,
) -> pd.Series:
    """
    Forecast using a straight line between the first and last
    training observations.
    """
    if len(training_series) < 2:
        raise ValueError(
            "Drift forecast requires at least two observations."
        )

    first_value = float(training_series.iloc[0])
    last_value = float(training_series.iloc[-1])

    slope = (
        last_value - first_value
    ) / (len(training_series) - 1)

    forecast_steps = np.arange(
        1,
        len(forecast_index) + 1,
        dtype=float,
    )

    forecast_values = last_value + slope * forecast_steps

    return pd.Series(
        forecast_values,
        index=forecast_index,
        name="drift",
        dtype=float,
    )


def generate_benchmark_forecasts(
    training_series: pd.Series,
    test_series: pd.Series,
    seasonal_period: int = 52,
) -> pd.DataFrame:
    """
    Generate all required benchmark forecasts.
    """
    forecast_index = test_series.index

    forecasts = pd.DataFrame(
        index=forecast_index
    )

    forecasts["actual"] = test_series.astype(float)

    forecasts["mean"] = mean_forecast(
        training_series,
        forecast_index,
    )

    forecasts["naive"] = naive_forecast(
        training_series,
        forecast_index,
    )

    forecasts["seasonal_naive"] = seasonal_naive_forecast(
        training_series,
        forecast_index,
        seasonal_period=seasonal_period,
    )

    forecasts["drift"] = drift_forecast(
        training_series,
        forecast_index,
    )

    return forecasts