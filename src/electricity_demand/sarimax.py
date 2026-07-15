from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import (
    SARIMAX,
    SARIMAXResultsWrapper,
)


def standardise_exogenous_data(
    training_exog: pd.DataFrame,
    test_exog: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Standardise exogenous variables using training statistics only.

    This prevents information from the test period influencing
    preprocessing.
    """
    training_mean = training_exog.mean()
    training_std = training_exog.std()

    if (training_std == 0).any():
        constant_columns = training_std[
            training_std == 0
        ].index.tolist()

        raise ValueError(
            f"Constant exogenous columns: {constant_columns}"
        )

    scaled_train = (
        training_exog - training_mean
    ) / training_std

    scaled_test = (
        test_exog - training_mean
    ) / training_std

    return (
        scaled_train,
        scaled_test,
        training_mean,
        training_std,
    )


def fit_sarimax_model(
    training_series: pd.Series,
    training_exog: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> SARIMAXResultsWrapper:
    """Fit SARIMAX with external covariates."""
    model = SARIMAX(
        endog=training_series.astype(float),
        exog=training_exog.astype(float),
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    return model.fit(
        disp=False,
        maxiter=300,
    )


def generate_sarimax_forecast(
    fitted_model: SARIMAXResultsWrapper,
    test_exog: pd.DataFrame,
    forecast_index: pd.Index,
) -> pd.DataFrame:
    """Generate SARIMAX forecasts and 95% intervals."""
    forecast_result = fitted_model.get_forecast(
        steps=len(forecast_index),
        exog=test_exog.astype(float),
    )

    confidence_interval = forecast_result.conf_int(
        alpha=0.05
    )

    forecast = pd.DataFrame(
        index=forecast_index
    )

    forecast["sarimax"] = np.asarray(
        forecast_result.predicted_mean,
        dtype=float,
    )

    forecast["lower_95"] = np.asarray(
        confidence_interval.iloc[:, 0],
        dtype=float,
    )

    forecast["upper_95"] = np.asarray(
        confidence_interval.iloc[:, 1],
        dtype=float,
    )

    return forecast