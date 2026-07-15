from __future__ import annotations

import itertools
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper


def generate_parameter_combinations(
    p_values: Iterable[int],
    d_values: Iterable[int],
    q_values: Iterable[int],
    seasonal_p_values: Iterable[int],
    seasonal_d_values: Iterable[int],
    seasonal_q_values: Iterable[int],
    seasonal_period: int = 52,
) -> list[
    tuple[
        tuple[int, int, int],
        tuple[int, int, int, int],
    ]
]:
    """
    Generate all non-seasonal and seasonal SARIMA combinations.
    """
    non_seasonal_orders = list(
        itertools.product(
            p_values,
            d_values,
            q_values,
        )
    )

    seasonal_components = list(
        itertools.product(
            seasonal_p_values,
            seasonal_d_values,
            seasonal_q_values,
        )
    )

    combinations = []

    for order in non_seasonal_orders:
        for seasonal_component in seasonal_components:
            seasonal_order = (
                seasonal_component[0],
                seasonal_component[1],
                seasonal_component[2],
                seasonal_period,
            )

            combinations.append(
                (order, seasonal_order)
            )

    return combinations


def fit_sarima_model(
    training_series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    max_iterations: int = 200,
) -> SARIMAXResultsWrapper:
    """
    Fit one SARIMA model to the training series.
    """
    model = SARIMAX(
        training_series.astype(float),
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted_model = model.fit(
        disp=False,
        maxiter=max_iterations,
    )

    return fitted_model


def search_sarima_parameters(
    training_series: pd.Series,
    parameter_combinations: list[
        tuple[
            tuple[int, int, int],
            tuple[int, int, int, int],
        ]
    ],
) -> pd.DataFrame:
    """
    Fit all candidate SARIMA models and rank them by AIC.

    Failed or non-converged models are recorded rather than
    terminating the complete search.
    """
    results = []
    total_models = len(parameter_combinations)

    print(
        f"Starting SARIMA search across "
        f"{total_models:,} parameter combinations."
    )

    for model_number, (
        order,
        seasonal_order,
    ) in enumerate(
        parameter_combinations,
        start=1,
    ):
        record = {
            "order": str(order),
            "seasonal_order": str(seasonal_order),
            "p": order[0],
            "d": order[1],
            "q": order[2],
            "P": seasonal_order[0],
            "D": seasonal_order[1],
            "Q": seasonal_order[2],
            "seasonal_period": seasonal_order[3],
            "aic": np.nan,
            "bic": np.nan,
            "converged": False,
            "error": "",
        }

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                fitted_model = fit_sarima_model(
                    training_series=training_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    max_iterations=50,
                )

            record["aic"] = float(fitted_model.aic)
            record["bic"] = float(fitted_model.bic)

            return_values = fitted_model.mle_retvals
            record["converged"] = bool(
                return_values.get("converged", True)
            )

        except (
            ValueError,
            np.linalg.LinAlgError,
            RuntimeError,
        ) as error:
            record["error"] = str(error)

        results.append(record)

        if (
            model_number == 1
            or model_number % 25 == 0
            or model_number == total_models
        ):
            successful = sum(
                pd.notna(item["aic"])
                for item in results
            )

            print(
                f"Processed {model_number:,}/"
                f"{total_models:,} models; "
                f"{successful:,} successful."
            )

    results_table = pd.DataFrame(results)

    successful_results = results_table.dropna(
        subset=["aic"]
    )

    if successful_results.empty:
        raise RuntimeError(
            "No SARIMA candidate model fitted successfully."
        )

    results_table = results_table.sort_values(
        by=["aic", "bic"],
        na_position="last",
    ).reset_index(drop=True)

    return results_table


def extract_best_parameters(
    search_results: pd.DataFrame,
) -> tuple[
    tuple[int, int, int],
    tuple[int, int, int, int],
]:
    """
    Extract the lowest-AIC SARIMA orders.
    """
    valid_results = (
    search_results
    .dropna(subset=["aic"])
    .sort_values(
        by=["aic", "bic"],
        ascending=True,
        )
    )

    if valid_results.empty:
        raise ValueError(
        "Search results contain no successful models."
        )

    best_row = valid_results.iloc[0]

    order = (
        int(best_row["p"]),
        int(best_row["d"]),
        int(best_row["q"]),
    )

    seasonal_order = (
        int(best_row["P"]),
        int(best_row["D"]),
        int(best_row["Q"]),
        int(best_row["seasonal_period"]),
    )

    return order, seasonal_order


def generate_sarima_forecast(
    fitted_model: SARIMAXResultsWrapper,
    forecast_index: pd.Index,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Generate point forecasts and confidence intervals.
    """
    forecast_object = fitted_model.get_forecast(
        steps=len(forecast_index)
    )

    forecast_mean = forecast_object.predicted_mean
    confidence_interval = (
        forecast_object.conf_int(
            alpha=1.0 - confidence_level
        )
    )

    forecast = pd.DataFrame(
        index=forecast_index
    )

    forecast["sarima"] = np.asarray(
        forecast_mean,
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


def calculate_residual_diagnostics(
    fitted_model: SARIMAXResultsWrapper,
    number_of_lags: int = 20,
) -> pd.DataFrame:
    """
    Calculate residual summary and Ljung-Box diagnostics.
    """
    residuals = pd.Series(
        fitted_model.resid
    ).dropna()

    residuals = residuals.loc[
        np.isfinite(residuals)
    ]

    ljung_box = acorr_ljungbox(
        residuals,
        lags=[number_of_lags],
        return_df=True,
    )

    diagnostics = pd.DataFrame(
        {
            "residual_mean": [
                float(residuals.mean())
            ],
            "residual_standard_deviation": [
                float(residuals.std())
            ],
            "ljung_box_lag": [
                number_of_lags
            ],
            "ljung_box_statistic": [
                float(
                    ljung_box["lb_stat"].iloc[0]
                )
            ],
            "ljung_box_p_value": [
                float(
                    ljung_box["lb_pvalue"].iloc[0]
                )
            ],
        }
    )

    return diagnostics