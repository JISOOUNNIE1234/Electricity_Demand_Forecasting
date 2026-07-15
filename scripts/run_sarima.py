import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.benchmarks import (
    split_train_test,
)
from electricity_demand.config import (
    FORECASTS_DIR,
    METRICS_DIR,
    SARIMA_BEST_MODEL_FILE,
    SARIMA_D_VALUES,
    SARIMA_FORECAST_FILE,
    SARIMA_METRICS_FILE,
    SARIMA_P_VALUES,
    SARIMA_Q_VALUES,
    SARIMA_SEARCH_RESULTS_FILE,
    SARIMA_SEASONAL_D_VALUES,
    SARIMA_SEASONAL_P_VALUES,
    SARIMA_SEASONAL_Q_VALUES,
    TEST_HORIZON_WEEKS,
    WEEKLY_PROCESSED_FILE,
    WEEKLY_SEASONAL_PERIOD,
)
from electricity_demand.evaluation import (
    evaluate_forecast,
)
from electricity_demand.plotting import (
    plot_sarima_forecast,
    plot_sarima_residual_diagnostics,
)
from electricity_demand.sarima import (
    calculate_residual_diagnostics,
    extract_best_parameters,
    fit_sarima_model,
    generate_parameter_combinations,
    generate_sarima_forecast,
    search_sarima_parameters,
)


def load_weekly_series() -> pd.Series:
    """Load weekly German electricity demand."""
    if not WEEKLY_PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"Weekly data not found: "
            f"{WEEKLY_PROCESSED_FILE}"
        )

    weekly_data = pd.read_csv(
        WEEKLY_PROCESSED_FILE,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )

    return weekly_data["load_gw"]


def main() -> None:
    """Search, fit, forecast and evaluate SARIMA."""
    weekly_series = load_weekly_series()

    train, test = split_train_test(
        weekly_series,
        test_size=TEST_HORIZON_WEEKS,
    )

    print("\nStage 1: Searching all required p, d and q values")
    print("------------------------------------------------")

    stage_one_combinations = (
        generate_parameter_combinations(
            p_values=SARIMA_P_VALUES,
            d_values=SARIMA_D_VALUES,
            q_values=SARIMA_Q_VALUES,
            seasonal_p_values=[1],
            seasonal_d_values=[0],
            seasonal_q_values=[1],
            seasonal_period=WEEKLY_SEASONAL_PERIOD,
        )
    )

    stage_one_results = search_sarima_parameters(
        training_series=train,
        parameter_combinations=stage_one_combinations,
    )

    best_non_seasonal_order, _ = extract_best_parameters(
        stage_one_results
    )

    print(
        f"\nBest non-seasonal order from Stage 1: "
        f"{best_non_seasonal_order}"
    )

    print("\nStage 2: Refining the seasonal order")
    print("------------------------------------")

    seasonal_candidates = [
        (0, 0, 0, WEEKLY_SEASONAL_PERIOD),
        (1, 0, 0, WEEKLY_SEASONAL_PERIOD),
        (0, 0, 1, WEEKLY_SEASONAL_PERIOD),
        (1, 0, 1, WEEKLY_SEASONAL_PERIOD),
        (0, 1, 0, WEEKLY_SEASONAL_PERIOD),
        (1, 1, 0, WEEKLY_SEASONAL_PERIOD),
        (0, 1, 1, WEEKLY_SEASONAL_PERIOD),
        (1, 1, 1, WEEKLY_SEASONAL_PERIOD),
    ]

    stage_two_combinations = [
        (
            best_non_seasonal_order,
            seasonal_order,
        )
        for seasonal_order in seasonal_candidates
    ]

    stage_two_results = search_sarima_parameters(
        training_series=train,
        parameter_combinations=stage_two_combinations,
    )

    search_results = pd.concat(
        [
            stage_one_results.assign(search_stage="non_seasonal_search"),
            stage_two_results.assign(search_stage="seasonal_refinement"),
        ],
        ignore_index=True,
    )

    search_results = search_results.sort_values(
        by=["aic", "bic"],
        na_position="last",
    ).reset_index(drop=True)

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    FORECASTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    search_results.to_csv(
        SARIMA_SEARCH_RESULTS_FILE,
        index=False,
    )

    best_order, best_seasonal_order = (
        extract_best_parameters(
            search_results
        )
    )

    print("\nBest SARIMA parameters")
    print("----------------------")
    print(f"Order:          {best_order}")
    print(
        f"Seasonal order: {best_seasonal_order}"
    )
    print(
        f"Best AIC:       "
        f"{search_results.iloc[0]['aic']:.3f}"
    )

    fitted_model = fit_sarima_model(
        training_series=train,
        order=best_order,
        seasonal_order=best_seasonal_order,
        max_iterations=300,
    )

    sarima_forecast = (
        generate_sarima_forecast(
            fitted_model=fitted_model,
            forecast_index=test.index,
            confidence_level=0.95,
        )
    )

    sarima_forecast.insert(
        0,
        "actual",
        test.astype(float),
    )

    sarima_forecast.to_csv(
        SARIMA_FORECAST_FILE
    )

    metrics = evaluate_forecast(
        actual=sarima_forecast["actual"],
        forecast=sarima_forecast["sarima"],
        training_series=train,
        model_name="sarima",
        seasonal_period=(
            WEEKLY_SEASONAL_PERIOD
        ),
    )

    metrics_table = pd.DataFrame(
        [metrics]
    )

    metrics_table.to_csv(
        SARIMA_METRICS_FILE,
        index=False,
    )

    residual_diagnostics = (
        calculate_residual_diagnostics(
            fitted_model=fitted_model,
            number_of_lags=20,
        )
    )

    residual_diagnostics.to_csv(
        METRICS_DIR
        / "sarima_residual_diagnostics.csv",
        index=False,
    )

    best_model_summary = pd.DataFrame(
        {
            "order": [str(best_order)],
            "seasonal_order": [
                str(best_seasonal_order)
            ],
            "aic": [float(fitted_model.aic)],
            "bic": [float(fitted_model.bic)],
            "log_likelihood": [
                float(fitted_model.llf)
            ],
        }
    )

    best_model_summary.to_csv(
        SARIMA_BEST_MODEL_FILE,
        index=False,
    )

    forecast_figure = plot_sarima_forecast(
        training_series=train,
        forecast_data=sarima_forecast,
    )

    residual_figure = (
        plot_sarima_residual_diagnostics(
            residuals=fitted_model.resid
        )
    )

    print("\nSARIMA metrics")
    print("--------------")
    print(
        metrics_table.to_string(
            index=False
        )
    )

    print("\nResidual diagnostics")
    print("--------------------")
    print(
        residual_diagnostics.to_string(
            index=False
        )
    )

    print(
        f"\nSearch results: {SARIMA_SEARCH_RESULTS_FILE}"
    )
    print(
        f"Forecasts:      {SARIMA_FORECAST_FILE}"
    )
    print(
        f"Forecast plot:  {forecast_figure}"
    )
    print(
        f"Residual plot:  {residual_figure}"
    )


if __name__ == "__main__":
    main()