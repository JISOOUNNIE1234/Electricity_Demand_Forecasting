import ast
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.config import (
    SARIMA_BEST_MODEL_FILE,
    SARIMAX_FORECAST_FILE,
    SARIMAX_METRICS_FILE,
    SARIMAX_MODEL_SUMMARY_FILE,
    TEST_HORIZON_WEEKS,
    WEEKLY_MODEL_DATA_FILE,
    WEEKLY_SEASONAL_PERIOD,
)
from electricity_demand.evaluation import (
    evaluate_forecast,
)
from electricity_demand.plotting import (
    plot_sarimax_forecast,
)
from electricity_demand.sarimax import (
    fit_sarimax_model,
    generate_sarimax_forecast,
    standardise_exogenous_data,
)


def load_best_sarima_orders() -> tuple[
    tuple[int, int, int],
    tuple[int, int, int, int],
]:
    """Load the best orders selected during Part 3."""
    model_summary = pd.read_csv(
        SARIMA_BEST_MODEL_FILE
    )

    order = tuple(
        ast.literal_eval(
            model_summary.loc[0, "order"]
        )
    )

    seasonal_order = tuple(
        ast.literal_eval(
            model_summary.loc[0, "seasonal_order"]
        )
    )

    return order, seasonal_order


def main() -> None:
    """Fit and evaluate the temperature-based SARIMAX model."""
    model_data = pd.read_csv(
        WEEKLY_MODEL_DATA_FILE,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )

    model_data = model_data.sort_index()

    train_data = model_data.iloc[
        :-TEST_HORIZON_WEEKS
    ].copy()

    test_data = model_data.iloc[
        -TEST_HORIZON_WEEKS:
    ].copy()

    target_column = "load_gw"

    exogenous_columns = [
        "temp_mean",
        "temp_min",
        "temp_max",
        "heating_degree_days",
        "cooling_degree_days",
    ]

    train_target = train_data[target_column]
    test_target = test_data[target_column]

    train_exog = train_data[exogenous_columns]
    test_exog = test_data[exogenous_columns]

    (
        scaled_train_exog,
        scaled_test_exog,
        training_mean,
        training_std,
    ) = standardise_exogenous_data(
        train_exog,
        test_exog,
    )

    order, seasonal_order = (
        load_best_sarima_orders()
    )

    print(f"Using SARIMA order: {order}")
    print(
        f"Using seasonal order: {seasonal_order}"
    )

    fitted_model = fit_sarimax_model(
        training_series=train_target,
        training_exog=scaled_train_exog,
        order=order,
        seasonal_order=seasonal_order,
    )

    forecast = generate_sarimax_forecast(
        fitted_model=fitted_model,
        test_exog=scaled_test_exog,
        forecast_index=test_target.index,
    )

    forecast.insert(
        0,
        "actual",
        test_target.astype(float),
    )

    forecast.to_csv(
        SARIMAX_FORECAST_FILE
    )

    metrics = evaluate_forecast(
        actual=forecast["actual"],
        forecast=forecast["sarimax"],
        training_series=train_target,
        model_name="sarimax",
        seasonal_period=WEEKLY_SEASONAL_PERIOD,
    )

    metrics_table = pd.DataFrame(
        [metrics]
    )

    metrics_table.to_csv(
        SARIMAX_METRICS_FILE,
        index=False,
    )

    coefficient_table = pd.DataFrame(
        {
            "parameter": fitted_model.params.index,
            "estimate": fitted_model.params.values,
            "standard_error": fitted_model.bse.values,
            "p_value": fitted_model.pvalues.values,
        }
    )

    coefficient_table.to_csv(
        SARIMAX_MODEL_SUMMARY_FILE,
        index=False,
    )

    scaling_table = pd.DataFrame(
        {
            "variable": training_mean.index,
            "training_mean": training_mean.values,
            "training_standard_deviation": (
                training_std.values
            ),
        }
    )

    scaling_table.to_csv(
        Path(SARIMAX_MODEL_SUMMARY_FILE).with_name(
            "sarimax_exogenous_scaling.csv"
        ),
        index=False,
    )

    figure_path = plot_sarimax_forecast(
        training_series=train_target,
        forecast_data=forecast,
    )

    print("\nSARIMAX completed.")
    print(metrics_table.to_string(index=False))
    print(f"\nAIC: {fitted_model.aic:.3f}")
    print(f"BIC: {fitted_model.bic:.3f}")
    print(f"Forecast file: {SARIMAX_FORECAST_FILE}")
    print(f"Forecast figure: {figure_path}")

    print(
        "\nImportant: observed temperature from the test "
        "period was used. This is a conditional/explanatory "
        "forecast, not a fully operational forecast."
    )


if __name__ == "__main__":
    main()