import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.config import (
    FEATURE_DATA_FILE,
    FEATURE_IMPORTANCE_FILE,
    FEATURE_MODEL_FORECAST_FILE,
    FEATURE_MODEL_METRICS_FILE,
    FEATURE_MODEL_SEARCH_FILE,
    RANDOM_STATE,
    TEST_HORIZON_WEEKS,
    WEEKLY_SEASONAL_PERIOD,
)
from electricity_demand.evaluation import (
    evaluate_multiple_forecasts,
)
from electricity_demand.feature_models import (
    calculate_feature_importance,
    generate_feature_model_forecasts,
    search_gradient_boosting,
    search_random_forest,
)
from electricity_demand.plotting import (
    plot_feature_importance,
    plot_feature_model_forecasts,
)


def main() -> None:
    """
    Train, tune and evaluate feature-based regression models.
    """
    data = pd.read_csv(
        FEATURE_DATA_FILE,
        parse_dates=["timestamp"],
        index_col="timestamp",
    ).sort_index()

    target_column = "load_gw"

    # Contemporary observed test-period weather makes these
    # conditional forecasts. Target-derived features remain strictly
    # historical because they were shifted before construction.
    feature_columns = [
        column
        for column in data.columns
        if column != target_column
    ]

    train_data = data.iloc[
        :-TEST_HORIZON_WEEKS
    ].copy()

    test_data = data.iloc[
        -TEST_HORIZON_WEEKS:
    ].copy()

    train_x = train_data[feature_columns]
    train_y = train_data[target_column]

    test_x = test_data[feature_columns]
    test_y = test_data[target_column]

    print("Searching Random Forest parameters...")

    random_forest, random_forest_results = (
        search_random_forest(
            training_features=train_x,
            training_target=train_y,
            random_state=RANDOM_STATE,
        )
    )

    print("Searching Gradient Boosting parameters...")

    gradient_boosting, gradient_results = (
        search_gradient_boosting(
            training_features=train_x,
            training_target=train_y,
            random_state=RANDOM_STATE,
        )
    )

    forecasts = generate_feature_model_forecasts(
        random_forest_model=random_forest,
        gradient_boosting_model=gradient_boosting,
        test_features=test_x,
    )

    forecasts.insert(
        0,
        "actual",
        test_y.astype(float),
    )

    metrics = evaluate_multiple_forecasts(
        forecasts=forecasts,
        training_series=train_y,
        seasonal_period=WEEKLY_SEASONAL_PERIOD,
    )

    random_forest_importance = (
        calculate_feature_importance(
            model=random_forest,
            feature_names=feature_columns,
            model_name="random_forest",
        )
    )

    gradient_importance = (
        calculate_feature_importance(
            model=gradient_boosting,
            feature_names=feature_columns,
            model_name="gradient_boosting",
        )
    )

    combined_importance = pd.concat(
        [
            random_forest_importance,
            gradient_importance,
        ],
        ignore_index=True,
    )

    combined_search_results = pd.concat(
        [
            random_forest_results,
            gradient_results,
        ],
        ignore_index=True,
    )

    forecasts.to_csv(
        FEATURE_MODEL_FORECAST_FILE
    )

    metrics.to_csv(
        FEATURE_MODEL_METRICS_FILE,
        index=False,
    )

    combined_importance.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
    )

    combined_search_results.to_csv(
        FEATURE_MODEL_SEARCH_FILE,
        index=False,
    )

    forecast_plot = (
        plot_feature_model_forecasts(
            training_series=train_y,
            forecast_data=forecasts,
        )
    )

    importance_plot = plot_feature_importance(
        random_forest_importance
    )

    print("\nFeature model metrics")
    print("---------------------")
    print(metrics.to_string(index=False))

    print("\nBest Random Forest configuration")
    print("--------------------------------")
    print(
        random_forest_results.head(1).to_string(
            index=False
        )
    )

    print("\nBest Gradient Boosting configuration")
    print("------------------------------------")
    print(
        gradient_results.head(1).to_string(
            index=False
        )
    )

    print(f"\nForecast plot: {forecast_plot}")
    print(f"Importance plot: {importance_plot}")

    print(
        "\nObserved contemporary temperature was used in "
        "the test period. These results are therefore "
        "conditional forecasts."
    )


if __name__ == "__main__":
    main()