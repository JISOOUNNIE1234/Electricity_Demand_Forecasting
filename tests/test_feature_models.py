import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.feature_models import (
    calculate_feature_importance,
    generate_feature_model_forecasts,
)


def test_feature_forecast_matches_test_length() -> None:
    training_x = pd.DataFrame(
        {
            "feature_1": np.arange(20),
            "feature_2": np.arange(20) * 2,
        }
    )

    training_y = pd.Series(
        np.arange(20, dtype=float)
    )

    test_x = pd.DataFrame(
        {
            "feature_1": [20, 21, 22],
            "feature_2": [40, 42, 44],
        },
        index=pd.date_range(
            start="2020-01-05",
            periods=3,
            freq="W-SUN",
        ),
    )

    random_forest = RandomForestRegressor(
        n_estimators=10,
        random_state=42,
    )

    gradient_model = RandomForestRegressor(
        n_estimators=10,
        random_state=42,
    )

    random_forest.fit(
        training_x,
        training_y,
    )

    gradient_model.fit(
        training_x,
        training_y,
    )

    forecasts = generate_feature_model_forecasts(
        random_forest,
        gradient_model,
        test_x,
    )

    assert len(forecasts) == len(test_x)
    assert forecasts.isna().sum().sum() == 0


def test_feature_importances_sum_to_one() -> None:
    features = pd.DataFrame(
        {
            "x1": np.arange(20),
            "x2": np.arange(20) ** 2,
        }
    )

    target = pd.Series(
        np.arange(20, dtype=float)
    )

    model = RandomForestRegressor(
        n_estimators=10,
        random_state=42,
    )

    model.fit(features, target)

    importance = calculate_feature_importance(
        model,
        ["x1", "x2"],
        "random_forest",
    )

    assert np.isclose(
        importance["importance"].sum(),
        1.0,
    )