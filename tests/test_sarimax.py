import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.sarimax import (
    standardise_exogenous_data,
)


def test_scaling_uses_training_statistics_only() -> None:
    train = pd.DataFrame(
        {
            "temperature": [10.0, 20.0, 30.0]
        }
    )

    test = pd.DataFrame(
        {
            "temperature": [100.0]
        }
    )

    scaled_train, scaled_test, mean, std = (
        standardise_exogenous_data(
            train,
            test,
        )
    )

    assert mean["temperature"] == 20.0
    assert round(
        scaled_train["temperature"].mean(),
        10,
    ) == 0.0

    expected_test_value = (
        100.0 - mean["temperature"]
    ) / std["temperature"]

    assert (
        scaled_test["temperature"].iloc[0]
        == expected_test_value
    )