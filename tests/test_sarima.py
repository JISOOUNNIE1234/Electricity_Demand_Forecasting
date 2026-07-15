import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.sarima import (
    extract_best_parameters,
    generate_parameter_combinations,
)


def test_parameter_grid_contains_all_combinations() -> None:
    combinations = generate_parameter_combinations(
        p_values=range(0, 2),
        d_values=range(0, 2),
        q_values=range(0, 2),
        seasonal_p_values=range(0, 2),
        seasonal_d_values=range(0, 2),
        seasonal_q_values=range(0, 2),
        seasonal_period=52,
    )

    assert len(combinations) == 64


def test_parameter_grid_uses_weekly_seasonality() -> None:
    combinations = generate_parameter_combinations(
        p_values=[1],
        d_values=[0],
        q_values=[1],
        seasonal_p_values=[1],
        seasonal_d_values=[1],
        seasonal_q_values=[1],
        seasonal_period=52,
    )

    order, seasonal_order = combinations[0]

    assert order == (1, 0, 1)
    assert seasonal_order == (1, 1, 1, 52)


def test_extract_best_parameters_uses_lowest_aic() -> None:
    search_results = pd.DataFrame(
        {
            "p": [1, 2],
            "d": [0, 1],
            "q": [1, 2],
            "P": [0, 1],
            "D": [1, 1],
            "Q": [1, 0],
            "seasonal_period": [52, 52],
            "aic": [200.0, 150.0],
            "bic": [210.0, 165.0],
        }
    )

    order, seasonal_order = (
        extract_best_parameters(
            search_results
        )
    )

    assert order == (2, 1, 2)
    assert seasonal_order == (1, 1, 0, 52)