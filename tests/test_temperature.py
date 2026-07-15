import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.temperature import (
    create_weekly_temperature_features,
)


def test_weekly_temperature_features_are_created() -> None:
    index = pd.date_range(
        start="2015-01-01",
        periods=24 * 14,
        freq="h",
        tz="UTC",
    )

    hourly_data = pd.DataFrame(
        {
            "temperature_2m": np.linspace(
                5.0,
                20.0,
                len(index),
            )
        },
        index=index,
    )

    result = create_weekly_temperature_features(
        hourly_data
    )

    required_columns = {
        "temp_mean",
        "temp_min",
        "temp_max",
        "heating_degree_days",
        "cooling_degree_days",
    }

    assert required_columns.issubset(result.columns)
    assert result.isna().sum().sum() == 0