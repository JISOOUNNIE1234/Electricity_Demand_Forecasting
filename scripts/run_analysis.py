import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.analysis import run_part1_analysis
from electricity_demand.config import (
    DAILY_PROCESSED_FILE,
    HOURLY_PROCESSED_FILE,
    WEEKLY_PROCESSED_FILE,
)


def load_processed_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load a processed electricity-demand dataset.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {file_path}. "
            "Run: python scripts/make_features.py"
        )

    data = pd.read_csv(
        file_path,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )

    return data


def main() -> None:
    """Run exploratory and stationarity analysis."""
    hourly_data = load_processed_dataset(
        HOURLY_PROCESSED_FILE
    )
    daily_data = load_processed_dataset(
        DAILY_PROCESSED_FILE
    )
    weekly_data = load_processed_dataset(
        WEEKLY_PROCESSED_FILE
    )

    run_part1_analysis(
        hourly_data=hourly_data,
        daily_data=daily_data,
        weekly_data=weekly_data,
    )


if __name__ == "__main__":
    main()