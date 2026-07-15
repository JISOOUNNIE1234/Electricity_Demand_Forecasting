import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.features import (
    build_weekly_model_dataset,
)
from electricity_demand.preprocessing import (
    build_processed_datasets,
)
from electricity_demand.temperature import (
    build_weekly_temperature_dataset,
)


def main() -> None:
    """Build all processed modelling datasets."""
    build_processed_datasets()
    build_weekly_temperature_dataset()
    model_file = build_weekly_model_dataset()

    print(f"\nFinal model dataset: {model_file}")


if __name__ == "__main__":
    main()