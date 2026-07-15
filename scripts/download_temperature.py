import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.temperature import (
    build_weekly_temperature_dataset,
)


def main() -> None:
    """Download and process Berlin temperature data."""
    output_file = build_weekly_temperature_dataset()
    print(f"Temperature processing completed: {output_file}")


if __name__ == "__main__":
    main()