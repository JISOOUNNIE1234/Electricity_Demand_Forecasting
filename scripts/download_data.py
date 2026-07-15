import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.data import download_electricity_data


def main() -> None:
    """Download the raw German electricity time-series source file."""
    downloaded_file = download_electricity_data(overwrite=False)
    print(f"Raw data available at: {downloaded_file}")


if __name__ == "__main__":
    main()