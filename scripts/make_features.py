import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.preprocessing import build_processed_datasets


def main() -> None:
    """Create cleaned hourly, daily and weekly electricity datasets."""
    generated_files = build_processed_datasets()

    print("\nGenerated files")
    print("---------------")

    for dataset_name, file_path in generated_files.items():
        print(f"{dataset_name.title()}: {file_path}")


if __name__ == "__main__":
    main()