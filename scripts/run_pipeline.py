import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.pipeline import (
    PipelineOptions,
    run_pipeline,
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line pipeline options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the German electricity-demand forecasting pipeline."
        )
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading the raw electricity dataset.",
    )

    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing and feature construction.",
    )

    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip exploratory and stationarity analysis.",
    )

    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Skip benchmark forecasting.",
    )

    parser.add_argument(
        "--skip-sarima",
        action="store_true",
        help="Skip SARIMA modelling.",
    )

    parser.add_argument(
        "--skip-sarimax",
        action="store_true",
        help="Skip SARIMAX modelling.",
    )

    parser.add_argument(
        "--skip-feature-models",
        action="store_true",
        help="Skip feature-based regression models.",
    )

    parser.add_argument(
        "--include-lstm",
        action="store_true",
        help=(
            "Include the LSTM stage. This requires the Python 3.12 "
            "TensorFlow environment."
        ),
    )

    parser.add_argument(
        "--force-models",
        action="store_true",
        help=(
            "Rerun expensive modelling stages even when their output "
            "files already exist."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Run the pipeline using command-line options.
    """
    arguments = parse_arguments()

    options = PipelineOptions(
        download_data=not arguments.skip_download,
        preprocess_data=not arguments.skip_preprocessing,
        run_analysis=not arguments.skip_analysis,
        run_benchmarks=not arguments.skip_benchmarks,
        run_sarima=not arguments.skip_sarima,
        run_sarimax=not arguments.skip_sarimax,
        run_feature_models=not arguments.skip_feature_models,
        run_lstm=arguments.include_lstm,
        force_models=arguments.force_models,
    )

    run_pipeline(options)


if __name__ == "__main__":
    main()