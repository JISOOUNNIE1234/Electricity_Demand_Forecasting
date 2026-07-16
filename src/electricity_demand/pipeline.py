from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from electricity_demand.config import (
    BENCHMARK_FORECAST_FILE,
    FEATURE_MODEL_FORECAST_FILE,
    LSTM_FORECAST_FILE,
    SARIMA_FORECAST_FILE,
    SARIMAX_FORECAST_FILE,
    WEEKLY_PROCESSED_FILE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


@dataclass
class PipelineOptions:
    """
    Control which parts of the forecasting pipeline are executed.

    Expensive modelling stages are skipped when their outputs already
    exist unless force_models=True.
    """

    download_data: bool = True
    preprocess_data: bool = True
    run_analysis: bool = True
    run_benchmarks: bool = True
    run_sarima: bool = True
    run_sarimax: bool = True
    run_feature_models: bool = True
    run_lstm: bool = False
    force_models: bool = False


def run_script(script_name: str) -> None:
    """
    Execute one project script using the active Python interpreter.
    """
    script_path = SCRIPTS_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Pipeline script does not exist: {script_path}"
        )

    print("\n" + "=" * 70)
    print(f"Running: {script_name}")
    print("=" * 70)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def run_if_output_missing(
    script_name: str,
    expected_output: Path,
    force: bool = False,
) -> None:
    """
    Run a modelling script only when its expected output is missing.

    Parameters
    ----------
    script_name:
        Script inside the scripts directory.
    expected_output:
        Output file used to determine whether the stage has already
        been completed.
    force:
        Rerun the stage even when the output exists.
    """
    if expected_output.exists() and not force:
        print(
            f"Skipping {script_name}: output already exists at "
            f"{expected_output}"
        )
        return

    run_script(script_name)


def run_pipeline(
    options: PipelineOptions | None = None,
) -> None:
    """
    Run the German electricity-demand forecasting pipeline.

    The default workflow executes lightweight stages and reuses
    existing model outputs. LSTM is disabled by default because it
    requires the separate Python 3.12 TensorFlow environment.
    """
    if options is None:
        options = PipelineOptions()

    print("\nGerman Electricity Demand Forecasting Pipeline")
    print("==============================================")

    if options.download_data:
        run_script("download_data.py")

    if options.preprocess_data:
        run_script("make_features.py")

    if not WEEKLY_PROCESSED_FILE.exists():
        raise FileNotFoundError(
            "Processed weekly data are unavailable. "
            "Run the preprocessing stage first."
        )

    if options.run_analysis:
        run_script("run_analysis.py")

    if options.run_benchmarks:
        run_if_output_missing(
            script_name="run_benchmarks.py",
            expected_output=BENCHMARK_FORECAST_FILE,
            force=options.force_models,
        )

    if options.run_sarima:
        run_if_output_missing(
            script_name="run_sarima.py",
            expected_output=SARIMA_FORECAST_FILE,
            force=options.force_models,
        )

    if options.run_sarimax:
        run_if_output_missing(
            script_name="run_sarimax.py",
            expected_output=SARIMAX_FORECAST_FILE,
            force=options.force_models,
        )

    if options.run_feature_models:
        run_if_output_missing(
            script_name="run_feature_models.py",
            expected_output=FEATURE_MODEL_FORECAST_FILE,
            force=options.force_models,
        )

    if options.run_lstm:
        run_if_output_missing(
            script_name="run_lstm.py",
            expected_output=LSTM_FORECAST_FILE,
            force=options.force_models,
        )

    print("\n" + "=" * 70)
    print("Pipeline completed successfully.")
    print("=" * 70)

    print(
        "\nExisting expensive model outputs were reused unless "
        "--force-models was supplied."
    )