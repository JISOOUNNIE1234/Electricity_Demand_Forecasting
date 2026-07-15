from pathlib import Path

import pandas as pd

from electricity_demand.config import (
    TEMPERATURE_WEEKLY_FILE,
    WEEKLY_MODEL_DATA_FILE,
    WEEKLY_PROCESSED_FILE,
)


def load_weekly_dataset(
    file_path: Path,
) -> pd.DataFrame:
    """Load a weekly dataset with a timestamp index."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required weekly dataset not found: {file_path}"
        )

    data = pd.read_csv(
        file_path,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )

    data.index = pd.to_datetime(
        data.index,
        utc=True,
    )

    return data.sort_index()


def merge_weekly_load_and_temperature() -> pd.DataFrame:
    """
    Merge weekly electricity load and Berlin temperature.

    Only timestamps available in both datasets are retained.
    """
    load_data = load_weekly_dataset(
        WEEKLY_PROCESSED_FILE
    )

    temperature_data = load_weekly_dataset(
        TEMPERATURE_WEEKLY_FILE
    )

    model_data = load_data.join(
        temperature_data,
        how="inner",
    )

    model_data = model_data.dropna()

    if model_data.empty:
        raise ValueError(
            "Merged weekly load and temperature dataset is empty."
        )

    return model_data


def build_weekly_model_dataset() -> Path:
    """Build and save the merged weekly modelling dataset."""
    model_data = merge_weekly_load_and_temperature()

    WEEKLY_MODEL_DATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_data.to_csv(
        WEEKLY_MODEL_DATA_FILE
    )

    print(
        f"Saved weekly model data to: "
        f"{WEEKLY_MODEL_DATA_FILE}"
    )
    print(
        f"Merged observations: {len(model_data)}"
    )
    print(
        f"Date range: {model_data.index.min()} "
        f"to {model_data.index.max()}"
    )

    return WEEKLY_MODEL_DATA_FILE