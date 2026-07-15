from pathlib import Path

import pandas as pd

from electricity_demand.config import (
    DAILY_PROCESSED_FILE,
    HOURLY_PROCESSED_FILE,
    PROCESSED_DATA_DIR,
    START_DATE,
    WEEKLY_PROCESSED_FILE,
)
from electricity_demand.data import load_raw_germany_electricity_data


def clean_hourly_load(
    data: pd.DataFrame,
    start_date: str = START_DATE,
) -> pd.DataFrame:
    """
    Clean the German hourly electricity demand series.

    Processing steps:
    - sort observations chronologically;
    - remove duplicate timestamps;
    - keep observations from the requested start date;
    - convert electricity demand from MW to GW;
    - remove missing target observations;
    - set timestamp as the index.

    Parameters
    ----------
    data:
        Raw German electricity load data.
    start_date:
        Earliest date to retain.

    Returns
    -------
    pandas.DataFrame
        Clean hourly electricity demand data.
    """
    required_columns = {"timestamp", "load_mw"}

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Input data are missing columns: {sorted(missing_columns)}"
        )

    cleaned = data.copy()

    cleaned["timestamp"] = pd.to_datetime(
        cleaned["timestamp"],
        utc=True,
        errors="coerce",
    )

    cleaned["load_mw"] = pd.to_numeric(
        cleaned["load_mw"],
        errors="coerce",
    )

    cleaned = cleaned.dropna(subset=["timestamp"])
    cleaned = cleaned.sort_values("timestamp")
    cleaned = cleaned.drop_duplicates(subset=["timestamp"], keep="first")

    start_timestamp = pd.Timestamp(start_date, tz="UTC")
    cleaned = cleaned.loc[cleaned["timestamp"] >= start_timestamp]

    cleaned["load_gw"] = cleaned["load_mw"] / 1000.0

    cleaned = cleaned.dropna(subset=["load_gw"])
    cleaned = cleaned.set_index("timestamp")

    cleaned = cleaned[["load_gw"]]

    return cleaned


def aggregate_electricity_load(
    hourly_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate hourly electricity demand to daily and weekly averages.

    Parameters
    ----------
    hourly_data:
        Clean hourly electricity demand indexed by timestamp.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Daily and weekly average electricity demand.
    """
    if "load_gw" not in hourly_data.columns:
        raise ValueError("Expected a 'load_gw' column.")

    if not isinstance(hourly_data.index, pd.DatetimeIndex):
        raise TypeError("The hourly data must use a DatetimeIndex.")

    daily_data = hourly_data.resample("D").mean()
    daily_data = daily_data.dropna(subset=["load_gw"])

    weekly_data = hourly_data.resample("W-SUN").mean()
    weekly_data = weekly_data.dropna(subset=["load_gw"])

    return daily_data, weekly_data


def save_processed_data(
    hourly_data: pd.DataFrame,
    daily_data: pd.DataFrame,
    weekly_data: pd.DataFrame,
) -> None:
    """
    Save the processed hourly, daily and weekly datasets.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    hourly_data.to_csv(HOURLY_PROCESSED_FILE)
    daily_data.to_csv(DAILY_PROCESSED_FILE)
    weekly_data.to_csv(WEEKLY_PROCESSED_FILE)

    print(f"Saved hourly data to: {HOURLY_PROCESSED_FILE}")
    print(f"Saved daily data to: {DAILY_PROCESSED_FILE}")
    print(f"Saved weekly data to: {WEEKLY_PROCESSED_FILE}")


def build_processed_datasets() -> dict[str, Path]:
    """
    Run the complete electricity-demand preprocessing workflow.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths to the generated processed datasets.
    """
    raw_data = load_raw_germany_electricity_data()

    hourly_data = clean_hourly_load(raw_data)
    daily_data, weekly_data = aggregate_electricity_load(hourly_data)

    save_processed_data(
        hourly_data=hourly_data,
        daily_data=daily_data,
        weekly_data=weekly_data,
    )

    print("\nDataset summary")
    print("----------------")
    print(f"Hourly observations: {len(hourly_data):,}")
    print(f"Daily observations:  {len(daily_data):,}")
    print(f"Weekly observations: {len(weekly_data):,}")
    print(f"Start date: {hourly_data.index.min()}")
    print(f"End date:   {hourly_data.index.max()}")
    print(f"Missing hourly load values: {hourly_data['load_gw'].isna().sum()}")

    return {
        "hourly": HOURLY_PROCESSED_FILE,
        "daily": DAILY_PROCESSED_FILE,
        "weekly": WEEKLY_PROCESSED_FILE,
    }