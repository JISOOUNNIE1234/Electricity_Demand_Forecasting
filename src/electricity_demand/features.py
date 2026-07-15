from pathlib import Path

import pandas as pd

from electricity_demand.config import (
    FEATURE_DATA_FILE,
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


import numpy as np


TARGET_LAGS = [1, 2, 4, 12, 52]
ROLLING_WINDOWS = [4, 12, 52]


def add_calendar_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add calendar features that are known at the forecast origin.
    """
    result = data.copy()

    result["week_of_year"] = (
        result.index.isocalendar().week.astype(int)
    )
    result["month"] = result.index.month
    result["quarter"] = result.index.quarter
    result["year"] = result.index.year

    result["week_sin"] = np.sin(
        2.0
        * np.pi
        * result["week_of_year"]
        / 52.0
    )

    result["week_cos"] = np.cos(
        2.0
        * np.pi
        * result["week_of_year"]
        / 52.0
    )

    return result


def add_target_lag_features(
    data: pd.DataFrame,
    target_column: str = "load_gw",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add historical target values as predictor variables.

    Every lag uses only observations occurring before the prediction
    timestamp.
    """
    if lags is None:
        lags = TARGET_LAGS

    if target_column not in data.columns:
        raise ValueError(
            f"Target column not found: {target_column}"
        )

    result = data.copy()

    for lag in lags:
        result[f"load_lag_{lag}"] = (
            result[target_column].shift(lag)
        )

    return result


def add_rolling_features(
    data: pd.DataFrame,
    target_column: str = "load_gw",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add leakage-safe rolling demand features.

    The target is shifted by one week before calculating rolling
    statistics. Therefore, the current week's demand is never used
    to predict itself.
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    if target_column not in data.columns:
        raise ValueError(
            f"Target column not found: {target_column}"
        )

    result = data.copy()

    historical_target = result[target_column].shift(1)

    for window in windows:
        result[f"load_rolling_mean_{window}"] = (
            historical_target
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
        )

        result[f"load_rolling_std_{window}"] = (
            historical_target
            .rolling(
                window=window,
                min_periods=window,
            )
            .std()
        )

    return result


def add_temperature_lag_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add historical temperature variables.

    Contemporary observed temperature may be used only for a
    conditional forecast. Lagged temperature values are available
    from past observations and do not leak future information.
    """
    result = data.copy()

    temperature_columns = [
        "temp_mean",
        "temp_min",
        "temp_max",
        "heating_degree_days",
        "cooling_degree_days",
    ]

    for column in temperature_columns:
        if column not in result.columns:
            continue

        result[f"{column}_lag_1"] = (
            result[column].shift(1)
        )

        result[f"{column}_lag_2"] = (
            result[column].shift(2)
        )

    return result


def create_feature_dataset(
    model_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the complete weekly supervised-learning dataset.
    """
    features = model_data.copy()
    features = features.sort_index()

    features = add_calendar_features(features)
    features = add_target_lag_features(features)
    features = add_rolling_features(features)
    features = add_temperature_lag_features(features)

    features = features.dropna().copy()

    return features


def build_feature_dataset() -> Path:
    """
    Build and save the feature-based modelling dataset.
    """
    model_data = merge_weekly_load_and_temperature()

    feature_data = create_feature_dataset(
        model_data
    )

    FEATURE_DATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_data.to_csv(
        FEATURE_DATA_FILE
    )

    print(
        f"Saved feature dataset to: "
        f"{FEATURE_DATA_FILE}"
    )
    print(
        f"Feature observations: {len(feature_data)}"
    )
    print(
        f"Feature columns: {len(feature_data.columns)}"
    )

    return FEATURE_DATA_FILE