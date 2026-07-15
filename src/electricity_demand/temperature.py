from pathlib import Path

import pandas as pd
import requests

from electricity_demand.config import (
    BERLIN_LATITUDE,
    BERLIN_LONGITUDE,
    TEMPERATURE_API_URL,
    TEMPERATURE_END_DATE,
    TEMPERATURE_RAW_FILE,
    TEMPERATURE_START_DATE,
    TEMPERATURE_WEEKLY_FILE,
)


def download_berlin_temperature(
    overwrite: bool = False,
) -> Path:
    """
    Download hourly Berlin temperature from Open-Meteo.

    The temperature data represent historical observed/reanalysis
    values. Future test-set temperature must therefore be described
    as a conditional or explanatory covariate.
    """
    TEMPERATURE_RAW_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if TEMPERATURE_RAW_FILE.exists() and not overwrite:
        print(
            f"Temperature file already exists: "
            f"{TEMPERATURE_RAW_FILE}"
        )
        return TEMPERATURE_RAW_FILE

    parameters = {
        "latitude": BERLIN_LATITUDE,
        "longitude": BERLIN_LONGITUDE,
        "start_date": TEMPERATURE_START_DATE,
        "end_date": TEMPERATURE_END_DATE,
        "hourly": "temperature_2m",
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "models": "era5",
    }

    print("Downloading Berlin temperature data...")

    try:
        response = requests.get(
            TEMPERATURE_API_URL,
            params=parameters,
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()

    except requests.RequestException as error:
        raise RuntimeError(
            "Failed to download Berlin temperature data."
        ) from error

    if "hourly" not in payload:
        raise ValueError(
            "Open-Meteo response does not contain hourly data."
        )

    hourly_payload = payload["hourly"]

    temperature_data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                hourly_payload["time"],
                utc=True,
            ),
            "temperature_2m": pd.to_numeric(
                hourly_payload["temperature_2m"],
                errors="coerce",
            ),
        }
    )

    temperature_data.to_csv(
        TEMPERATURE_RAW_FILE,
        index=False,
    )

    print(
        f"Saved hourly temperature data to: "
        f"{TEMPERATURE_RAW_FILE}"
    )

    return TEMPERATURE_RAW_FILE


def load_hourly_temperature(
    file_path: Path = TEMPERATURE_RAW_FILE,
) -> pd.DataFrame:
    """Load downloaded hourly temperature data."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Temperature data not found: {file_path}. "
            "Run the temperature download script first."
        )

    temperature_data = pd.read_csv(
        file_path,
        parse_dates=["timestamp"],
    )

    temperature_data["timestamp"] = pd.to_datetime(
        temperature_data["timestamp"],
        utc=True,
    )

    temperature_data["temperature_2m"] = pd.to_numeric(
        temperature_data["temperature_2m"],
        errors="coerce",
    )

    temperature_data = (
        temperature_data
        .dropna(subset=["timestamp", "temperature_2m"])
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    return temperature_data


def create_weekly_temperature_features(
    hourly_temperature: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate hourly temperature to weekly weather features.
    """
    required_column = "temperature_2m"

    if required_column not in hourly_temperature.columns:
        raise ValueError(
            "Expected a temperature_2m column."
        )

    weekly_temperature = (
        hourly_temperature["temperature_2m"]
        .resample("W-SUN")
        .agg(
            temp_mean="mean",
            temp_min="min",
            temp_max="max",
        )
    )

    daily_mean = (
        hourly_temperature["temperature_2m"]
        .resample("D")
        .mean()
        .to_frame("daily_temp")
    )

    daily_mean["heating_degree_days"] = (
        18.0 - daily_mean["daily_temp"]
    ).clip(lower=0.0)

    daily_mean["cooling_degree_days"] = (
        daily_mean["daily_temp"] - 18.0
    ).clip(lower=0.0)

    weekly_degree_days = (
        daily_mean[
            [
                "heating_degree_days",
                "cooling_degree_days",
            ]
        ]
        .resample("W-SUN")
        .sum()
    )

    weekly_temperature = weekly_temperature.join(
        weekly_degree_days,
        how="left",
    )

    weekly_temperature = weekly_temperature.dropna()

    return weekly_temperature


def build_weekly_temperature_dataset() -> Path:
    """Download, process and save weekly temperature features."""
    download_berlin_temperature(overwrite=False)

    hourly_temperature = load_hourly_temperature()

    weekly_temperature = (
        create_weekly_temperature_features(
            hourly_temperature
        )
    )

    TEMPERATURE_WEEKLY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weekly_temperature.to_csv(
        TEMPERATURE_WEEKLY_FILE
    )

    print(
        f"Saved weekly temperature data to: "
        f"{TEMPERATURE_WEEKLY_FILE}"
    )
    print(
        f"Weekly temperature observations: "
        f"{len(weekly_temperature)}"
    )

    return TEMPERATURE_WEEKLY_FILE