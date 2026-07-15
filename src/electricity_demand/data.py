from pathlib import Path
import pandas as pd
import requests

from electricity_demand.config import (
    DATETIME_COLUMN,
    ELECTRICITY_DATA_URL,
    ELECTRICITY_RAW_FILE,
    GERMANY_LOAD_COLUMN_CANDIDATES,
    RAW_DATA_DIR,
)


def download_file(
    url: str,
    destination: Path,
    overwrite: bool = False,
) -> Path:
    """
    Download a file from a URL.

    Parameters
    ----------
    url:
        Source URL.
    destination:
        Local path where the file will be saved.
    overwrite:
        Download the file again when it already exists.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        print(f"File already exists: {destination}")
        return destination

    print(f"Downloading data from:\n{url}")
    print(f"Saving to:\n{destination}")

    try:
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()

            total_bytes = int(response.headers.get("content-length", 0))
            downloaded_bytes = 0

            with destination.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue

                    file.write(chunk)
                    downloaded_bytes += len(chunk)

                    if total_bytes:
                        percentage = downloaded_bytes / total_bytes * 100
                        print(
                            f"\rDownloaded: {percentage:6.2f}%",
                            end="",
                            flush=True,
                        )

        print("\nDownload completed successfully.")
        return destination

    except requests.RequestException as error:
        if destination.exists():
            destination.unlink()

        raise RuntimeError(
            f"Failed to download data from {url}"
        ) from error


def download_electricity_data(overwrite: bool = False) -> Path:
    """
    Download the Open Power System Data hourly time-series file.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    return download_file(
        url=ELECTRICITY_DATA_URL,
        destination=ELECTRICITY_RAW_FILE,
        overwrite=overwrite,
    )

def identify_germany_load_column(csv_path: Path) -> str:
    """
    Identify the German electricity load column in the OPSD dataset.

    Parameters
    ----------
    csv_path:
        Path to the raw OPSD CSV file.

    Returns
    -------
    str
        Name of the German electricity load column.

    Raises
    ------
    ValueError
        If no suitable German load column is found.
    """
    available_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()

    for candidate in GERMANY_LOAD_COLUMN_CANDIDATES:
        if candidate in available_columns:
            return candidate

    possible_columns = [
        column
        for column in available_columns
        if column.startswith("DE_") and "load_actual" in column
    ]

    if possible_columns:
        return possible_columns[0]

    raise ValueError(
        "Could not identify a German electricity load column. "
        f"Available German columns include: "
        f"{[column for column in available_columns if column.startswith('DE_')]}"
    )


def load_raw_germany_electricity_data(
    csv_path: Path = ELECTRICITY_RAW_FILE,
) -> pd.DataFrame:
    """
    Load the timestamp and German electricity demand columns.

    Only the required columns are loaded to reduce memory usage.

    Parameters
    ----------
    csv_path:
        Path to the raw OPSD CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the timestamp and German load.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw electricity data not found at {csv_path}. "
            "Run: python scripts/download_data.py"
        )

    load_column = identify_germany_load_column(csv_path)

    print(f"Using German load column: {load_column}")

    data = pd.read_csv(
        csv_path,
        usecols=[DATETIME_COLUMN, load_column],
        parse_dates=[DATETIME_COLUMN],
    )

    data = data.rename(
        columns={
            DATETIME_COLUMN: "timestamp",
            load_column: "load_mw",
        }
    )

    return data