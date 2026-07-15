from pathlib import Path

import requests

from electricity_demand.config import (
    ELECTRICITY_DATA_URL,
    ELECTRICITY_RAW_FILE,
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