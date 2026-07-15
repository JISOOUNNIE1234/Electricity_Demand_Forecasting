from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
FORECASTS_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"

ELECTRICITY_DATA_URL = (
    "https://data.open-power-system-data.org/time_series/"
    "2020-10-06/time_series_60min_singleindex.csv"
)

ELECTRICITY_RAW_FILE = RAW_DATA_DIR / "time_series_60min_singleindex.csv"
HOURLY_PROCESSED_FILE = PROCESSED_DATA_DIR / "germany_load_hourly.csv"
DAILY_PROCESSED_FILE = PROCESSED_DATA_DIR / "germany_load_daily.csv"
WEEKLY_PROCESSED_FILE = PROCESSED_DATA_DIR / "germany_load_weekly.csv"

START_DATE = "2015-01-01"

DATETIME_COLUMN = "utc_timestamp"

GERMANY_LOAD_COLUMN_CANDIDATES = [
    "DE_load_actual_entsoe_transparency",
    "DE_load_actual_entsoe_power_statistics",
]

TEST_HORIZON_WEEKS = 104
WEEKLY_SEASONAL_PERIOD = 52

BENCHMARK_FORECAST_FILE = FORECASTS_DIR / "benchmark_forecasts.csv"
BENCHMARK_METRICS_FILE = METRICS_DIR / "benchmark_metrics.csv"