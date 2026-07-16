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

SARIMA_SEARCH_RESULTS_FILE = METRICS_DIR / "sarima_aic_search_results.csv"
SARIMA_BEST_MODEL_FILE = METRICS_DIR / "sarima_best_model.csv"
SARIMA_FORECAST_FILE = FORECASTS_DIR / "sarima_forecast.csv"
SARIMA_METRICS_FILE = METRICS_DIR / "sarima_metrics.csv"

SARIMA_P_VALUES = range(0, 7)
SARIMA_D_VALUES = range(0, 3)
SARIMA_Q_VALUES = range(0, 7)

SARIMA_SEASONAL_P_VALUES = range(0, 2)
SARIMA_SEASONAL_D_VALUES = range(0, 2)
SARIMA_SEASONAL_Q_VALUES = range(0, 2)

TEMPERATURE_API_URL = "https://archive-api.open-meteo.com/v1/archive"

BERLIN_LATITUDE = 52.52
BERLIN_LONGITUDE = 13.405

TEMPERATURE_START_DATE = "2015-01-01"
TEMPERATURE_END_DATE = "2020-10-06"

TEMPERATURE_RAW_FILE = RAW_DATA_DIR / "berlin_temperature_hourly.csv"
TEMPERATURE_WEEKLY_FILE = PROCESSED_DATA_DIR / "berlin_temperature_weekly.csv"
WEEKLY_MODEL_DATA_FILE = PROCESSED_DATA_DIR / "weekly_model_data.csv"

SARIMAX_FORECAST_FILE = FORECASTS_DIR / "sarimax_forecast.csv"
SARIMAX_METRICS_FILE = METRICS_DIR / "sarimax_metrics.csv"
SARIMAX_MODEL_SUMMARY_FILE = METRICS_DIR / "sarimax_model_summary.csv"

FEATURE_DATA_FILE = PROCESSED_DATA_DIR / "weekly_feature_data.csv"

FEATURE_MODEL_FORECAST_FILE = (
    FORECASTS_DIR / "feature_model_forecasts.csv"
)

FEATURE_MODEL_METRICS_FILE = (
    METRICS_DIR / "feature_model_metrics.csv"
)

FEATURE_IMPORTANCE_FILE = (
    METRICS_DIR / "feature_importance.csv"
)

FEATURE_MODEL_SEARCH_FILE = (
    METRICS_DIR / "feature_model_search_results.csv"
)

RANDOM_STATE = 42

HOURLY_TEST_HOURS = 24 * 365 * 2

LSTM_SEQUENCE_LENGTH = 24 * 7

LSTM_FORECAST_FILE = (
    FORECASTS_DIR / "lstm_hourly_forecast.csv"
)

LSTM_METRICS_FILE = (
    METRICS_DIR / "lstm_metrics.csv"
)

LSTM_SEARCH_RESULTS_FILE = (
    METRICS_DIR / "lstm_search_results.csv"
)

LSTM_TRAINING_HISTORY_FILE = (
    METRICS_DIR / "lstm_training_history.csv"
)

LSTM_MODEL_FILE = (
    MODELS_DIR / "hourly_lstm.keras"
)