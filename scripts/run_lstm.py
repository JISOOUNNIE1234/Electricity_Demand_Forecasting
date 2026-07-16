import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.config import (
    HOURLY_PROCESSED_FILE,
    HOURLY_TEST_HOURS,
    LSTM_FORECAST_FILE,
    LSTM_METRICS_FILE,
    LSTM_MODEL_FILE,
    LSTM_SEARCH_RESULTS_FILE,
    LSTM_SEQUENCE_LENGTH,
    LSTM_TRAINING_HISTORY_FILE,
    MODELS_DIR,
    RANDOM_STATE,
)
from electricity_demand.evaluation import (
    calculate_bias,
    calculate_mae,
    calculate_rmse,
)
from electricity_demand.lstm_model import (
    create_training_validation_data,
    fit_target_scaler,
    generate_lstm_forecast,
    search_lstm_hyperparameters,
    split_hourly_series,
    transform_series,
)
from electricity_demand.plotting import (
    plot_lstm_error_distribution,
    plot_lstm_hourly_forecast,
    plot_lstm_training_history,
)


def main() -> None:
    """
    Train, tune and evaluate an hourly LSTM model.
    """
    if not HOURLY_PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"Hourly data not found: "
            f"{HOURLY_PROCESSED_FILE}. "
            "Run: python scripts/make_features.py"
        )

    hourly_data = pd.read_csv(
        HOURLY_PROCESSED_FILE,
        parse_dates=["timestamp"],
        index_col="timestamp",
    ).sort_index()

    full_series = hourly_data[
        "load_gw"
    ].dropna()

    train_series, test_series = split_hourly_series(
        full_series,
        test_hours=HOURLY_TEST_HOURS,
    )

    print("\nHourly LSTM dataset")
    print("-------------------")
    print(f"Training observations: {len(train_series):,}")
    print(f"Test observations:     {len(test_series):,}")
    print(
        f"Training period: "
        f"{train_series.index.min()} to "
        f"{train_series.index.max()}"
    )
    print(
        f"Test period: "
        f"{test_series.index.min()} to "
        f"{test_series.index.max()}"
    )

    scaler = fit_target_scaler(
        train_series
    )

    scaled_training_values = transform_series(
        train_series,
        scaler,
    )

    (
        train_x,
        train_y,
        validation_x,
        validation_y,
    ) = create_training_validation_data(
        scaled_training_values=scaled_training_values,
        sequence_length=LSTM_SEQUENCE_LENGTH,
        validation_fraction=0.15,
    )

    print("\nLSTM sequence shapes")
    print("--------------------")
    print(f"Train X:      {train_x.shape}")
    print(f"Train y:      {train_y.shape}")
    print(f"Validation X: {validation_x.shape}")
    print(f"Validation y: {validation_y.shape}")

    (
        best_model,
        best_history,
        search_results,
    ) = search_lstm_hyperparameters(
        train_x=train_x,
        train_y=train_y,
        validation_x=validation_x,
        validation_y=validation_y,
        sequence_length=LSTM_SEQUENCE_LENGTH,
        random_state=RANDOM_STATE,
    )

    forecast = generate_lstm_forecast(
        model=best_model,
        full_series=full_series,
        training_length=len(train_series),
        sequence_length=LSTM_SEQUENCE_LENGTH,
        scaler=scaler,
        batch_size=256,
    )

    forecast_data = pd.DataFrame(
        {
            "actual": test_series.astype(float),
            "lstm": forecast.astype(float),
        },
        index=test_series.index,
    )

    metrics = pd.DataFrame(
        [
            {
                "model": "lstm",
                "MAE": calculate_mae(
                    forecast_data["actual"],
                    forecast_data["lstm"],
                ),
                "RMSE": calculate_rmse(
                    forecast_data["actual"],
                    forecast_data["lstm"],
                ),
                "Bias": calculate_bias(
                    forecast_data["actual"],
                    forecast_data["lstm"],
                ),
                "evaluation_frequency": "hourly",
                "forecast_design": (
                    "rolling one-step-ahead"
                ),
            }
        ]
    )

    LSTM_FORECAST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LSTM_METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast_data.to_csv(
        LSTM_FORECAST_FILE
    )

    metrics.to_csv(
        LSTM_METRICS_FILE,
        index=False,
    )

    search_results.to_csv(
        LSTM_SEARCH_RESULTS_FILE,
        index=False,
    )

    best_history.to_csv(
        LSTM_TRAINING_HISTORY_FILE,
        index=False,
    )

    best_model.save(
        LSTM_MODEL_FILE
    )

    history_plot = (
        plot_lstm_training_history(
            best_history
        )
    )

    forecast_plot = (
        plot_lstm_hourly_forecast(
            training_series=train_series,
            forecast_data=forecast_data,
        )
    )

    error_plot = (
        plot_lstm_error_distribution(
            forecast_data
        )
    )

    print("\nLSTM metrics")
    print("------------")
    print(metrics.to_string(index=False))

    print("\nBest hyperparameters")
    print("--------------------")
    print(
        search_results.head(1).to_string(
            index=False
        )
    )

    print(f"\nForecasts: {LSTM_FORECAST_FILE}")
    print(f"Metrics:   {LSTM_METRICS_FILE}")
    print(f"Model:     {LSTM_MODEL_FILE}")
    print(f"History:   {history_plot}")
    print(f"Forecast:  {forecast_plot}")
    print(f"Errors:    {error_plot}")

    print(
        "\nInterpretation note: this is a rolling "
        "one-hour-ahead evaluation across the final "
        "two years. It is not a single-origin recursive "
        "17,520-hour forecast."
    )


if __name__ == "__main__":
    main()