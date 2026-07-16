from __future__ import annotations

import gc
import itertools
import random
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.optimizers import Adam


def set_random_seeds(
    random_state: int = 42,
) -> None:
    """
    Set random seeds for reproducible neural-network training.
    """
    random.seed(random_state)
    np.random.seed(random_state)
    tf.random.set_seed(random_state)


def split_hourly_series(
    series: pd.Series,
    test_hours: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Split an hourly time series chronologically.

    The final test_hours observations form the test period.
    """
    clean_series = (
        series
        .dropna()
        .sort_index()
        .astype(float)
    )

    if len(clean_series) <= test_hours:
        raise ValueError(
            "Hourly series is shorter than the requested "
            "test period."
        )

    train = clean_series.iloc[:-test_hours].copy()
    test = clean_series.iloc[-test_hours:].copy()

    return train, test


def fit_target_scaler(
    training_series: pd.Series,
) -> MinMaxScaler:
    """
    Fit a MinMax scaler using the training target only.
    """
    scaler = MinMaxScaler(
        feature_range=(0.0, 1.0)
    )

    scaler.fit(
        training_series.to_numpy().reshape(-1, 1)
    )

    return scaler


def transform_series(
    series: pd.Series,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """
    Transform a target series using a fitted scaler.
    """
    transformed = scaler.transform(
        series.to_numpy().reshape(-1, 1)
    )

    return transformed.astype(np.float32)


def create_lstm_sequences(
    scaled_values: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a univariate series into sequence-to-one samples.

    The input contains the previous sequence_length observations.
    The target is the immediately following observation.
    """
    if scaled_values.ndim == 1:
        scaled_values = scaled_values.reshape(-1, 1)

    if len(scaled_values) <= sequence_length:
        raise ValueError(
            "Series must contain more values than the "
            "sequence length."
        )

    features = []
    targets = []

    for index in range(
        sequence_length,
        len(scaled_values),
    ):
        features.append(
            scaled_values[
                index - sequence_length:index
            ]
        )

        targets.append(
            scaled_values[index, 0]
        )

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def create_training_validation_data(
    scaled_training_values: np.ndarray,
    sequence_length: int,
    validation_fraction: float = 0.15,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Create chronological training and validation sequences.
    """
    features, targets = create_lstm_sequences(
        scaled_values=scaled_training_values,
        sequence_length=sequence_length,
    )

    validation_size = int(
        len(features) * validation_fraction
    )

    if validation_size < 1:
        raise ValueError(
            "Validation set would contain no observations."
        )

    split_position = len(features) - validation_size

    train_x = features[:split_position]
    train_y = targets[:split_position]

    validation_x = features[split_position:]
    validation_y = targets[split_position:]

    return (
        train_x,
        train_y,
        validation_x,
        validation_y,
    )


def build_lstm_model(
    sequence_length: int,
    units: int,
    number_of_layers: int,
    dropout_rate: float,
    learning_rate: float,
) -> tf.keras.Model:
    """
    Build a sequence-to-one LSTM forecasting model.
    """
    if number_of_layers not in {1, 2}:
        raise ValueError(
            "number_of_layers must be 1 or 2."
        )

    model = Sequential()
    model.add(
        Input(
            shape=(sequence_length, 1)
        )
    )

    if number_of_layers == 1:
        model.add(
            LSTM(
                units=units,
                return_sequences=False,
            )
        )

    else:
        model.add(
            LSTM(
                units=units,
                return_sequences=True,
            )
        )
        model.add(
            Dropout(dropout_rate)
        )
        model.add(
            LSTM(
                units=max(units // 2, 8),
                return_sequences=False,
            )
        )

    model.add(
        Dropout(dropout_rate)
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model


def train_lstm_candidate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    sequence_length: int,
    units: int,
    number_of_layers: int,
    dropout_rate: float,
    learning_rate: float,
    batch_size: int,
    maximum_epochs: int = 30,
    random_state: int = 42,
) -> tuple[
    tf.keras.Model,
    pd.DataFrame,
    float,
]:
    """
    Train one LSTM configuration with early stopping.
    """
    set_random_seeds(random_state)

    model = build_lstm_model(
        sequence_length=sequence_length,
        units=units,
        number_of_layers=number_of_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        min_delta=1e-5,
    )

    history = model.fit(
        train_x,
        train_y,
        validation_data=(
            validation_x,
            validation_y,
        ),
        epochs=maximum_epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=1,
    )

    history_table = pd.DataFrame(
        history.history
    )

    best_validation_loss = float(
        history_table["val_loss"].min()
    )

    return (
        model,
        history_table,
        best_validation_loss,
    )


def search_lstm_hyperparameters(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    sequence_length: int,
    random_state: int = 42,
) -> tuple[
    tf.keras.Model,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Evaluate a compact LSTM hyperparameter grid.

    The search is intentionally limited because hourly neural
    training is computationally expensive.
    """
    parameter_grid = {
        "units": [32, 64],
        "number_of_layers": [1, 2],
        "dropout_rate": [0.1, 0.2],
        "learning_rate": [0.001],
        "batch_size": [64],
    }

    combinations = list(
        itertools.product(
            parameter_grid["units"],
            parameter_grid["number_of_layers"],
            parameter_grid["dropout_rate"],
            parameter_grid["learning_rate"],
            parameter_grid["batch_size"],
        )
    )

    search_records: list[dict[str, Any]] = []

    best_model = None
    best_history = None
    best_validation_loss = np.inf

    total_candidates = len(combinations)

    for candidate_number, (
        units,
        number_of_layers,
        dropout_rate,
        learning_rate,
        batch_size,
    ) in enumerate(
        combinations,
        start=1,
    ):
        print(
            f"\nTraining LSTM candidate "
            f"{candidate_number}/{total_candidates}"
        )

        print(
            {
                "units": units,
                "number_of_layers": number_of_layers,
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            }
        )

        model, history, validation_loss = (
            train_lstm_candidate(
                train_x=train_x,
                train_y=train_y,
                validation_x=validation_x,
                validation_y=validation_y,
                sequence_length=sequence_length,
                units=units,
                number_of_layers=number_of_layers,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                batch_size=batch_size,
                maximum_epochs=30,
                random_state=random_state,
            )
        )

        search_records.append(
            {
                "units": units,
                "number_of_layers": number_of_layers,
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs_completed": len(history),
                "best_validation_loss": validation_loss,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model = model
            best_history = history.copy()
        else:
            del model

        tf.keras.backend.clear_session()
        gc.collect()

    if best_model is None or best_history is None:
        raise RuntimeError(
            "No LSTM model was trained successfully."
        )

    search_results = (
        pd.DataFrame(search_records)
        .sort_values("best_validation_loss")
        .reset_index(drop=True)
    )

    return (
        best_model,
        best_history,
        search_results,
    )


def create_rolling_test_sequences(
    scaled_full_series: np.ndarray,
    test_start_position: int,
    sequence_length: int,
) -> np.ndarray:
    """
    Create rolling test inputs using only observations before each target.

    This supports one-step-ahead rolling evaluation across the held-out
    test period.
    """
    sequences = []

    for target_position in range(
        test_start_position,
        len(scaled_full_series),
    ):
        sequence_start = (
            target_position - sequence_length
        )

        if sequence_start < 0:
            raise ValueError(
                "Insufficient history for the requested sequence."
            )

        sequences.append(
            scaled_full_series[
                sequence_start:target_position
            ]
        )

    return np.asarray(
        sequences,
        dtype=np.float32,
    )


def generate_lstm_forecast(
    model: tf.keras.Model,
    full_series: pd.Series,
    training_length: int,
    sequence_length: int,
    scaler: MinMaxScaler,
    batch_size: int = 256,
) -> pd.Series:
    """
    Generate rolling one-step-ahead forecasts over the test period.
    """
    scaled_full_series = transform_series(
        full_series,
        scaler,
    )

    test_sequences = create_rolling_test_sequences(
        scaled_full_series=scaled_full_series,
        test_start_position=training_length,
        sequence_length=sequence_length,
    )

    scaled_predictions = model.predict(
        test_sequences,
        batch_size=batch_size,
        verbose=1,
    )

    predictions = scaler.inverse_transform(
        scaled_predictions.reshape(-1, 1)
    ).reshape(-1)

    forecast_index = full_series.index[
        training_length:
    ]

    return pd.Series(
        predictions,
        index=forecast_index,
        name="lstm",
        dtype=float,
    )