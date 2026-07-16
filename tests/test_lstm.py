import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from electricity_demand.lstm_model import (
    create_lstm_sequences,
    create_rolling_test_sequences,
    fit_target_scaler,
    split_hourly_series,
    transform_series,
)


def test_hourly_split_uses_final_observations() -> None:
    index = pd.date_range(
        start="2015-01-01",
        periods=1000,
        freq="h",
        tz="UTC",
    )

    series = pd.Series(
        np.arange(1000, dtype=float),
        index=index,
    )

    train, test = split_hourly_series(
        series,
        test_hours=200,
    )

    assert len(train) == 800
    assert len(test) == 200
    assert train.index.max() < test.index.min()


def test_lstm_sequences_use_previous_values() -> None:
    values = np.arange(
        10,
        dtype=np.float32,
    ).reshape(-1, 1)

    features, targets = create_lstm_sequences(
        scaled_values=values,
        sequence_length=3,
    )

    assert np.array_equal(
        features[0].reshape(-1),
        np.array([0.0, 1.0, 2.0]),
    )

    assert targets[0] == 3.0


def test_scaler_is_fitted_on_training_only() -> None:
    training = pd.Series(
        [0.0, 5.0, 10.0]
    )

    test = pd.Series(
        [100.0]
    )

    scaler = fit_target_scaler(
        training
    )

    scaled_test = transform_series(
        test,
        scaler,
    )

    assert scaled_test[0, 0] > 1.0


def test_rolling_test_sequence_count() -> None:
    values = np.arange(
        20,
        dtype=np.float32,
    ).reshape(-1, 1)

    sequences = create_rolling_test_sequences(
        scaled_full_series=values,
        test_start_position=15,
        sequence_length=5,
    )

    assert len(sequences) == 5
    assert sequences.shape == (5, 5, 1)

    assert np.array_equal(
        sequences[0].reshape(-1),
        np.array(
            [10.0, 11.0, 12.0, 13.0, 14.0]
        ),
    )