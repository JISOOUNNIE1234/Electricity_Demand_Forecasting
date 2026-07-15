from __future__ import annotations

import itertools

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def create_time_series_splits(
    number_of_splits: int = 4,
) -> TimeSeriesSplit:
    """
    Create chronological cross-validation folds.
    """
    return TimeSeriesSplit(
        n_splits=number_of_splits
    )


def evaluate_parameter_set(
    model,
    features: pd.DataFrame,
    target: pd.Series,
    splitter: TimeSeriesSplit,
) -> float:
    """
    Evaluate one model configuration using time-series validation.
    """
    fold_rmse_values = []

    for train_indices, validation_indices in splitter.split(
        features
    ):
        fold_train_x = features.iloc[train_indices]
        fold_validation_x = features.iloc[
            validation_indices
        ]

        fold_train_y = target.iloc[train_indices]
        fold_validation_y = target.iloc[
            validation_indices
        ]

        model.fit(
            fold_train_x,
            fold_train_y,
        )

        predictions = model.predict(
            fold_validation_x
        )

        rmse = mean_squared_error(
            fold_validation_y,
            predictions,
        ) ** 0.5

        fold_rmse_values.append(float(rmse))

    return float(
        sum(fold_rmse_values)
        / len(fold_rmse_values)
    )


def search_random_forest(
    training_features: pd.DataFrame,
    training_target: pd.Series,
    random_state: int = 42,
) -> tuple[
    RandomForestRegressor,
    pd.DataFrame,
]:
    """
    Tune a Random Forest using chronological validation.
    """
    parameter_grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 8, 16],
        "min_samples_leaf": [1, 3, 5],
        "max_features": ["sqrt", 0.7],
    }

    splitter = create_time_series_splits(
        number_of_splits=4
    )

    search_records = []

    combinations = itertools.product(
        parameter_grid["n_estimators"],
        parameter_grid["max_depth"],
        parameter_grid["min_samples_leaf"],
        parameter_grid["max_features"],
    )

    for (
        n_estimators,
        max_depth,
        min_samples_leaf,
        max_features,
    ) in combinations:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )

        validation_rmse = evaluate_parameter_set(
            model=model,
            features=training_features,
            target=training_target,
            splitter=splitter,
        )

        search_records.append(
            {
                "model": "random_forest",
                "n_estimators": n_estimators,
                "max_depth": (
                    "None"
                    if max_depth is None
                    else max_depth
                ),
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
                "validation_rmse": validation_rmse,
            }
        )

    results = (
        pd.DataFrame(search_records)
        .sort_values("validation_rmse")
        .reset_index(drop=True)
    )

    best = results.iloc[0]

    best_max_depth = (
        None
        if str(best["max_depth"]) == "None"
        else int(best["max_depth"])
    )

    best_model = RandomForestRegressor(
        n_estimators=int(best["n_estimators"]),
        max_depth=best_max_depth,
        min_samples_leaf=int(
            best["min_samples_leaf"]
        ),
        max_features=best["max_features"],
        random_state=random_state,
        n_jobs=-1,
    )

    best_model.fit(
        training_features,
        training_target,
    )

    return best_model, results


def search_gradient_boosting(
    training_features: pd.DataFrame,
    training_target: pd.Series,
    random_state: int = 42,
) -> tuple[
    GradientBoostingRegressor,
    pd.DataFrame,
]:
    """
    Tune a Gradient Boosting regressor using chronological folds.
    """
    parameter_grid = {
        "n_estimators": [100, 300],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [2, 3],
        "min_samples_leaf": [1, 4],
    }

    splitter = create_time_series_splits(
        number_of_splits=4
    )

    search_records = []

    combinations = itertools.product(
        parameter_grid["n_estimators"],
        parameter_grid["learning_rate"],
        parameter_grid["max_depth"],
        parameter_grid["min_samples_leaf"],
    )

    for (
        n_estimators,
        learning_rate,
        max_depth,
        min_samples_leaf,
    ) in combinations:
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            loss="squared_error",
        )

        validation_rmse = evaluate_parameter_set(
            model=model,
            features=training_features,
            target=training_target,
            splitter=splitter,
        )

        search_records.append(
            {
                "model": "gradient_boosting",
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "validation_rmse": validation_rmse,
            }
        )

    results = (
        pd.DataFrame(search_records)
        .sort_values("validation_rmse")
        .reset_index(drop=True)
    )

    best = results.iloc[0]

    best_model = GradientBoostingRegressor(
        n_estimators=int(best["n_estimators"]),
        learning_rate=float(
            best["learning_rate"]
        ),
        max_depth=int(best["max_depth"]),
        min_samples_leaf=int(
            best["min_samples_leaf"]
        ),
        random_state=random_state,
        loss="squared_error",
    )

    best_model.fit(
        training_features,
        training_target,
    )

    return best_model, results


def generate_feature_model_forecasts(
    random_forest_model: RandomForestRegressor,
    gradient_boosting_model: GradientBoostingRegressor,
    test_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate forecasts from both feature-based models.
    """
    forecasts = pd.DataFrame(
        index=test_features.index
    )

    forecasts["random_forest"] = (
        random_forest_model.predict(
            test_features
        )
    )

    forecasts["gradient_boosting"] = (
        gradient_boosting_model.predict(
            test_features
        )
    )

    return forecasts


def calculate_feature_importance(
    model,
    feature_names: list[str],
    model_name: str,
) -> pd.DataFrame:
    """
    Extract impurity-based feature importance values.
    """
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": (
                model.feature_importances_
            ),
            "model": model_name,
        }
    )

    return importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)