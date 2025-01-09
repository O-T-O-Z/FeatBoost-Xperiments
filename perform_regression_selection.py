import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from tqdm import tqdm
from xgboost import XGBRegressor

from dataloading import load_regression_dataset
from test_utils import (
    train_boruta,
    train_featboost,
    train_mrmr,
    train_relief,
    train_xgb,
)


def run_cross_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dataset_name: str,
    feature_selectors: dict,
    t_rand: int,
) -> None:
    """
    Run cross validation for the feature selection algorithms on specific seeds.

    :param X_train: The training features.
    :param y_train: The training labels.
    :param dataset_name: The name of the dataset.
    :param feature_selectors: The dictionary containing the results of the feature selection algorithms.
    :param t_rand: train seed.
    """
    xgb_params = {
        "n_estimators": 100,
        "max_depth": 20,
        "n_jobs": -1,
        "random_state": t_rand,
    }
    folds = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, _ in tqdm(folds.split(X), total=10):  # type: ignore
        for selector, func in [
            ("RReliefF", train_relief),
            ("Boruta", train_boruta),
            ("XGBoost", train_xgb),
            ("FeatBoost-X", train_featboost),
            ("MRMR", train_mrmr),
        ]:
            if func == train_featboost:
                features, importances = func(
                    X_train[train_index],
                    y_train[train_index],
                    [XGBRegressor(**xgb_params), LinearRegression()],
                )
            elif func == train_mrmr:
                features, importances = func(
                    pd.DataFrame(X_train[train_index]),
                    pd.DataFrame(y_train[train_index]),
                    XGBRegressor(**xgb_params),
                )
            else:
                features, importances = func(
                    X_train[train_index],
                    y_train[train_index],
                    XGBRegressor(**xgb_params),
                )

            if np.unique(importances).shape[0] != 1:
                # use the mean value of the scores as a lower threshold.
                most_important_ = np.where(importances >= np.mean(importances))[0]
                features = features[: len(most_important_)]
                # get importance of those features
                importances = importances[: len(most_important_)]

            # limit the number of features to 100
            if len(features) > 100:
                features = features[:100]
                importances = importances[:100]

            importances = [float(i) for i in list(importances)]
            features = [int(f) for f in list(features)]
            feature_selectors[selector].append(features)
            with open(
                f"regression_features/{dataset_name}_features_selected.json", "w"
            ) as f:
                json.dump(feature_selectors, f)


def run_experiment(dataset_name: str, X: np.ndarray, y: np.ndarray) -> None:
    """
    Run the experiment for the given dataset.

    :param dataset_name: The name of the dataset.
    :param X: Input features.
    :param y: Target values.
    """
    print(f"Running experiment {dataset_name}")
    X_train = np.array(X)
    y_train = np.array(y)
    feature_selectors = {
        k: [] for k in ["Boruta", "FeatBoost-X", "RReliefF", "XGBoost", "MRMR"]
    }
    for t_rand, _ in tqdm([(42, 84), (55, 110), (875, 1750)]):
        run_cross_validation(
            X_train,
            y_train,
            dataset_name,
            feature_selectors,
            t_rand,
        )

    for selector, results in feature_selectors.items():
        lengths = [len(lst) for lst in results]
        mode = max(set(lengths), key=lengths.count)
        equal_or_greater_than_mode = [
            lst if len(lst) >= mode else [] for lst in results
        ]
        trimmed = [lst[:mode] for lst in equal_or_greater_than_mode]
        feature_selectors[selector] = trimmed

    with open(f"regression_features/{dataset_name}_features_selected.json", "w") as f:
        json.dump(feature_selectors, f)


if __name__ == "__main__":
    for dataset in ["crime", "diabetes", "housing", "parkinsons", "msd"]:
        X, y = load_regression_dataset(dataset)
        run_experiment(dataset, X.values, y.values)
