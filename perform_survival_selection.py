import json

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from dataloading import load_survival_dataset
from test_utils import train_boruta_survival, train_featboost, train_xgb_survival
from xgb_survival_regressor import XGBSurvivalRegressor

XGB_PARAMS = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 50,
    "subsample": 1.0,
    "colsample_bynode": 1.0,
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1,
    "tree_method": "hist",
    "booster": "gbtree",
    "grow_policy": "lossguide",
    "lambda": 0.01,
    "alpha": 0.02,
    "n_jobs": -1,
}


def run_cross_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dataset_name: str,
    feature_selectors: dict,
    params_train: dict,
) -> None:
    """
    Run cross validation for the feature selection algorithms on specific seeds.

    :param X_train: The training features.
    :param y_train: The training labels.
    :param dataset_name: The name of the dataset.
    :param feature_selectors: The dictionary containing the results of the feature selection algorithms.
    :param params_train: train parameters.
    """
    censored = np.array([0 if i[0] == i[1] else 1 for i in y_train])
    folds = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, _ in tqdm(folds.split(X, censored), total=10):  # type: ignore
        for selector, func in [
            ("Boruta", train_boruta_survival),
            ("XGBoost", train_xgb_survival),
            ("FeatBoost-X", train_featboost),
        ]:
            if func == train_featboost:
                features, importances = func(
                    X_train[train_index],
                    y_train[train_index],
                    XGBSurvivalRegressor(**params_train),
                    metric="c_index",
                )
            else:
                features, importances = func(
                    X_train[train_index],
                    y_train[train_index],
                    XGBSurvivalRegressor(**params_train),
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
                f"survival_features/{dataset_name}_features_selected.json", "w"
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
    feature_selectors = {k: [] for k in ["Boruta", "FeatBoost-X", "XGBoost"]}
    for t_rand, _ in tqdm([(42, 84), (55, 110), (875, 1750)]):
        xgb_params_train = XGB_PARAMS.copy()
        xgb_params_train["random_state"] = t_rand

        run_cross_validation(
            X_train,
            y_train,
            dataset_name,
            feature_selectors,
            xgb_params_train,
        )


if __name__ == "__main__":
    for dataset in ["metabric", "nacd", "nhanes", "support"]:
        X, y = load_survival_dataset(dataset)
        run_experiment(dataset, X.values, y.values)
