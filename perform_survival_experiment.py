import json
from typing import Any
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sksurv.ensemble import RandomSurvivalForest
from tqdm import tqdm

from dataloading import load_survival_dataset
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


def plot_results(dataset_name: str, feature_selectors: dict) -> None:
    """
    Plot the results of the feature selection algorithms.

    :param dataset_name: name of the dataset.
    :param feature_selectors: dictionary containing the results of the feature selection algorithms.
    """
    for selector, results in feature_selectors.items():
        lengths = [len(lst) for lst in results["cindex_per_fold"]]
        if not lengths:
            continue
        mode = max(set(lengths), key=lengths.count)
        for inner_key, lst_of_lsts in results.items():
            equal_or_greater_than_mode = [
                lst for lst in lst_of_lsts if len(lst) >= mode
            ]
            results[inner_key] = [lst[:mode] for lst in equal_or_greater_than_mode]
        feature_selectors[selector]["mean_cindex"] = list(
            np.mean(results["cindex_per_fold"], axis=0)
        )

    for selector, results in feature_selectors.items():
        if "mean_cindex" not in results.keys():
            continue
        plt.plot(
            range(len(results["mean_cindex"])),
            results["mean_cindex"],
            label=selector,
        )
    plt.xlabel("# features")
    plt.legend()
    plt.ylabel("C-Index")
    plt.xlim(0, 100)
    plt.title(f"{dataset_name} Survival Prediction")
    plt.savefig(f"survival_features/{dataset_name}.pdf")
    plt.close()


def run_eval_survival(
    X_train: np.array,
    X_val: np.array,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    all_features: list[int],
    clf: Any,
) -> list[float]:
    """
    Run evaluation for survival data.

    :param X_train: training data.
    :param X_val: validation data.
    :param y_train: training labels.
    :param y_val: validation labels.
    :param all_features: all features.
    :param clf: classifier.
    :return: the C-index per added feature.
    """
    eval_features = []
    cindex_per_added_feature = []
    if clf.__class__.__name__ == "RandomSurvivalForest":
        y_train = np.array(
            [(event, time) for event, time in zip(y_train[:, 0], y_train[:, 2])],
            dtype=[("event", "bool"), ("time", "float64")],
        )
    for feature in all_features:
        eval_features.append(feature)
        clf.fit(X_train[:, eval_features], y_train)
        y_pred = clf.predict(X_val[:, eval_features])
        cindex = concordance_index(
            y_val[:, 0], y_pred, (y_val[:, 0] == y_val[:, 1]).astype(int)
        )
        cindex_per_added_feature.append(cindex)
    return cindex_per_added_feature


if __name__ == "__main__":
    simplefilter("ignore", category=ConvergenceWarning)
    for regressor in [RandomSurvivalForest]:  # TODO: Add XGBSurvivalRegressor
        print(f"Running {regressor}")
        for dataset in ["metabric", "nacd", "nhanes", "support"]:
            print(f"Running {dataset}")
            X, y = load_survival_dataset(dataset)
            y["event"] = (y.loc[:, "lower_bound"] == y.loc[:, "upper_bound"]).astype(
                bool
            )

            with open(f"survival_features/{dataset}_features_selected.json") as f:
                data = json.load(f)

            X_train = np.array(X.values)
            y_train = np.array(y.values)
            results = {
                k: {"cindex_per_fold": []} for k in ["Boruta", "FeatBoost-X", "XGBoost"]
            }
            idx = 0
            for _, e_rand in tqdm([(42, 84), (55, 110), (875, 1750)]):
                print(f"Running random state {e_rand}")
                folds = KFold(n_splits=10, shuffle=True, random_state=42)
                if regressor == XGBSurvivalRegressor:
                    evaluator = regressor(random_state=e_rand, **XGB_PARAMS)
                else:
                    evaluator = regressor(random_state=e_rand)
                for train_index, test_index in tqdm(folds.split(X), total=10):  # type: ignore
                    # standardize the data
                    X_train[train_index] = (
                        X_train[train_index] - X_train[train_index].mean()
                    ) / X_train[train_index].std()
                    X_train[test_index] = (
                        X_train[test_index] - X_train[train_index].mean()
                    ) / X_train[train_index].std()
                    for selector in results:
                        print(f"Running {selector}")
                        features = data[selector][idx]
                        if len(features) == 0:
                            continue
                        cindex = run_eval_survival(
                            X_train[train_index],
                            X_train[test_index],
                            y_train[train_index],
                            y_train[test_index],
                            features,
                            evaluator,
                        )
                        results[selector]["cindex_per_fold"].append(cindex)
                        with open(
                            f"survival_features/{dataset}_{evaluator.__class__.__name__}.json",
                            "w",
                        ) as f:
                            json.dump(results, f)
                    idx += 1
            plot_results(f"{dataset}_{evaluator.__class__.__name__}", results)
