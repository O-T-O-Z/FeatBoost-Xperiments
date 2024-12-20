import json
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from tqdm import tqdm

from dataloading import load_regression_dataset
from test_utils import run_eval


def plot_results(dataset_name: str, feature_selectors: dict) -> None:
    """
    Plot the results of the feature selection algorithms.

    :param dataset_name: name of the dataset.
    :param feature_selectors: dictionary containing the results of the feature selection algorithms.
    """
    for selector, results in feature_selectors.items():
        lengths = [len(lst) for lst in results["mae_per_fold"]]
        if not lengths:
            continue
        mode = max(set(lengths), key=lengths.count)
        for inner_key, lst_of_lsts in results.items():
            equal_or_greater_than_mode = [
                lst for lst in lst_of_lsts if len(lst) >= mode
            ]
            results[inner_key] = [lst[:mode] for lst in equal_or_greater_than_mode]
        feature_selectors[selector]["mean_mae"] = list(
            np.mean(results["mae_per_fold"], axis=0)
        )

    for selector, results in feature_selectors.items():
        if "mean_mae" not in results.keys():
            continue
        plt.plot(
            range(len(results["mean_mae"])),
            results["mean_mae"],
            label=selector,
        )
    plt.xlabel("# features")
    plt.legend()
    plt.ylabel("MAE")
    plt.xlim(0, 100)
    plt.title(f"{dataset_name} Regression Prediction")
    plt.savefig(f"regression_features/{dataset_name}.pdf")
    plt.close()


if __name__ == "__main__":
    simplefilter("ignore", category=ConvergenceWarning)

    params = {
        "learning_rate": 0.01,
        "max_depth": 4,
        "n_iter_no_change": 10,
    }
    for regressor in [LinearRegression, GradientBoostingRegressor]:
        print(f"Running {regressor}")
        for dataset in ["crime", "diabetes", "housing", "msd", "parkinsons"]:
            print(f"Running {dataset}")
            X, y = load_regression_dataset(dataset)
            with open(f"regression_features/{dataset}_features_selected.json") as f:
                data = json.load(f)

            X_train = np.array(X.values)
            y_train = np.array(y.values).ravel()
            results = {
                k: {"mae_per_fold": []}
                for k in ["Boruta", "FeatBoost-X", "RReliefF", "XGBoost", "MRMR"]
            }
            idx = 0
            for _, e_rand in tqdm([(42, 84), (55, 110), (875, 1750)]):
                print(f"Running random state {e_rand}")
                folds = KFold(n_splits=10, shuffle=True, random_state=42)
                if regressor == GradientBoostingRegressor:
                    evaluator = regressor(random_state=e_rand, **params)
                else:
                    evaluator = regressor()
                for train_index, test_index in tqdm(folds.split(X), total=10):
                    for selector in results:
                        print(f"Running {selector}")
                        features = data[selector][idx]
                        if len(features) == 0:
                            continue
                        mae = run_eval(
                            X_train[train_index],
                            X_train[test_index],
                            y_train[train_index],
                            y_train[test_index],
                            features,
                            evaluator,
                        )
                        results[selector]["mae_per_fold"].append(mae)
                        with open(
                            f"regression_features/{dataset}_{evaluator.__class__.__name__}.json",
                            "w",
                        ) as f:
                            json.dump(results, f)
                    idx += 1
            plot_results(f"{dataset}_{evaluator.__class__.__name__}", results)
