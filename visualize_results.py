import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dataloading import load_regression_dataset, load_survival_dataset


def trim_results(feature_selectors: dict, metric: str) -> dict:
    """
    Remove the results that have shorter sets than the mode and calculate the mean of the metric.

    :param feature_selectors: feature selectors dictionary.
    :param metric: metric to calculate the mean.
    :return: feature selectors dictionary with the mean of the metric.
    """
    for selector, results in feature_selectors.items():
        lengths = [len(lst) for lst in results[f"{metric}_per_fold"]]
        mode = max(set(lengths), key=lengths.count)
        for inner_key, lst_of_lsts in results.items():
            equal_or_greater_than_mode = [
                lst for lst in lst_of_lsts if len(lst) >= mode
            ]
            results[inner_key] = [lst[:mode] for lst in equal_or_greater_than_mode]
        feature_selectors[selector][f"mean_{metric}"] = list(
            np.mean(results[f"{metric}_per_fold"], axis=0)
        )
    return feature_selectors


def reformat_dict(cell: dict) -> str:
    """
    Reformat the dictionary to a string with rate ± std.

    :param cell: dictionary with rate and std.
    :return: string with rate ± std.
    """
    if isinstance(cell, dict) and "rate" in cell and "std" in cell:
        rounded_rate = round(cell["rate"] * 1000, 4)
        rounded_sd = round(cell["std"] * 1000, 4)
        return f"{rounded_rate:.2f} ± {rounded_sd:.2f}"
    return cell


def load_dataset_regression(dataset: str, clf: str) -> tuple:
    """
    Load the dataset and the results of the feature selection algorithms.

    :param dataset: dataset name.
    :param clf: classifier name.
    :return: dataset and feature selectors.
    """
    X, y = load_regression_dataset(dataset)
    with open(f"regression_features/{dataset}_{clf}.json") as f:
        data = json.load(f)
    return X, trim_results(data, "mae")


def load_dataset_survival(dataset: str, clf: str) -> tuple:
    """
    Load the dataset and the results of the feature selection algorithms.

    :param dataset: dataset name.
    :param clf: classifier name.
    :return: dataset and feature selectors.
    """
    X, y = load_survival_dataset(dataset)
    with open(f"survival_features/{dataset}_{clf}.json") as f:
        data = json.load(f)
    return X, trim_results(data, "cindex")


def calculate_redundancy(X: pd.DataFrame, dataset: str, type_: str) -> dict:
    """
    Calculate the redundancy rate of the selected features.

    :param X: dataset features.
    :param dataset: dataset name.
    :param type_: type of the dataset (regression or survival).
    :return: redundancy rates of the selected features.
    """
    X = X.to_numpy()
    n_features = X.shape[1]
    with open(f"{type_}_features/{dataset}_features_selected.json") as f:
        feature_selectors = json.load(f)
    redundancy_rates = {feat_selector: {} for feat_selector in feature_selectors}
    for feat_selector, features_selected in feature_selectors.items():
        rates = []
        for subset in features_selected:
            if subset == []:
                continue
            corr = []
            for i in subset:
                for j in subset:
                    if i < j:
                        if (
                            np.unique(X[:, i]).shape[0] > 1
                            and np.unique(X[:, j]).shape[0] > 1
                        ):
                            corr.append(abs(np.corrcoef(X[:, i], X[:, j])[0, 1]))
                        else:
                            continue
            redundancy_rate = 1 / (n_features * (n_features - 1)) * sum(corr)
            rates.append(redundancy_rate)
        redundancy_rates[feat_selector]["rate"] = np.mean(rates)
        redundancy_rates[feat_selector]["std"] = np.std(rates)
    return dict(sorted(redundancy_rates.items(), key=lambda item: item[1]["rate"]))


def plot_multiplot(datasets: list[str], type_: str, clf: str, save_path: str) -> None:
    """
    Plot the results of the feature selection algorithms for multiple datasets.

    :param datasets: list of dataset names.
    :param type_: type of the dataset (regression or survival).
    :param clf: classifier name.
    :param save_path: path to save the plot.
    """
    colors = {
        "Boruta": "orange",
        "FeatBoost-X": "green",
        "XGBoost": "red",
        "RReliefF": "blue",
        "MRMR": "purple",
    }
    markers = {
        "Boruta": "s",
        "FeatBoost-X": "^",
        "XGBoost": "x",
        "RReliefF": "o",
        "MRMR": "D",
    }

    sns.set_style("whitegrid")
    rows = 3 if len(datasets) > 4 else 2
    size = (16, 18) if len(datasets) > 4 else (18, 12)
    fig, axes = plt.subplots(rows, 2, figsize=size)
    dataset_loader = (
        load_dataset_regression if type_ == "regression" else load_dataset_survival
    )
    metric = "mae" if type_ == "regression" else "cindex"
    sns.despine()
    j = 0
    for i, ds in enumerate(datasets):
        if i == 2:
            i, j = 0, 1
        elif i == 3:
            i, j = 1, 1
        elif i == 4:
            i, j = 0, 2
        X, feature_selectors = dataset_loader(ds, clf)
        for feat_selector, res in feature_selectors.items():
            axes[j][i].plot(
                range(1, len(res[f"mean_{metric}"]) + 1),
                res[f"mean_{metric}"],
                label=feat_selector,
                marker=markers[feat_selector],
                linestyle="-",
                markersize=3.5,
                color=colors[feat_selector],
            )
        n, p = X.shape
        axes[j][i].axvline(
            x=p, color="black", linestyle="--", linewidth=1, label="All Features"
        )
        axes[j][i].title.set_text(
            f"{ds.capitalize() if ds != 'msd' else 'MSD'}, $p={p}$, $n={n}$"
        )
        axes[j][i].set_xlabel("Number of Features", fontsize=12)
        axes[j][i].set_ylabel("MAE" if metric == "mae" else "C-Index", fontsize=12)
        axes[j][i].tick_params(axis="y", labelcolor="black", labelsize=10)
        lines, labels = axes[j][i].get_legend_handles_labels()
        # reorder legend
        if len(labels) > 4:
            order = [1, 0, 2, 3, 4, 5]
        elif len(labels) == 4:
            order = [1, 0, 2, 3]
        lines = [lines[i] for i in order]
        labels = [labels[i] for i in order]
        max_lim = 100 if type_ == "regression" else 40
        if type_ == "survival":
            axes[j][i].set_ylim(0.5, 0.9)

        axes[j][i].set_xlim(0, max_lim)
        axes[j][i].grid(
            color="grey", linestyle="--", linewidth=0.25, axis="both", alpha=0.3
        )
    axes[0][1].legend(
        lines,
        labels,
        fancybox=True,
        shadow=True,
        ncol=2,
        fontsize=10,
    )
    # remove empty plot
    if len(datasets) > 4:
        fig.delaxes(axes[2][1])
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    survival_datasets = ["metabric", "nacd", "nhanes", "support"]
    regression_datasets = ["crime", "diabetes", "housing", "msd", "parkinsons"]
    datasets = [regression_datasets, survival_datasets]

    for d in datasets:
        red_per_dataset = {}
        for dataset in d:
            type_ = "regression" if dataset in regression_datasets else "survival"
            X, _ = (
                load_regression_dataset(dataset)
                if type_ == "regression"
                else load_survival_dataset(dataset)
            )
            redundancy_rates = calculate_redundancy(X, dataset, type_)
            red_per_dataset[dataset] = redundancy_rates
        df = pd.DataFrame(red_per_dataset)
        df = df.T
        formatted_df = df.applymap(reformat_dict)
        print(formatted_df)

    for i, to_plot in enumerate(
        [
            ("regression", "LinearRegression"),
            ("regression", "GradientBoostingRegressor"),
            ("survival", "RandomSurvivalForest"),
            ("survival", "XGBSurvivalRegressor"),
        ]
    ):
        type_, clf = to_plot
        datasets = survival_datasets if type_ == "survival" else regression_datasets
        plot_multiplot(datasets, type_, clf, f"plots/Figure_{i+2}.pdf")
