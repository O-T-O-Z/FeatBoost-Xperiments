import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dataloading import load_regression_dataset, load_survival_dataset

pd.set_option("display.max_columns", None)


def trim_results(feature_selectors: dict, metric: str) -> dict:
    """
    Remove the results that have shorter sets than the mode and calculate the mean of the metric.

    :param feature_selectors: feature selectors dictionary.
    :param metric: metric to calculate the mean.
    :return: feature selectors dictionary with the mean of the metric.
    """
    for selector, results in feature_selectors.items():
        if not results:
            feature_selectors[selector] = []
            continue
        lengths = [len(lst) for lst in results[f"{metric}_per_fold"]]
        if lengths == []:
            feature_selectors[selector] = []
            continue
        mode = max(set(lengths), key=lengths.count)
        for inner_key, lst_of_lsts in results.items():
            if not isinstance(lst_of_lsts[0], list):
                continue
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


def plot_multiplot(
    datasets: list[str], type_: str, clf: str, save_path: str, metric: str
) -> None:
    """
    Plot the results of the feature selection algorithms for multiple datasets.

    :param datasets: list of dataset names.
    :param type_: type of the dataset (regression or survival).
    :param clf: classifier name.
    :param save_path: path to save the plot.
    """
    colors = {
        "Boruta": "orange",
        "SHAPBoost-C": "darkolivegreen",
        "SHAPBoost": "teal",
        "SHAPBoost (XGB)": "lightgreen",
        "SHAPBoost (RSF)": "lightgreen",
        "XGBoost": "darkred",
        "RReliefF": "cornflowerblue",
        "MRMR": "purple",
        "P-value": "orchid",
        "Forward": "gray",
        "Backward": "gold",
    }
    markers = {
        "Boruta": "s",
        "SHAPBoost-C": "*",
        "SHAPBoost": "*",
        "SHAPBoost (XGB)": "*",
        "SHAPBoost (RSF)": "*",
        "XGBoost": "x",
        "RReliefF": "o",
        "MRMR": "D",
        "P-value": "^",
        "Forward": ">",
        "Backward": "<",
    }

    sns.set_style("whitegrid")
    rows = ((len(datasets) - 1) // 2) + 1
    size = (16, 18) if len(datasets) > 4 else (18, 12)
    fig, axes = plt.subplots(rows, 2, figsize=size)
    dataset_loader = (
        load_dataset_regression if type_ == "regression" else load_dataset_survival
    )
    axes = axes.flatten()
    sns.despine()
    for i, ds in enumerate(datasets):
        X, feature_selectors = dataset_loader(ds, clf)
        if metric in ["ibs", "mean_auc"]:
            feature_selectors = trim_results(feature_selectors, metric)

        for feat_selector, res in feature_selectors.items():
            # if not feat_selector in ["SHAPBoost-C", "SHAPBoost", "SHAPBoost (XGB)", "SHAPBoost (RSF)"]:
            #     continue
            if res == []:
                continue
            axes[i].plot(
                range(1, len(res[f"mean_{metric}"]) + 1),
                res[f"mean_{metric}"],
                label=feat_selector,
                marker=markers[feat_selector],
                linestyle="-",
                markersize=2.5,
                linewidth=0.5,
                color=colors[feat_selector],
            )
        n, p = X.shape
        axes[i].axvline(
            x=p, color="black", linestyle="--", linewidth=1, label="All Features"
        )
        shapboost_len = len(feature_selectors["SHAPBoost"][f"mean_{metric}"])
        axes[i].axvline(
            x=shapboost_len,
            color="teal",
            linestyle="dotted",
            linewidth=1,
            label="SHAPBoost",
        )

        if ds == "metabric_regression":
            ds = "METABRIC"
        axes[i].title.set_text(
            f"{ds.capitalize() if ds != 'msd' else 'MSD'}, $p={p}$, $n={n}$"
        )
        axes[i].set_xlabel("Number of Features", fontsize=12)

        if metric == "mae":
            label = "MAE"
        elif metric == "cindex":
            label = "C-Index"
        elif metric == "ibs":
            label = "IBS"
        elif metric == "mean_auc":
            label = "AUC"
        axes[i].set_ylabel(label, fontsize=12)
        axes[i].tick_params(axis="y", labelcolor="black", labelsize=10)
        lines, labels = axes[i].get_legend_handles_labels()
        # reorder legend such that SHAPBoost is first
        for lab in ["SHAPBoost (XGB)", "SHAPBoost (RSF)", "SHAPBoost-C", "SHAPBoost"]:
            if lab in labels:
                idx = labels.index(lab)
                lines.insert(0, lines.pop(idx))
                labels.insert(0, labels.pop(idx))

        max_lim = 100 if p > 100 else p
        if type_ == "survival":
            if metric in ["ibs", "mean_auc"]:
                axes[i].set_ylim(0.0, 1.0)
            else:
                axes[i].set_ylim(0.5, 0.9)

        axes[i].set_xlim(1, max_lim + 1)
        axes[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axes[i].grid(
            color="grey", linestyle="--", linewidth=0.25, axis="both", alpha=0.3
        )
    axes[1].legend(
        lines,
        labels,
        fancybox=True,
        shadow=True,
        ncol=2,
        fontsize=10,
    )
    # remove empty plot
    if len(datasets) > 6:
        fig.delaxes(axes[7])
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    survival_datasets = [
        "metabric_full",
        "breast_cancer",
        "nhanes",
        "support",
        "nacd",
        "aids",
        "whas500",
    ]
    regression_datasets = [
        "metabric_regression",
        "eyedata",
        "crime",
        "msd",
        "parkinsons",
        "diabetes",
        "housing",
    ]
    datasets = [regression_datasets, survival_datasets]

    for i, to_plot in enumerate(
        [
            ("regression", "LinearRegression"),
            ("regression", "GradientBoostingRegressor"),
            ("survival", "RandomSurvivalForest"),
            ("survival", "XGBSurvivalRegressor"),
            ("survival", "CoxPHFitter"),
        ]
    ):
        type_, clf = to_plot
        datasets = survival_datasets if type_ == "survival" else regression_datasets
        metric = "mae" if type_ == "regression" else "cindex"
        plot_multiplot(datasets, type_, clf, f"plots/Figure_{i+2}_final.pdf", metric)
