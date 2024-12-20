import numpy as np
import pandas as pd
from pycox.datasets import metabric
from shap.datasets import nhanesi
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

pd.set_option("future.no_silent_downcasting", True)


def prepare_nacd() -> None:
    """Prepare the NACD dataset from the PSSP website."""
    df = pd.read_csv(
        "http://pssp.srv.ualberta.ca/system/predictors/datasets/000/000/032/original/All_Data_updated_may2011_CLEANED.csv?1350302245"  # noqa
    )
    X_nacd = df.drop(["SURVIVAL", "CENSORED"], axis=1)
    y_nacd = df[["SURVIVAL", "CENSORED"]].copy()
    y_nacd.loc[:, "lower_bound"] = y_nacd["SURVIVAL"]
    y_nacd.loc[:, "upper_bound"] = y_nacd["SURVIVAL"]
    y_nacd.loc[y_nacd["CENSORED"] == 1, "upper_bound"] = np.inf
    y_nacd = y_nacd.drop(["SURVIVAL", "CENSORED"], axis=1)
    df = pd.concat([X_nacd, y_nacd], axis=1)
    df.to_csv("datasets/nacd_cleaned.csv", index=False)


def prepare_support() -> None:
    """Prepare the SUPPORT dataset."""
    FILL_VALUES = {
        "alb": 3.5,
        "pafi": 333.3,
        "bili": 1.01,
        "crea": 1.01,
        "bun": 6.51,
        "wblc": 9.0,
        "urine": 2502.0,
    }

    TO_DROP = [
        "aps",
        "sps",
        "surv2m",
        "surv6m",
        "prg2m",
        "prg6m",
        "dnr",
        "dnrday",
        "sfdm2",
        "hospdead",
        "slos",
        "charges",
        "totcst",
        "totmcst",
    ]

    # load, drop columns, fill using specified fill values
    df = (
        pd.read_csv("raw_datasets/support2.csv")
        .drop(TO_DROP, axis=1)
        .fillna(value=FILL_VALUES)
    )
    df = pd.get_dummies(df, dummy_na=True)
    df = df.fillna(df.median())
    X_support = df.drop(["death", "d.time"], axis=1)
    X_support = X_support.replace(True, 1).replace(False, 0)

    y_support = df[["death", "d.time"]]
    y_support = y_support.copy()
    y_support.loc[:, "lower_bound"] = y_support["d.time"].astype(float)
    y_support.loc[:, "upper_bound"] = y_support["d.time"].astype(float)
    y_support.loc[y_support["death"] == 1, "upper_bound"] = np.inf
    y_support = y_support.drop(["death", "d.time"], axis=1)

    # combine the two datasets
    df = pd.concat([X_support, y_support], axis=1)
    df.to_csv("datasets/support_cleaned.csv", index=False)


def prepare_nhanes() -> None:
    """Prepare the NHANES dataset."""
    X, y = nhanesi()
    y_lower_bound = abs(y)
    y_upper_bound = np.array([np.inf if i < 0 else i for i in y])
    y = pd.DataFrame(
        np.array([y_lower_bound, y_upper_bound]).T,
        columns=["lower_bound", "upper_bound"],
        index=X.index,
    )
    # fill missing values with median
    X = X.replace(True, 1).replace(False, 0)
    X = X.fillna(X.median())

    df = pd.concat([X, y], axis=1)
    df.to_csv("datasets/nhanes_cleaned.csv", index=False)


def prepare_metabric() -> None:
    """Prepare the METABRIC dataset."""
    df = metabric.read_df()
    X = df.drop(columns=["duration", "event"])
    y = df[["duration", "event"]]
    y = y.copy()
    y["lower_bound"] = y["duration"]
    y["upper_bound"] = y["duration"]
    y.loc[y["event"] == 0, "upper_bound"] = np.inf
    y = y.drop(columns=["duration", "event"])
    df = pd.concat([X, y], axis=1)
    df.to_csv("datasets/metabric_cleaned.csv", index=False)


def prepare_diabetes() -> None:
    """Prepare the diabetes dataset."""
    data = load_diabetes()
    X = pd.DataFrame(data["data"], columns=data["feature_names"])  # type: ignore
    y = pd.DataFrame(data["target"], columns=["target"])  # type: ignore
    df = pd.concat([X, y], axis=1)
    df.to_csv("datasets/diabetes_cleaned.csv", index=False)


def prepare_housing() -> None:
    """Prepare the housing dataset."""
    data = fetch_california_housing()
    X = pd.DataFrame(data["data"], columns=data["feature_names"])  # type: ignore
    y = pd.DataFrame(data["target"], columns=data["target_names"])  # type: ignore
    df = pd.concat([X, y], axis=1)
    df.to_csv("datasets/housing_cleaned.csv", index=False)


def prepare_crime() -> None:
    """Prepare the crime dataset."""
    communities_and_crime_unnormalized = fetch_ucirepo(id=211)
    X = communities_and_crime_unnormalized.data.features  # type: ignore
    y = communities_and_crime_unnormalized.data.targets  # type: ignore
    X.loc[:, "State"] = LabelEncoder().fit_transform(X["State"])
    X = X.astype(float)
    X = X.fillna(X.median())
    y = y["violentPerPop"]
    y = y.astype(float)
    y = y.dropna()
    X = X.loc[y.index]
    df = pd.concat([X, y], axis=1)
    df.to_csv("datasets/crime_cleaned.csv", index=False)


def prepare_parkinsons() -> None:
    """Prepare the parkinsons dataset."""
    parkinsons_telemonitoring = fetch_ucirepo(id=189)
    X = parkinsons_telemonitoring.data.features  # type: ignore
    y = parkinsons_telemonitoring.data.targets["total_UPDRS"]  # type: ignore
    df = pd.concat([X, y], axis=1)
    df.to_csv("datasets/parkinsons_cleaned.csv", index=False)


def prepare_msd() -> None:
    """Prepare the Million Song Dataset."""
    data = pd.read_csv("raw_datasets/YearPredictionMSD.txt", header=None)
    y = data.pop(0)  # type: ignore
    X = data
    df = pd.concat([X, y], axis=1)
    df.sample(10000, random_state=42).to_csv("datasets/msd_cleaned.csv", index=False)


def load_regression_dataset(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a regression dataset from the datasets folder.

    :param name: The name of the dataset to load.
    :return: A tuple containing the features and the target.
    """
    df = pd.read_csv(f"datasets/{name}_cleaned.csv")
    if name == "crime":
        target = "violentPerPop"
    elif name == "diabetes":
        target = "target"
    elif name == "housing":
        target = "MedHouseVal"
    elif name == "msd":
        target = "0"
    elif name == "parkinsons":
        target = "total_UPDRS"

    X = df.drop(target, axis=1)
    y = df[[target]]
    return X, y


def load_survival_dataset(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a survival dataset from the datasets folder.

    :param name: The name of the dataset to load.
    :return: A tuple containing the features and the target.
    """
    df = pd.read_csv(f"datasets/{name}_cleaned.csv")
    X = df.drop(["lower_bound", "upper_bound"], axis=1)
    y = df[["lower_bound", "upper_bound"]]
    return X, y
