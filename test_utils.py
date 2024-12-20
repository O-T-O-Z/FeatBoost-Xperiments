import numpy as np
import pandas as pd
from boruta import BorutaPy
from featboostx import FeatBoostRegressor
from lifelines.utils import concordance_index
from mrmr import mrmr_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from skrebate import ReliefF
from xgboost import XGBRegressor

pd.options.mode.chained_assignment = None


def train_boruta(X, y, clf):
    boruta_feature_selector = BorutaPy(
        estimator=clf,
        n_estimators="auto",  # type: ignore
        verbose=0,
        random_state=0,
        max_iter=100,
    )
    boruta_feature_selector.fit(X, y)
    features = [x for x in range(X.shape[1])]

    if not any(boruta_feature_selector.support_):
        return features, np.array([1 / X.shape[1] for _ in range(X.shape[1])])
    features = [x for x in range(X.shape[1]) if boruta_feature_selector.support_[x]]

    post_ranker = clf.fit(X[:, boruta_feature_selector.support_], y)

    imp_per_feat = [(k, v) for k, v in zip(features, post_ranker.feature_importances_)]
    imp_per_feat = sorted(imp_per_feat, key=lambda x: x[1], reverse=True)
    features = [x[0] for x in imp_per_feat]
    importances = [x[1] for x in imp_per_feat]
    return features, np.array(importances)


def train_boruta_survival(X, y, clf):
    boruta_feature_selector = BorutaPy(
        estimator=clf,
        n_estimators="auto",  # type: ignore
        verbose=0,
        random_state=0,
        max_iter=100,
    )
    boruta_feature_selector.fit(X, y)
    features = [x for x in range(X.shape[1])]
    if not any(boruta_feature_selector.support_):
        return features, np.array([1 / X.shape[1] for _ in range(X.shape[1])])
    features = [x for x in range(X.shape[1]) if boruta_feature_selector.support_[x]]
    post_ranker = clf.fit(X[:, boruta_feature_selector.support_], y)

    fscore = post_ranker.get_score(importance_type="gain")  # type: ignore
    feature_importance = np.zeros(X.shape[1])
    for k, v in fscore.items():
        feature_importance[int(k[1:])] = v
    imp_per_feat = [(k, v) for k, v in zip(range(X.shape[1]), feature_importance)]
    imp_per_feat = sorted(imp_per_feat, key=lambda x: x[1], reverse=True)
    features = [x[0] for x in imp_per_feat]
    importances = [x[1] for x in imp_per_feat]
    return features, np.array(importances)


def train_relief(X, y, placeholder=None):
    y = y.flatten()
    X = X.astype(np.float64)
    clf = ReliefF(n_features_to_select=X.shape[1], n_neighbors=10, n_jobs=-1)
    clf.fit(X, y)
    imp_per_feat = [(k, v) for k, v in zip(range(X.shape[1]), clf.feature_importances_)]
    imp_per_feat = sorted(imp_per_feat, key=lambda x: x[1], reverse=True)
    features = [x[0] for x in imp_per_feat]
    importances = [x[1] for x in imp_per_feat]
    return features, np.array(importances)


def train_featboost(
    X,
    y,
    estimators=[
        XGBRegressor(n_estimators=100, max_depth=20, n_jobs=-1),
        LinearRegression(),
    ],
    metric="mae",
):
    n_features = X.shape[1] if X.shape[1] < 50 else 50
    feature_selector = FeatBoostRegressor(
        estimators,
        loss="adaptive",
        metric=metric,
        verbose=0,
        number_of_folds=5,
        siso_ranking_size=n_features,
        max_number_of_features=100,
        siso_order=1,
        num_resets=1,
        epsilon=1e-10,
    )
    feature_selector.fit(X, y)
    return feature_selector.selected_subset_, np.array(
        [
            1 / len(feature_selector.selected_subset_)
            for _ in range(len(feature_selector.selected_subset_))
        ]
    )


def train_xgb(X, y, clf):
    clf.fit(X, y)
    imp_per_feat = [(k, v) for k, v in zip(range(X.shape[1]), clf.feature_importances_)]
    imp_per_feat = sorted(imp_per_feat, key=lambda x: x[1], reverse=True)
    features = [x[0] for x in imp_per_feat]
    importances = [x[1] for x in imp_per_feat]
    return features, np.array(importances)


def train_mrmr(X, y, clf):
    features = mrmr_regression(X, y, K=X.shape[1])
    return features, np.array([1 / X.shape[1] for _ in range(X.shape[1])])


def train_xgb_survival(X, y, clf):
    post_ranker = clf.fit(X, y)

    fscore = post_ranker.get_score(importance_type="gain")  # type: ignore
    feature_importance = np.zeros(X.shape[1])
    for k, v in fscore.items():
        feature_importance[int(k[1:])] = v
    imp_per_feat = [(k, v) for k, v in zip(range(X.shape[1]), feature_importance)]
    imp_per_feat = sorted(imp_per_feat, key=lambda x: x[1], reverse=True)
    features = [x[0] for x in imp_per_feat]
    importances = [x[1] for x in imp_per_feat]
    return features, np.array(importances)


def run_eval(X_train, X_val, y_train, y_val, all_features, evaluator):
    eval_features = []
    mae_per_added_feature = []
    for feature in all_features:
        eval_features.append(feature)
        evaluator.fit(X_train[:, eval_features], y_train)
        mae = mean_absolute_error(y_val, evaluator.predict(X_val[:, eval_features]))
        mae_per_added_feature.append(mae)
    return mae_per_added_feature


def run_eval_survival(X_train, X_val, y_train, y_val, all_features, clf):
    eval_features = []
    cindex_per_added_feature = []
    for feature in all_features:
        eval_features.append(feature)
        clf.fit(X_train[:, eval_features], y_train)
        y_pred = clf.predict(X_val[:, eval_features])
        if clf.get_params()["objective"] == "survival:cox":
            y_pred = -y_pred
        cindex = concordance_index(
            y_val[:, 0], y_pred, (y_val[:, 0] == y_val[:, 1]).astype(int)
        )
        cindex_per_added_feature.append(cindex)
    return cindex_per_added_feature
