# FeatBoost-Xperiments
All experiments that were ran for the paper "FeatBoost-X: Feature selection through gradient boosting for regression and survival analysis".
See the [paper]() for details.

# Pre-requisites

First install all dependencies:
```shell
pip install -r requirements.txt
```

# Running feature selection experiments

To run the feature selection experiments:
```shell
python perform_regression_selection.py
```
for regression or
```shell
python perform_survival_selection.py
```
for survival.

# Running evaluation experiments

To run the evaluation experiments:
```shell
python perform_regression_experiment.py
```
for regression or
```shell
python perform_survival_experiment.py
```
for survival.
