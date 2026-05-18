import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd

# конфиг variant a из q1_task/exp021_run.py
LGBM_PARAMS_VARIANT_A = {
    "objective": "quantile",
    "alpha": 0.5,
    "metric": "quantile",
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "n_estimators": 2000,
    "seed": 42,
    "verbose": -1,
    "num_threads": 0,
}


def fit_model(X, y, feat_list, X_val=None, y_val=None, params=None, sample_weight=None):
    p = dict(LGBM_PARAMS_VARIANT_A)
    if params:
        p.update(params)
    Xf = X[feat_list]
    callbacks = [lgb.log_evaluation(0)]
    model = lgb.LGBMRegressor(**p)
    if X_val is not None and y_val is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=200, verbose=False))
        model.fit(Xf, y, eval_set=[(X_val[feat_list], y_val)], callbacks=callbacks, sample_weight=sample_weight)
    else:
        model.fit(Xf, y, callbacks=callbacks, sample_weight=sample_weight)
    return model


def predict(model, X, feat_list):
    return model.predict(X[feat_list])


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
