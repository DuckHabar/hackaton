"""
EXP-006: Optuna HP search на GPU с большим budget.

LightGBM device='gpu' + n_jobs=4 parallel trials (A100 шарится по trials).
Используем EXP-005 feature set (bias-corrected). Budget: 200 trials.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae, smooth_power_curve_ti
from lgbm_q50 import prepare_features, get_feature_cols
from nwp_bias_correction import apply_bias, add_p_phys_corrected, estimate_bias

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-006"
EXP_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = EXP_DIR / "optuna.db"

N_TRIALS = 150
N_JOBS = 4
TIMEOUT_SEC = 3600 * 3


def prep_once():
    print("[prep] loading + bias correction...")
    train_f, valid_f = prepare_features()
    pc = smooth_power_curve_ti(ti=0.20, rated_speed=10.0)
    bt, _ = estimate_bias(train_f, pc)
    train_f = apply_bias(train_f, bt, mode="mean")
    train_f = add_p_phys_corrected(train_f, pc)
    valid_f = apply_bias(valid_f, bt, mode="mean")
    valid_f = add_p_phys_corrected(valid_f, pc)
    cols = get_feature_cols(train_f)
    tc = train_f.dropna(subset=[TARGET_COL]).copy()
    for c in cols:
        if tc[c].isna().any():
            tc[c] = tc[c].fillna(tc[c].median())
    print(f"[prep] {tc.shape}, {len(cols)} feats")
    return tc, cols


_DATA = None


def get_data():
    global _DATA
    if _DATA is None:
        _DATA = prep_once()
    return _DATA


def objective(trial):
    tc, cols = get_data()
    X = tc[cols].values
    y = tc[TARGET_COL].values
    n_avail = tc["n_avail"].values
    groups = tc["month_key"].values

    sw_q1 = trial.suggest_float("sw_q1_factor", 1.0, 4.0, step=0.5)
    sw = np.where(tc["dt"].dt.month.isin([1, 2, 3]).values, sw_q1, 1.0)

    params = {
        "objective": "quantile",
        "alpha": 0.5,
        "metric": "quantile",
        "verbose": -1,
        "seed": 42,
        "device": "gpu",
        "gpu_use_dp": False,  # single precision на A100 - быстрее
        "max_bin": 63,         # GPU loves меньшие bins
        "num_leaves": trial.suggest_int("num_leaves", 64, 2048, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 200.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 200.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 2.0),
        "path_smooth": trial.suggest_float("path_smooth", 0.0, 1.0),
    }
    n_boost = trial.suggest_int("n_boost", 500, 3000, step=500)

    cat_idx = [cols.index("sector_8")] if "sector_8" in cols else "auto"
    gkf = GroupKFold(n_splits=12)
    fold = []
    q1 = []
    for tr, te in gkf.split(X, y, groups):
        try:
            ds_tr = lgb.Dataset(X[tr], label=y[tr], weight=sw[tr], categorical_feature=cat_idx)
            ds_te = lgb.Dataset(X[te], label=y[te], weight=sw[te], reference=ds_tr, categorical_feature=cat_idx)
            m = lgb.train(params, ds_tr, num_boost_round=n_boost, valid_sets=[ds_te],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        except lgb.basic.LightGBMError as e:
            # Если GPU exhausted (parallel trials), fallback на CPU для этого fold
            params_cpu = dict(params); params_cpu["device"] = "cpu"
            params_cpu.pop("gpu_use_dp", None); params_cpu.pop("max_bin", None)
            ds_tr = lgb.Dataset(X[tr], label=y[tr], weight=sw[tr], categorical_feature=cat_idx)
            ds_te = lgb.Dataset(X[te], label=y[te], weight=sw[te], reference=ds_tr, categorical_feature=cat_idx)
            m = lgb.train(params_cpu, ds_tr, num_boost_round=n_boost, valid_sets=[ds_te],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred = m.predict(X[te], num_iteration=m.best_iteration)
        pred_mw = np.clip(pred, 0.0, n_avail[te] * TURBINE_RATED_MW)
        fold.append(nmae(y[te], pred_mw))
        is_q1 = tc["dt"].iloc[te].dt.month.isin([1, 2, 3]).values
        if is_q1.sum() > 50:
            q1.append(nmae(y[te][is_q1], pred_mw[is_q1]))

    cv = float(np.mean(fold))
    cv_q1 = float(np.mean(q1)) if q1 else cv
    score = 0.4 * cv + 0.6 * cv_q1
    trial.set_user_attr("cv_mean", cv)
    trial.set_user_attr("cv_q1_only", cv_q1)
    trial.set_user_attr("cv_std", float(np.std(fold)))
    return score


def main():
    print(f"[start] GPU Optuna {N_TRIALS} trials × {N_JOBS} parallel, timeout {TIMEOUT_SEC}s")
    storage = f"sqlite:///{DB_PATH}"
    sampler = optuna.samplers.TPESampler(seed=43)
    study = optuna.create_study(
        study_name="exp006_lgbm_gpu_q1",
        storage=storage,
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SEC, n_jobs=N_JOBS,
                   gc_after_trial=True)
    elapsed = time.time() - t0

    best = study.best_trial
    print(f"\n=== EXP-006 done ({len(study.trials)} trials, {elapsed:.1f}s) ===")
    print(f"Best score: {best.value:.4f}")
    print(f"Best cv_mean: {best.user_attrs.get('cv_mean'):.4f}%")
    print(f"Best cv_q1_only: {best.user_attrs.get('cv_q1_only'):.4f}%")

    (EXP_DIR / "best_params.json").write_text(json.dumps({
        "best_score": round(best.value, 4),
        "best_cv_mean": best.user_attrs.get("cv_mean"),
        "best_cv_q1_only": best.user_attrs.get("cv_q1_only"),
        "best_params": best.params,
        "n_trials": len(study.trials),
        "elapsed_sec": round(elapsed, 1),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
