"""
EXP-004 = Optuna HP search поверх EXP-003 feature set (с bias correction).

Минимизируем CV nMAE (12 фолдов GroupKFold по month_key, direct mode).
Budget: 50 trials, ~5 минут на trial -> 4-5 часов overall на CPU.
Storage: SQLite в ~/wind_hackathon/experiments/active/EXP-004/optuna.db (resumable).
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

sys.path.insert(0, str(Path("~/wind_hackathon/pipeline").expanduser()))
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae
sys.path.insert(0, str(Path("~/wind_hackathon/pipeline/q1_task").expanduser()))
from lgbm_q50 import prepare_features, get_feature_cols
from nwp_bias_correction import (
    apply_bias, add_p_phys_corrected, estimate_bias,
)
from common.physics_features import smooth_power_curve_ti
ROOT = Path("~/wind_hackathon").expanduser()
EXP_DIR = ROOT / "experiments/active/EXP-004"
DB_PATH = EXP_DIR / "optuna.db"
BEST_PARAMS = EXP_DIR / "best_params.json"
SUMMARY = EXP_DIR / "summary.json"

N_TRIALS = 50
TIMEOUT_SEC = 3600 * 2  # 2 часа max


def prep_data_once():
    """Один раз готовим train_clean + valid + bias-corrected features."""
    print("[prep] Loading + applying bias correction...")
    train_f, valid_f = prepare_features()
    pc_smooth = smooth_power_curve_ti(ti=0.20, rated_speed=10.0)
    bias_table, _ = estimate_bias(train_f, pc_smooth)
    train_f = apply_bias(train_f, bias_table, mode="mean")
    train_f = add_p_phys_corrected(train_f, pc_smooth)
    valid_f = apply_bias(valid_f, bias_table, mode="mean")
    valid_f = add_p_phys_corrected(valid_f, pc_smooth)

    feature_cols = get_feature_cols(train_f)
    train_clean = train_f.dropna(subset=[TARGET_COL]).copy()
    for c in feature_cols:
        if train_clean[c].isna().any():
            med = train_clean[c].median()
            train_clean[c] = train_clean[c].fillna(med)
            valid_f[c] = valid_f[c].fillna(med)
    print(f"[prep] train_clean shape={train_clean.shape}, features={len(feature_cols)}")
    return train_clean, valid_f, feature_cols


# Глобальные данные (для caching между trials)
_DATA = None


def get_data():
    global _DATA
    if _DATA is None:
        _DATA = prep_data_once()
    return _DATA


def objective(trial):
    train_clean, valid_f, feature_cols = get_data()
    X = train_clean[feature_cols].values
    n_avail = train_clean["n_avail"].values
    y = train_clean[TARGET_COL].values
    groups = train_clean["month_key"].values

    sw_q1_factor = trial.suggest_float("sw_q1_factor", 1.0, 4.0, step=0.5)
    sw = np.where(train_clean["dt"].dt.month.isin([1, 2, 3]).values, sw_q1_factor, 1.0)

    params = {
        "objective": "quantile",
        "alpha": 0.5,
        "metric": "quantile",
        "verbose": -1,
        "num_threads": 0,
        "device": "cpu",
        "seed": 42,
        "num_leaves": trial.suggest_int("num_leaves", 64, 1024, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 200.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 200.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
    }
    n_boost = trial.suggest_int("n_boost", 500, 3000, step=500)

    cat_idx = [feature_cols.index("sector_8")] if "sector_8" in feature_cols else "auto"
    gkf = GroupKFold(n_splits=12)
    fold_nmae = []
    fold_q1 = []
    for tr, te in gkf.split(X, y, groups):
        ds_tr = lgb.Dataset(X[tr], label=y[tr], weight=sw[tr], categorical_feature=cat_idx)
        ds_te = lgb.Dataset(X[te], label=y[te], weight=sw[te], reference=ds_tr, categorical_feature=cat_idx)
        model = lgb.train(
            params, ds_tr,
            num_boost_round=n_boost,
            valid_sets=[ds_te],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        pred = model.predict(X[te], num_iteration=model.best_iteration)
        pred_mw = np.clip(pred, 0.0, n_avail[te] * TURBINE_RATED_MW)
        fold_nmae.append(nmae(y[te], pred_mw))
        is_q1 = train_clean["dt"].iloc[te].dt.month.isin([1, 2, 3]).values
        if is_q1.sum() > 50:
            fold_q1.append(nmae(y[te][is_q1], pred_mw[is_q1]))

    cv_mean = float(np.mean(fold_nmae))
    cv_q1 = float(np.mean(fold_q1)) if fold_q1 else cv_mean

    # Минимизируем взвешенно overall и Q1-only (Q1 веса 0.6)
    score = 0.4 * cv_mean + 0.6 * cv_q1
    trial.set_user_attr("cv_mean", cv_mean)
    trial.set_user_attr("cv_q1_only", cv_q1)
    trial.set_user_attr("cv_std", float(np.std(fold_nmae)))
    return score


def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[start] Optuna search: n_trials={N_TRIALS}, timeout={TIMEOUT_SEC}s")
    storage = f"sqlite:///{DB_PATH}"
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name="exp004_lgbm_q1",
        storage=storage,
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SEC, gc_after_trial=True)
    elapsed = time.time() - t0

    best = study.best_trial
    BEST_PARAMS.write_text(json.dumps({
        "best_score": round(best.value, 4),
        "best_cv_mean": best.user_attrs.get("cv_mean"),
        "best_cv_q1_only": best.user_attrs.get("cv_q1_only"),
        "best_cv_std": best.user_attrs.get("cv_std"),
        "best_params": best.params,
        "n_trials_completed": len(study.trials),
        "elapsed_sec": round(elapsed, 1),
    }, ensure_ascii=False, indent=2))

    print(f"\n=== EXP-004 Optuna done ({len(study.trials)} trials, {elapsed:.1f}s) ===")
    print(f"Best score: {best.value:.4f}")
    print(f"Best CV mean: {best.user_attrs.get('cv_mean'):.4f}%")
    print(f"Best CV Q1-only: {best.user_attrs.get('cv_q1_only'):.4f}%")
    print(f"Best params: {json.dumps(best.params, indent=2)}")


if __name__ == "__main__":
    main()
