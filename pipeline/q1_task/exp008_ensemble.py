"""
EXP-008: ensemble LightGBM q0.5 + CatBoost MAE + XGBoost MAE с Optuna weights.

Fit 3 моделей независимо (CV bagging внутри каждой) -> OOF prediction -> подбор весов
для минимизации overall + Q1-only nMAE.

Каждая модель тренируется на EXP-007-style feature set (bias correction + lags).
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import optuna
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae, smooth_power_curve_ti
from lgbm_q50 import prepare_features, get_feature_cols, compute_segments
from nwp_bias_correction import apply_bias, add_p_phys_corrected, estimate_bias

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-008"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Optuna best params из EXP-004
study = optuna.load_study(
    study_name="exp004_lgbm_q1",
    storage="sqlite:////home/duck/wind_hackathon/experiments/active/EXP-004/optuna.db",
)
best_lgbm = study.best_params.copy()
sw_q1 = best_lgbm.pop("sw_q1_factor", 2.0)
n_boost_lgbm = best_lgbm.pop("n_boost", 2000)


def prep_data():
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
            med = tc[c].median()
            tc[c] = tc[c].fillna(med)
            valid_f[c] = valid_f[c].fillna(med)
    return tc, valid_f, cols


def fit_lgbm_cv(X, y, n_avail, dt, groups, cat_idx, sw, n_splits=12):
    oof = np.zeros(len(X))
    preds_v = []
    models = []
    hps = {
        "objective": "quantile", "alpha": 0.5, "metric": "quantile",
        "verbose": -1, "num_threads": 0, "device": "cpu", "seed": 42,
        **best_lgbm,
    }
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        ds_tr = lgb.Dataset(X[tr], label=y[tr], weight=sw[tr], categorical_feature=cat_idx)
        ds_te = lgb.Dataset(X[te], label=y[te], weight=sw[te], reference=ds_tr, categorical_feature=cat_idx)
        m = lgb.train(hps, ds_tr, num_boost_round=n_boost_lgbm, valid_sets=[ds_te],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof[te] = np.clip(m.predict(X[te], num_iteration=m.best_iteration),
                          0.0, n_avail[te] * TURBINE_RATED_MW)
        models.append(m)
    return oof, models


def fit_catboost_cv(df_X, y, n_avail, dt, groups, cat_names, sw, n_splits=12):
    """df_X - pandas DataFrame с правильными dtypes для cat features (int)."""
    oof = np.zeros(len(df_X))
    models = []
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(df_X, y, groups):
        m = cb.CatBoostRegressor(
            loss_function="MAE",
            iterations=1500,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=10.0,
            random_seed=42,
            task_type="GPU",
            devices="0",
            cat_features=cat_names,
            verbose=0,
            early_stopping_rounds=50,
        )
        m.fit(df_X.iloc[tr], y[tr], sample_weight=sw[tr],
              eval_set=(df_X.iloc[te], y[te]), use_best_model=True, verbose=0)
        oof[te] = np.clip(m.predict(df_X.iloc[te]), 0.0, n_avail[te] * TURBINE_RATED_MW)
        models.append(m)
    return oof, models


def fit_xgb_cv(X, y, n_avail, dt, groups, sw, n_splits=12):
    oof = np.zeros(len(X))
    models = []
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        m = xgb.XGBRegressor(
            objective="reg:absoluteerror",
            tree_method="hist",
            device="cuda",
            n_estimators=1500,
            learning_rate=0.05,
            max_depth=8,
            reg_lambda=10.0,
            random_state=42,
            early_stopping_rounds=50,
            verbosity=0,
        )
        m.fit(X[tr], y[tr], sample_weight=sw[tr],
              eval_set=[(X[te], y[te])], verbose=False)
        oof[te] = np.clip(m.predict(X[te]), 0.0, n_avail[te] * TURBINE_RATED_MW)
        models.append(m)
    return oof, models


def find_weights(y, oof_lgbm, oof_cat, oof_xgb, dt):
    """Optuna search для весов w_lgbm + w_cat + w_xgb = 1."""
    is_q1 = dt.dt.month.isin([1, 2, 3]).values

    def obj(trial):
        a = trial.suggest_float("a", 0.0, 1.0)
        b = trial.suggest_float("b", 0.0, 1.0 - a)
        c = 1.0 - a - b
        if c < 0:
            return 100
        pred = a * oof_lgbm + b * oof_cat + c * oof_xgb
        cv = nmae(y, pred)
        cv_q1 = nmae(y[is_q1], pred[is_q1])
        return 0.4 * cv + 0.6 * cv_q1

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=200, show_progress_bar=False)
    a = study.best_params["a"]
    b = study.best_params["b"]
    c = 1.0 - a - b
    return a, b, c, study.best_value


def main():
    t0 = time.time()
    print("[1/5] Prepare features...")
    tc, vf, cols = prep_data()
    X = tc[cols].values
    y = tc[TARGET_COL].values
    n_avail = tc["n_avail"].values
    dt = tc["dt"]
    groups = tc["month_key"].values
    cat_idx = [cols.index("sector_8")] if "sector_8" in cols else []
    sw = np.where(dt.dt.month.isin([1, 2, 3]).values, sw_q1, 1.0)

    Xv = vf[cols].values

    # CatBoost требует pandas DataFrame с int cat features
    df_X = tc[cols].copy()
    if "sector_8" in df_X.columns:
        df_X["sector_8"] = df_X["sector_8"].astype(int)
    df_Xv = vf[cols].copy()
    if "sector_8" in df_Xv.columns:
        df_Xv["sector_8"] = df_Xv["sector_8"].astype(int)
    cat_names = ["sector_8"] if "sector_8" in cols else []

    print("[2/5] Fit LightGBM (CPU, tuned HPs)...")
    oof_lgbm, models_lgbm = fit_lgbm_cv(X, y, n_avail, dt, groups, cat_idx, sw)
    n_lgbm = nmae(y, oof_lgbm)
    print(f"   LGBM CV nMAE = {n_lgbm:.4f}%")

    print("[3/5] Fit CatBoost (GPU)...")
    oof_cat, models_cat = fit_catboost_cv(df_X, y, n_avail, dt, groups, cat_names, sw)
    n_cat = nmae(y, oof_cat)
    print(f"   CatBoost CV nMAE = {n_cat:.4f}%")

    print("[4/5] Fit XGBoost (GPU)...")
    oof_xgb, models_xgb = fit_xgb_cv(X, y, n_avail, dt, groups, sw)
    n_xgb = nmae(y, oof_xgb)
    print(f"   XGBoost CV nMAE = {n_xgb:.4f}%")

    print("[5/5] Optuna weight search...")
    a, b, c, score = find_weights(y, oof_lgbm, oof_cat, oof_xgb, dt)
    oof_ens = a * oof_lgbm + b * oof_cat + c * oof_xgb
    n_ens = nmae(y, oof_ens)
    is_q1 = dt.dt.month.isin([1, 2, 3]).values
    n_ens_q1 = nmae(y[is_q1], oof_ens[is_q1])

    # CV by month for fold std
    gkf = GroupKFold(n_splits=12)
    fold_nmae = [nmae(y[te], oof_ens[te]) for _, te in gkf.split(X, y, groups)]
    fold_q1 = []
    for _, te in gkf.split(X, y, groups):
        if is_q1[te].sum() > 50:
            fold_q1.append(nmae(y[te][is_q1[te]], oof_ens[te][is_q1[te]]))

    print(f"\n=== EXP-008 Ensemble ===")
    print(f"  LGBM:    {n_lgbm:.4f}%  weight={a:.3f}")
    print(f"  CatBoost:{n_cat:.4f}%  weight={b:.3f}")
    print(f"  XGBoost: {n_xgb:.4f}%  weight={c:.3f}")
    print(f"  Ensemble CV: {n_ens:.4f}% (Q1-only {n_ens_q1:.4f}%)")
    print(f"  vs EXP-005 (9.209/8.737): {9.209 - n_ens:+.4f}/{8.737 - n_ens_q1:+.4f}")

    # Predict valid
    print("Predict valid...")
    pv_lgbm = np.zeros(len(vf))
    for m in models_lgbm:
        pv_lgbm += m.predict(Xv, num_iteration=m.best_iteration)
    pv_lgbm /= len(models_lgbm)
    pv_lgbm = np.clip(pv_lgbm, 0.0, vf["n_avail"].values * TURBINE_RATED_MW)

    pv_cat = np.mean([m.predict(df_Xv) for m in models_cat], axis=0)
    pv_cat = np.clip(pv_cat, 0.0, vf["n_avail"].values * TURBINE_RATED_MW)

    pv_xgb = np.mean([m.predict(Xv) for m in models_xgb], axis=0)
    pv_xgb = np.clip(pv_xgb, 0.0, vf["n_avail"].values * TURBINE_RATED_MW)

    pv_ens = a * pv_lgbm + b * pv_cat + c * pv_xgb

    pd.DataFrame({
        "dt": vf["dt"].values, "pred_mw": pv_ens,
        "pred_lgbm": pv_lgbm, "pred_cat": pv_cat, "pred_xgb": pv_xgb,
        "n_avail": vf["n_avail"].values,
    }).to_parquet(EXP_DIR / "valid_pred.parquet", index=False)

    pd.DataFrame({
        "dt": dt.values, "y_true": y, "oof_pred": oof_ens,
        "oof_lgbm": oof_lgbm, "oof_cat": oof_cat, "oof_xgb": oof_xgb,
    }).to_parquet(EXP_DIR / "oof.parquet", index=False)

    seg = compute_segments(y, oof_ens, tc)

    summary = {
        "exp_id": "EXP-008",
        "name": "ensemble_lgbm_catboost_xgb",
        "weights": {"lgbm": a, "catboost": b, "xgboost": c},
        "components_cv": {"lgbm": n_lgbm, "catboost": n_cat, "xgboost": n_xgb},
        "cv_mean_nmae": round(n_ens, 4),
        "cv_q1_only_mean_nmae": round(n_ens_q1, 4),
        "cv_std_nmae": round(float(np.std(fold_nmae)), 4),
        "cv_q1_std": round(float(np.std(fold_q1)), 4) if fold_q1 else 0,
        "fold_nmae": [round(x, 3) for x in fold_nmae],
        "fold_nmae_q1_only": [round(x, 3) for x in fold_q1],
        "segments": seg,
        "elapsed_sec": round(time.time() - t0, 1),
        "n_features": len(cols),
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Summary: {EXP_DIR / 'summary.json'}")
    print(f"Time: {summary['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
