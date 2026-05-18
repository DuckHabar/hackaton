"""
EXP-011: XGBoost MAE GPU на EXP-009 features (alternate model для ensemble).

GPU job: пока LightGBM/Catboost crunch'ит CPU, XGBoost тренируется на GPU.
Цель: дать вариативность в ensemble - XGBoost ловит другие interaction patterns.

CV-only. Submit только если LB лучше EXP-009.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae, smooth_power_curve_ti
from lgbm_q50 import prepare_features, compute_segments
from nwp_bias_correction import apply_bias, add_p_phys_corrected, estimate_bias

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-011"
EXP_DIR.mkdir(parents=True, exist_ok=True)

exp005_summary = json.load(open(ROOT / "experiments/archive/rejected/worse_cv/EXP-005/summary.json"))
top15 = [f["feat"] for f in exp005_summary["top15_importance"][:15]]
must_have = [
    "wind_speed_120m", "wind_speed_80m", "wind_speed_180m", "wind_speed_10m",
    "ws_corrected_120", "ws_corrected_120_corr", "p_phys", "p_phys_corrected",
    "p_phys_no_density", "rho_air", "alpha_80_120",
    "wind_gusts_10m", "rews_simple", "rews_corr",
    "ws_at_84", "wd_120_sin", "wd_120_cos", "sector_8",
    "n_avail", "month", "hour_of_day", "temperature_80m",
    "pressure_msl", "available_frac", "icing_risk",
]
TOP_FEATS = list(dict.fromkeys(top15 + must_have))

EXTRA_LAG_HOURS = [-12, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 12]
EXTRA_LAG_COLS = ["wind_speed_120m", "ws_corrected_120", "wind_gusts_10m",
                  "wd_120_sin", "wd_120_cos", "alpha_80_120", "p_phys_corrected"]


def add_extra_lags(df_all):
    df_all = df_all.sort_values("dt").reset_index(drop=True)
    for col in EXTRA_LAG_COLS:
        if col not in df_all.columns:
            continue
        for h in EXTRA_LAG_HOURS:
            sign = "p" if h > 0 else "m"
            name = f"{col}_xlag_{sign}{abs(h)}"
            if name not in df_all.columns:
                df_all[name] = df_all[col].shift(-h)
    return df_all


def main():
    t0 = time.time()
    train_f, valid_f = prepare_features()
    pc = smooth_power_curve_ti(ti=0.20, rated_speed=10.0)
    bt, _ = estimate_bias(train_f, pc)
    train_f = apply_bias(train_f, bt, mode="mean")
    train_f = add_p_phys_corrected(train_f, pc)
    valid_f = apply_bias(valid_f, bt, mode="mean")
    valid_f = add_p_phys_corrected(valid_f, pc)

    combined = pd.concat([train_f.assign(_src="train"), valid_f.assign(_src="valid")], ignore_index=True)
    combined = add_extra_lags(combined)
    train_f = combined[combined["_src"] == "train"].drop(columns=["_src"]).copy()
    valid_f = combined[combined["_src"] == "valid"].drop(columns=["_src"]).copy()

    n_repair_col = "Кол-во_ВЭУ_в_ремонте"
    n_repair = train_f[n_repair_col]
    train_f_filtered = train_f[n_repair <= 4].copy()  # как EXP-009

    extra_lag_names = [f"{c}_xlag_{('p' if h > 0 else 'm')}{abs(h)}"
                       for c in EXTRA_LAG_COLS for h in EXTRA_LAG_HOURS]
    all_cols = list(dict.fromkeys(TOP_FEATS + extra_lag_names))
    final_cols = [c for c in all_cols if c in train_f_filtered.columns]

    train_clean = train_f_filtered.dropna(subset=[TARGET_COL]).copy()
    for c in final_cols:
        if train_clean[c].isna().any():
            med = train_clean[c].median()
            train_clean[c] = train_clean[c].fillna(med)
            valid_f[c] = valid_f[c].fillna(med)

    sw = np.ones(len(train_clean))
    is_q1 = train_clean["dt"].dt.month.isin([1, 2, 3]).values
    sw[is_q1] *= 1.5
    rep4 = (train_clean[n_repair_col] == 4).values
    sw[rep4] *= 1.2

    X = train_clean[final_cols].values
    y = train_clean[TARGET_COL].values
    n_avail = train_clean["n_avail"].values
    groups = train_clean["month_key"].values

    gkf = GroupKFold(n_splits=12)
    oof = np.zeros(len(train_clean))
    fold_nmae = []
    fold_q1 = []
    models = []
    print("[EXP-011 XGBoost MAE GPU]")
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        m = xgb.XGBRegressor(
            objective="reg:absoluteerror",
            tree_method="hist", device="cuda",
            n_estimators=3000, learning_rate=0.034,
            max_depth=9, reg_lambda=109.0, reg_alpha=2.0,
            min_child_weight=5,
            random_state=42, early_stopping_rounds=50,
            verbosity=0,
        )
        m.fit(X[tr], y[tr], sample_weight=sw[tr],
              eval_set=[(X[te], y[te])], verbose=False)
        pred = m.predict(X[te])
        pred_mw = np.clip(pred, 0.0, n_avail[te] * TURBINE_RATED_MW)
        oof[te] = pred_mw
        fold_nmae.append(nmae(y[te], pred_mw))
        is_q1_te = train_clean["dt"].iloc[te].dt.month.isin([1, 2, 3]).values
        if is_q1_te.sum() > 50:
            fold_q1.append(nmae(y[te][is_q1_te], pred_mw[is_q1_te]))
        models.append(m)
        print(f"  fold {fold:02d}: nMAE={fold_nmae[-1]:.3f}%")

    cv_mean = float(np.mean(fold_nmae))
    cv_q1 = float(np.mean(fold_q1)) if fold_q1 else cv_mean

    Xv = valid_f[final_cols].values
    preds = np.mean([m.predict(Xv) for m in models], axis=0)
    pred_mw_v = np.clip(preds, 0.0, valid_f["n_avail"].values * TURBINE_RATED_MW)

    pd.DataFrame({"dt": valid_f["dt"].values, "pred_mw": pred_mw_v,
                  "n_avail": valid_f["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")
    pd.DataFrame({"dt": train_clean["dt"].values, "y_true": y, "oof_pred": oof}).to_parquet(EXP_DIR / "oof.parquet")

    seg = compute_segments(y, oof, train_clean)
    print(f"\n[EXP-011] XGB GPU CV mean = {cv_mean:.4f}%   Q1 = {cv_q1:.4f}%")
    print(f"vs EXP-009 LGBM (9.234 / 8.771): {9.234-cv_mean:+.4f} / {8.771-cv_q1:+.4f}")

    summary = {
        "exp_id": "EXP-011", "name": "xgb_mae_gpu_on_exp009_features",
        "n_features": len(final_cols),
        "cv_mean_nmae": round(cv_mean, 4),
        "cv_q1_only_mean_nmae": round(cv_q1, 4),
        "fold_nmae": [round(x, 3) for x in fold_nmae],
        "segments": seg,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Time: {summary['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
