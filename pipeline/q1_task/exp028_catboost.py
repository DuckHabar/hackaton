"""
EXP-028: CatBoost Quantile τ=0.5 на EXP-021 features (5 src + disagreement).

Diversity от LGBM. CatBoost ordered boosting + GPU acceleration.
Гипотеза: разные модели = разные ошибки = blend gain потом.

Использует те же features что и EXP-021, GroupKFold/12 по месяцу.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import catboost as cb
import optuna
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae, smooth_power_curve_ti
from lgbm_q50 import prepare_features, compute_segments
from nwp_bias_correction import apply_bias, add_p_phys_corrected, estimate_bias
from sister_features import add_sister_features

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-028"
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

EXTRA_LAG_HOURS = [-48, -36, -24, -18, -12, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48]
EXTRA_LAG_COLS = ["wind_speed_120m", "ws_corrected_120", "wind_gusts_10m",
                  "wd_120_sin", "wd_120_cos", "alpha_80_120", "p_phys_corrected"]

SISTER_SOURCES = ["gfs", "icon", "arpege", "knmi", "dmi"]


def add_extra_lags_base(df_all):
    df_all = df_all.sort_values("dt").reset_index(drop=True)
    for col in EXTRA_LAG_COLS:
        if col not in df_all.columns: continue
        for h in EXTRA_LAG_HOURS:
            sign = "p" if h > 0 else "m"
            name = f"{col}_xlag_{sign}{abs(h)}"
            if name not in df_all.columns:
                df_all[name] = df_all[col].shift(-h)
    return df_all


def add_disagreement_features(df, sources):
    key_vars = ["wind_speed_120m", "wind_speed_80m", "wind_gusts_10m"]
    for var in key_vars:
        cols = [var]
        for s in sources:
            c = f"{var}__{s}"
            if c in df.columns: cols.append(c)
        if len(cols) < 2: continue
        df[f"{var}__mean_all"] = df[cols].mean(axis=1)
        df[f"{var}__std_all"] = df[cols].std(axis=1)
        df[f"{var}__range_all"] = df[cols].max(axis=1) - df[cols].min(axis=1)
    return df


def main():
    t0 = time.time()
    print("[EXP-028] preparing features ...")
    train_f, valid_f = prepare_features()
    pc = smooth_power_curve_ti(ti=0.20, rated_speed=10.0)
    bt, _ = estimate_bias(train_f, pc)
    train_f = apply_bias(train_f, bt, mode="mean"); train_f = add_p_phys_corrected(train_f, pc)
    valid_f = apply_bias(valid_f, bt, mode="mean"); valid_f = add_p_phys_corrected(valid_f, pc)

    combined = pd.concat([train_f.assign(_src="train"), valid_f.assign(_src="valid")], ignore_index=True)
    combined = add_extra_lags_base(combined)
    train_f = combined[combined["_src"] == "train"].drop(columns=["_src"]).copy()
    valid_f = combined[combined["_src"] == "valid"].drop(columns=["_src"]).copy()

    train_f, valid_f = add_sister_features(train_f, valid_f, sources=SISTER_SOURCES)
    train_f = add_disagreement_features(train_f, SISTER_SOURCES)
    valid_f = add_disagreement_features(valid_f, SISTER_SOURCES)

    n_repair_col = "Кол-во_ВЭУ_в_ремонте"
    train_f_filtered = train_f[train_f[n_repair_col] <= 5].copy()

    extra_lag_names_base = [f"{c}_xlag_{('p' if h > 0 else 'm')}{abs(h)}"
                            for c in EXTRA_LAG_COLS for h in EXTRA_LAG_HOURS]
    sister_cols = [c for c in train_f.columns if any(f"__{s}" in c for s in SISTER_SOURCES)]
    dis_cols = [c for c in train_f.columns if "__mean_all" in c or "__std_all" in c or "__range_all" in c]
    all_cols = list(dict.fromkeys(TOP_FEATS + extra_lag_names_base + sister_cols + dis_cols))
    final_cols = [c for c in all_cols if c in train_f_filtered.columns]
    print(f"[EXP-028] {len(final_cols)} features ...")

    train_clean = train_f_filtered.dropna(subset=[TARGET_COL]).copy()
    for c in final_cols:
        if train_clean[c].isna().any() or valid_f[c].isna().any():
            med = train_clean[c].median()
            if pd.isna(med): med = 0.0
            train_clean[c] = train_clean[c].fillna(med).astype(float)
            valid_f[c] = valid_f[c].fillna(med).astype(float)
        else:
            train_clean[c] = train_clean[c].astype(float)
            valid_f[c] = valid_f[c].astype(float)

    sw = np.ones(len(train_clean))
    is_q1 = train_clean["dt"].dt.month.isin([1, 2, 3]).values
    sw[is_q1] *= 1.5
    rep4 = (train_clean[n_repair_col] == 4).values
    rep5 = (train_clean[n_repair_col] == 5).values
    sw[rep4] *= 1.2; sw[rep5] *= 1.2

    X = train_clean[final_cols].values.astype(np.float32)
    y = train_clean[TARGET_COL].values.astype(np.float32)
    n_avail = train_clean["n_avail"].values
    groups = train_clean["month_key"].values
    Xv = valid_f[final_cols].values.astype(np.float32)
    nv_avail = valid_f["n_avail"].values

    print(f"X shape: {X.shape}, ready for CatBoost GPU ...")

    gkf = GroupKFold(n_splits=12)
    oof = np.zeros(len(train_clean))
    valid_preds = []
    fold_nmae = []; fold_q1 = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        ts = time.time()
        m = cb.CatBoostRegressor(
            loss_function="Quantile:alpha=0.5",
            iterations=3000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=5,
            random_seed=42,
            task_type="GPU",
            devices="0",
            early_stopping_rounds=80,
            verbose=0,
            allow_writing_files=False,
        )
        m.fit(X[tr], y[tr], sample_weight=sw[tr],
              eval_set=(X[te], y[te]),
              use_best_model=True)
        pred = m.predict(X[te])
        pred_mw = np.clip(pred, 0.0, n_avail[te] * TURBINE_RATED_MW)
        oof[te] = pred_mw
        fold_nmae.append(nmae(y[te], pred_mw))
        is_q1_te = train_clean["dt"].iloc[te].dt.month.isin([1, 2, 3]).values
        if is_q1_te.sum() > 50:
            fold_q1.append(nmae(y[te][is_q1_te], pred_mw[is_q1_te]))
        valid_preds.append(m.predict(Xv))
        print(f"  fold {fold:02d}: nMAE={fold_nmae[-1]:.3f}% iters={m.tree_count_} ({time.time()-ts:.1f}s)")

    cv_mean = float(np.mean(fold_nmae)); cv_std = float(np.std(fold_nmae))
    cv_q1 = float(np.mean(fold_q1)) if fold_q1 else cv_mean
    print(f"\n[EXP-028] CatBoost CV = {cv_mean:.4f}% +/- {cv_std:.4f}% Q1={cv_q1:.4f}%")
    print(f"vs EXP-021 LGBM (8.242 / 8.351): {8.242-cv_mean:+.4f} / {8.351-cv_q1:+.4f}")

    valid_pred_mean = np.mean(valid_preds, axis=0)
    pred_mw_v = np.clip(valid_pred_mean, 0.0, nv_avail * TURBINE_RATED_MW)
    pd.DataFrame({"dt": valid_f["dt"].values, "pred_mw": pred_mw_v,
                  "p_phys": valid_f["p_phys"].values, "n_avail": nv_avail}).to_parquet(EXP_DIR / "valid_pred.parquet")
    pd.DataFrame({"dt": train_clean["dt"].values, "y_true": y, "oof_pred": oof}).to_parquet(EXP_DIR / "oof.parquet")

    summary = {
        "exp_id": "EXP-028", "name": "catboost_quantile_5src_gpu",
        "n_features": len(final_cols), "n_train_clean": len(train_clean),
        "model": "CatBoost Quantile:alpha=0.5 GPU",
        "sister_sources": SISTER_SOURCES,
        "cv_mean_nmae": round(cv_mean, 4), "cv_std_nmae": round(cv_std, 4),
        "cv_q1_only_mean_nmae": round(cv_q1, 4),
        "fold_nmae": [round(x, 3) for x in fold_nmae],
        "fold_nmae_q1_only": [round(x, 3) for x in fold_q1],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Summary: {EXP_DIR / 'summary.json'}, time {summary['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
