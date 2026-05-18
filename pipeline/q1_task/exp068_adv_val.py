"""
EXP-068: Adversarial validation -> importance weighting.

Идея: train binary LGB на (train vs valid), pred prob = подобие к valid distribution.
Sample weight для финальной модели = p_test / (1 - p_test).
Это решает проблему "recency не работает": взвешиваем по distribution similarity, не по дате.

Pipeline:
1. Load train (32434), valid (2126).
2. Take same physics features (~50), exclude target, n_repair (which is monthly aggregate).
3. Binary LGB: y_adv = 1 if valid else 0.
4. CV 5-fold to estimate honest p_test for each train row.
5. w = p_test / (1 - p_test), clip [0.1, 10] to avoid extremes.
6. Refit EXP-021 setup with these sample weights (multiply with rep4/rep5 weights, NO sw[q1]).
7. Compare CV vs EXP-021 baseline.

Save: OOF, valid pred, summary.json. Submit-ready output.
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, GroupKFold

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae, smooth_power_curve_ti
from lgbm_q50 import prepare_features, compute_segments
from nwp_bias_correction import apply_bias, add_p_phys_corrected, estimate_bias
from sister_features import add_sister_features

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-068"
EXP_DIR.mkdir(parents=True, exist_ok=True)

SISTER_SOURCES = ["gfs", "icon", "arpege", "knmi", "dmi"]

# Same setup as exp021
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
            if c in df.columns:
                cols.append(c)
        if len(cols) < 2: continue
        df[f"{var}__mean_all"] = df[cols].mean(axis=1)
        df[f"{var}__std_all"] = df[cols].std(axis=1)
        df[f"{var}__range_all"] = df[cols].max(axis=1) - df[cols].min(axis=1)
        for c in cols:
            tag = c.replace(var, "").lstrip("_") or "ecmwf"
            df[f"{var}__bias_vs_mean__{tag}"] = df[c] - df[f"{var}__mean_all"]
    return df


def main():
    t0 = time.time()
    print("[EXP-068] prep features same as EXP-021 ...")
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
    train_f_filt = train_f[train_f[n_repair_col] <= 5].copy()

    extra_lag_names = [f"{c}_xlag_{('p' if h > 0 else 'm')}{abs(h)}"
                       for c in EXTRA_LAG_COLS for h in EXTRA_LAG_HOURS]
    sister_cols = [c for c in train_f.columns if any(f"__{s}" in c for s in SISTER_SOURCES)]
    dis_cols = [c for c in train_f.columns if any(suf in c for suf in
                ["__mean_all", "__std_all", "__range_all", "__bias_vs_mean"])]
    all_cols = list(dict.fromkeys(TOP_FEATS + extra_lag_names + sister_cols + dis_cols))
    final_cols = [c for c in all_cols if c in train_f_filt.columns]
    print(f"features: {len(final_cols)}")

    train_clean = train_f_filt.dropna(subset=[TARGET_COL]).copy()
    for c in final_cols:
        if train_clean[c].isna().any() or valid_f[c].isna().any():
            med = train_clean[c].median()
            if pd.isna(med): med = 0.0
            train_clean[c] = train_clean[c].fillna(med)
            valid_f[c] = valid_f[c].fillna(med)

    # === Adversarial validation ===
    print(f"[EXP-068] adversarial validation: train n={len(train_clean)}, valid n={len(valid_f)}")
    X_adv = pd.concat([train_clean[final_cols], valid_f[final_cols]], ignore_index=True)
    y_adv = np.concatenate([np.zeros(len(train_clean)), np.ones(len(valid_f))])
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    p_adv = np.zeros(len(X_adv))
    adv_params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
        "verbose": -1, "num_threads": 8, "seed": 42,
    }
    aucs = []
    for fold, (tr, te) in enumerate(cv.split(X_adv)):
        ds_tr = lgb.Dataset(X_adv.iloc[tr], label=y_adv[tr])
        ds_te = lgb.Dataset(X_adv.iloc[te], label=y_adv[te], reference=ds_tr)
        m = lgb.train(adv_params, ds_tr, num_boost_round=500, valid_sets=[ds_te],
                       callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        p_adv[te] = m.predict(X_adv.iloc[te], num_iteration=m.best_iteration)
        # AUC quick check
        from sklearn.metrics import roc_auc_score
        aucs.append(roc_auc_score(y_adv[te], p_adv[te]))
        print(f"  adv fold {fold}: AUC={aucs[-1]:.4f}")
    mean_auc = np.mean(aucs)
    print(f"  adv AUC mean: {mean_auc:.4f} (>=0.6 = significant shift)")

    p_test_train = p_adv[:len(train_clean)]
    # Importance weight: w = p_test / (1 - p_test), clip to avoid blowups
    w_iw = p_test_train / np.clip(1 - p_test_train, 1e-3, 1.0)
    w_iw = np.clip(w_iw, 0.1, 10.0)
    print(f"  w_iw stats: mean={w_iw.mean():.3f}, median={np.median(w_iw):.3f}, max={w_iw.max():.3f}, min={w_iw.min():.3f}")
    print(f"  Q1 rows in train weighted: mean={w_iw[train_clean['dt'].dt.month.isin([1,2,3]).values].mean():.3f} vs other mean={w_iw[~train_clean['dt'].dt.month.isin([1,2,3]).values].mean():.3f}")

    # === Now fit main model with importance weights ===
    import optuna
    study = optuna.load_study(
        study_name="exp004_lgbm_q1",
        storage="sqlite:////home/duck/wind_hackathon/experiments/archive/rejected/duplicate/EXP-004/optuna.db",
    )
    bp = study.best_params.copy()
    bp.pop("sw_q1_factor", None)
    n_boost = bp.pop("n_boost", 2000)
    bp["feature_fraction"] = 0.7
    bp.setdefault("min_data_in_leaf", 200)
    hps = {
        "objective": "quantile", "alpha": 0.5, "metric": "quantile",
        "verbose": -1, "num_threads": 0, "device": "cpu", "seed": 42, **bp,
    }
    # rep4/rep5 weights (как в EXP-021)
    rep4 = (train_clean[n_repair_col] == 4).values
    rep5 = (train_clean[n_repair_col] == 5).values
    sw = w_iw.copy()
    sw[rep4] *= 1.2; sw[rep5] *= 1.2

    X = train_clean[final_cols].values
    y = train_clean[TARGET_COL].values
    n_avail = train_clean["n_avail"].values
    groups = train_clean["month_key"].values
    cat_idx = [final_cols.index("sector_8")] if "sector_8" in final_cols else "auto"

    gkf = GroupKFold(n_splits=12)
    oof = np.zeros(len(train_clean))
    fold_nmae, fold_q1, feat_imp, models = [], [], np.zeros(len(final_cols)), []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        ds_tr = lgb.Dataset(X[tr], label=y[tr], weight=sw[tr], categorical_feature=cat_idx)
        ds_te = lgb.Dataset(X[te], label=y[te], weight=sw[te], reference=ds_tr, categorical_feature=cat_idx)
        m = lgb.train(hps, ds_tr, num_boost_round=n_boost, valid_sets=[ds_te],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred = m.predict(X[te], num_iteration=m.best_iteration)
        pred_mw = np.clip(pred, 0.0, n_avail[te] * TURBINE_RATED_MW)
        oof[te] = pred_mw
        fold_nmae.append(nmae(y[te], pred_mw))
        is_q1_te = train_clean["dt"].iloc[te].dt.month.isin([1, 2, 3]).values
        if is_q1_te.sum() > 50:
            fold_q1.append(nmae(y[te][is_q1_te], pred_mw[is_q1_te]))
        feat_imp += m.feature_importance("gain")
        models.append(m)
        print(f"  fold {fold:02d}: nMAE={fold_nmae[-1]:.3f}%")

    cv_mean = float(np.mean(fold_nmae)); cv_std = float(np.std(fold_nmae))
    cv_q1 = float(np.mean(fold_q1)) if fold_q1 else cv_mean
    print(f"\n[EXP-068] CV = {cv_mean:.4f}% +/- {cv_std:.4f}% Q1={cv_q1:.4f}%")
    print(f"vs EXP-021 (8.238 / 8.242): {8.238-cv_mean:+.4f} / {8.242-cv_q1:+.4f}")

    Xv = valid_f[final_cols].values
    preds = np.mean([m.predict(Xv, num_iteration=m.best_iteration) for m in models], axis=0)
    pred_mw_v = np.clip(preds, 0.0, valid_f["n_avail"].values * TURBINE_RATED_MW)
    pd.DataFrame({"dt": valid_f["dt"].values, "pred_mw": pred_mw_v,
                  "p_phys": valid_f["p_phys"].values, "n_avail": valid_f["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")
    pd.DataFrame({"dt": train_clean["dt"].values, "y_true": y, "oof_pred": oof,
                  "w_iw": w_iw}).to_parquet(EXP_DIR / "oof.parquet")

    summary = {
        "exp_id": "EXP-068", "name": "adversarial_validation_weights",
        "n_features": len(final_cols),
        "adv_auc_mean": float(mean_auc),
        "w_iw_stats": {
            "mean": float(w_iw.mean()), "median": float(np.median(w_iw)),
            "min": float(w_iw.min()), "max": float(w_iw.max()),
        },
        "cv_mean_nmae": cv_mean, "cv_std_nmae": cv_std,
        "cv_q1_only_mean_nmae": cv_q1,
        "elapsed_sec": time.time() - t0,
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\nDone: {EXP_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
