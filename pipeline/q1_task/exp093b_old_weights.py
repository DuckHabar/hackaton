"""EXP-093b: same features as EXP-093 but EXP-021 sample weights (revert recency/season change)."""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae, smooth_power_curve_ti
from lgbm_q50 import prepare_features
from nwp_bias_correction import apply_bias, add_p_phys_corrected, estimate_bias
from sister_features import add_sister_features

# reuse all the helpers from EXP-093
from exp093_phys_rolling import (
    add_extra_lags_base, add_disagreement_features,
    add_rolling_features, add_phys_features,
    fit_iso_feature, predict_iso_feature,
    SISTER_SOURCES, EXTRA_LAG_HOURS, EXTRA_LAG_COLS,
)

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-093b"
EXP_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    print("[EXP-093b] prepare_features ...")
    train_f, valid_f = prepare_features()
    pc = smooth_power_curve_ti(ti=0.20, rated_speed=10.0)
    bt, _ = estimate_bias(train_f, pc)
    train_f = apply_bias(train_f, bt, mode="mean")
    train_f = add_p_phys_corrected(train_f, pc)
    valid_f = apply_bias(valid_f, bt, mode="mean")
    valid_f = add_p_phys_corrected(valid_f, pc)
    combined = pd.concat([train_f.assign(_src="train"), valid_f.assign(_src="valid")], ignore_index=True)
    combined = add_extra_lags_base(combined)
    combined = add_rolling_features(combined)
    train_f = combined[combined["_src"] == "train"].drop(columns=["_src"]).copy()
    valid_f = combined[combined["_src"] == "valid"].drop(columns=["_src"]).copy()
    train_f, valid_f = add_sister_features(train_f, valid_f, sources=SISTER_SOURCES)
    train_f = add_disagreement_features(train_f, SISTER_SOURCES)
    valid_f = add_disagreement_features(valid_f, SISTER_SOURCES)
    train_f = add_phys_features(train_f)
    valid_f = add_phys_features(valid_f)

    n_repair_col = "Кол-во_ВЭУ_в_ремонте"
    train_clean = train_f[train_f[n_repair_col] <= 5].dropna(subset=[TARGET_COL]).copy()

    exp005_summary = json.load(open(ROOT / "experiments/archive/rejected/worse_cv/EXP-005/summary.json"))
    top15 = [f["feat"] for f in exp005_summary["top15_importance"][:15]]
    must_have = ["wind_speed_120m", "wind_speed_80m", "wind_speed_180m", "wind_speed_10m",
        "ws_corrected_120", "ws_corrected_120_corr", "p_phys", "p_phys_corrected",
        "p_phys_no_density", "rho_air", "alpha_80_120",
        "wind_gusts_10m", "rews_simple", "rews_corr", "ws_at_84", "wd_120_sin", "wd_120_cos", "sector_8",
        "n_avail", "month", "hour_of_day", "temperature_80m", "pressure_msl", "available_frac", "icing_risk"]
    base_feats = list(dict.fromkeys(top15 + must_have))
    extra_lag_names = [f"{c}_xlag_{('p' if h > 0 else 'm')}{abs(h)}"
                       for c in EXTRA_LAG_COLS for h in EXTRA_LAG_HOURS]
    rolling_cols = []
    for col in ["ws_corrected_120", "wind_speed_120m", "wind_gusts_10m"]:
        for w in [3, 6, 12, 24]:
            rolling_cols.append(f"{col}_roll_mean_{w}h"); rolling_cols.append(f"{col}_roll_std_{w}h")
        rolling_cols.extend([f"{col}_roll_min_6h", f"{col}_roll_max_6h", f"{col}_roll_range_6h",
            f"{col}_diff_3h", f"{col}_diff_6h", f"{col}_diff_abs_3h",
            f"{col}_ewm_3h", f"{col}_ewm_12h", f"{col}_ewm_48h",
            f"{col}_dev_ewm_3h", f"{col}_dev_ewm_12h", f"{col}_dev_ewm_48h",
            f"{col}_ti_proxy_6h", f"{col}_ratio_24h", f"{col}_local_range_3h"])
    phys_cols = ["solar_elev", "solar_elev_sin", "is_daytime", "insolation_proxy",
        "richardson_number", "atm_stable", "atm_unstable", "atm_neutral", "night_stable",
        "llj_ratio", "llj_flag", "llj_excess", "hodograph_span",
        "sector_NE", "sector_SE", "sector_S", "sector_SW", "sector_S_all", "night_x_south", "gust_excess"]
    iso_feat_col = ["iso_baseline_pred"]
    sister_cols = [c for c in train_clean.columns if any(f"__{s}" in c for s in SISTER_SOURCES)]
    dis_cols = [c for c in train_clean.columns if "__mean_all" in c or "__std_all" in c
                or "__range_all" in c or "__bias_vs_mean" in c]
    all_cols = list(dict.fromkeys(base_feats + extra_lag_names + rolling_cols + phys_cols + iso_feat_col + sister_cols + dis_cols))
    final_cols = [c for c in all_cols if c in train_clean.columns or c == "iso_baseline_pred"]

    for c in final_cols:
        if c == "iso_baseline_pred": continue
        if c not in train_clean.columns: train_clean[c] = 0.0
        if c not in valid_f.columns: valid_f[c] = 0.0
        if train_clean[c].isna().any() or valid_f[c].isna().any():
            med = train_clean[c].median()
            if pd.isna(med): med = 0.0
            train_clean[c] = train_clean[c].fillna(med)
            valid_f[c] = valid_f[c].fillna(med)

    # EXP-021 sample weights
    sw = np.ones(len(train_clean))
    is_q1 = train_clean["dt"].dt.month.isin([1, 2, 3]).values
    sw[is_q1] *= 1.5
    n_rep = train_clean[n_repair_col].values
    sw[(n_rep == 4)] *= 1.2
    sw[(n_rep == 5)] *= 1.2

    study = optuna.load_study(
        study_name="exp004_lgbm_q1",
        storage="sqlite:////home/duck/wind_hackathon/experiments/archive/rejected/duplicate/EXP-004/optuna.db")
    bp = study.best_params.copy()
    bp.pop("sw_q1_factor", None); n_boost = bp.pop("n_boost", 2000)
    bp["feature_fraction"] = 0.7; bp.setdefault("min_data_in_leaf", 200)
    hps = {"objective": "quantile", "alpha": 0.5, "metric": "quantile",
        "verbose": -1, "num_threads": 0, "device": "cpu", "seed": 42, **bp}

    y = train_clean[TARGET_COL].values
    n_avail = train_clean["n_avail"].values
    groups = train_clean["month_key"].values
    ws_eff = train_clean["ws_corrected_120_corr"].values
    ws_eff_v = valid_f["ws_corrected_120_corr"].values

    gkf = GroupKFold(n_splits=12)
    iso_pred_train = np.zeros(len(train_clean))
    for fold, (tr, te) in enumerate(gkf.split(train_clean, y, groups)):
        iso = fit_iso_feature(ws_eff[tr], y[tr], n_avail[tr], weights=sw[tr])
        iso_pred_train[te] = predict_iso_feature(iso, ws_eff[te], n_avail[te])
    iso_full = fit_iso_feature(ws_eff, y, n_avail, weights=sw)
    iso_pred_valid = predict_iso_feature(iso_full, ws_eff_v, valid_f["n_avail"].values)
    train_clean["iso_baseline_pred"] = iso_pred_train
    valid_f["iso_baseline_pred"] = iso_pred_valid

    cat_idx = [final_cols.index("sector_8")] if "sector_8" in final_cols else "auto"
    X = train_clean[final_cols].values
    Xv = valid_f[final_cols].values

    fold_nmae = []; models = []; oof = np.zeros(len(train_clean))
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        ds_tr = lgb.Dataset(X[tr], label=y[tr], weight=sw[tr], categorical_feature=cat_idx)
        ds_te = lgb.Dataset(X[te], label=y[te], weight=sw[te], reference=ds_tr, categorical_feature=cat_idx)
        m = lgb.train(hps, ds_tr, num_boost_round=n_boost, valid_sets=[ds_te],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred = m.predict(X[te], num_iteration=m.best_iteration)
        cap = n_avail[te] * TURBINE_RATED_MW
        pred_mw = np.clip(pred, 0.0, cap)
        oof[te] = pred_mw
        fold_nmae.append(nmae(y[te], pred_mw))
        models.append(m)
        print(f"  fold {fold:02d}: nMAE={fold_nmae[-1]:.3f}%")

    cv_mean = float(np.mean(fold_nmae))
    is_q1_2025 = ((train_clean["dt"].dt.year == 2025) & train_clean["dt"].dt.month.isin([1, 2, 3])).values
    q1_2025 = float(nmae(y[is_q1_2025], oof[is_q1_2025]))
    print(f"\n[EXP-093b] CV = {cv_mean:.4f}%   Q1 2025 OOF (n={is_q1_2025.sum()}) = {q1_2025:.4f}%")
    print(f"vs EXP-021 7.326: {7.326 - q1_2025:+.4f}")
    print(f"vs EXP-093 7.380: {7.380 - q1_2025:+.4f}")
    print(f"vs EXP-054 cal 7.278: {7.2782 - q1_2025:+.4f}")

    preds = np.mean([m.predict(Xv, num_iteration=m.best_iteration) for m in models], axis=0)
    pred_mw_v = np.clip(preds, 0.0, valid_f["n_avail"].values * TURBINE_RATED_MW)
    pd.DataFrame({"dt": valid_f["dt"].values, "pred_mw": pred_mw_v,
                  "p_phys": valid_f["p_phys"].values, "n_avail": valid_f["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")
    pd.DataFrame({"dt": train_clean["dt"].values, "y_true": y, "oof_pred": oof}).to_parquet(EXP_DIR / "oof.parquet")
    summary = {"exp_id": "EXP-093b", "n_features": len(final_cols),
        "cv_mean_nmae": round(cv_mean, 4), "q1_2025_oof_nmae": round(q1_2025, 4),
        "fold_nmae": [round(x, 3) for x in fold_nmae],
        "elapsed_sec": round(time.time() - t0, 1)}
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
