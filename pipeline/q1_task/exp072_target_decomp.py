"""
EXP-072: Target decomposition (contrarian top idea).

y_resid = y − climatology_lookup(month, hour, n_repair_base)
where n_repair_base = min(n_repair, 5).

Train LGBM на y_resid (EXP-021 setup), final pred = pred_resid + climatology.

Climatology fitted on Q1 2024+2025 (closest distribution to valid Q1 2026).
n_repair_base accounts for repair regime.
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae, smooth_power_curve_ti
from lgbm_q50 import prepare_features
from nwp_bias_correction import apply_bias, add_p_phys_corrected, estimate_bias
from sister_features import add_sister_features

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-072"
EXP_DIR.mkdir(parents=True, exist_ok=True)

SISTER_SOURCES = ["gfs", "icon", "arpege", "knmi", "dmi"]

REPAIR_COL = "Кол-во_ВЭУ_в_ремонте"


def build_climatology(train_clean, year_range=[2024, 2025]):
    """Returns dict {(month, hour, rep_base): median y}.
    Filter on Q1 train years closest to valid."""
    cal = train_clean[train_clean["dt"].dt.year.isin(year_range)].copy()
    cal["hour"] = cal["dt"].dt.hour
    cal["month"] = cal["dt"].dt.month
    cal["rep_base"] = np.minimum(cal[REPAIR_COL].values, 5)
    grp = cal.groupby(["month", "hour", "rep_base"])[TARGET_COL].median()
    return grp.to_dict()


def lookup_climatology(df, clim, default=None):
    """Apply climatology lookup to df. Returns array of climatology values."""
    months = df["dt"].dt.month.values
    hours = df["dt"].dt.hour.values
    reps = np.minimum(df[REPAIR_COL].values, 5).astype(int)
    out = np.zeros(len(df))
    if default is None:
        default = np.median(list(clim.values())) if clim else 30.0
    for i in range(len(df)):
        key = (int(months[i]), int(hours[i]), int(reps[i]))
        out[i] = clim.get(key, default)
    return out


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
            if c in df.columns: cols.append(c)
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
    print("[EXP-072] prepare features ...")
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

    train_f_filt = train_f[train_f[REPAIR_COL] <= 5].copy()
    train_clean = train_f_filt.dropna(subset=[TARGET_COL]).copy()

    # === Build climatology on Q1 2024+2025 (closest to valid 2026) ===
    clim = build_climatology(train_clean, year_range=[2024, 2025])
    print(f"Climatology entries: {len(clim)}")

    # Lookup on all train + valid
    train_clean["clim"] = lookup_climatology(train_clean, clim)
    valid_f["clim"] = lookup_climatology(valid_f, clim)
    print(f"Train clim stats: mean={train_clean['clim'].mean():.2f}, std={train_clean['clim'].std():.2f}")
    print(f"Valid clim stats: mean={valid_f['clim'].mean():.2f}")

    # Target = residual
    train_clean["y_resid"] = train_clean[TARGET_COL] - train_clean["clim"]
    print(f"y_resid stats: mean={train_clean['y_resid'].mean():.2f}, std={train_clean['y_resid'].std():.2f}")
    print(f"Original target stats: mean={train_clean[TARGET_COL].mean():.2f}, std={train_clean[TARGET_COL].std():.2f}")

    # === Train EXP-021 setup on y_resid ===
    extra_lag_names = [f"{c}_xlag_{('p' if h > 0 else 'm')}{abs(h)}"
                       for c in EXTRA_LAG_COLS for h in EXTRA_LAG_HOURS]
    sister_cols = [c for c in train_clean.columns if any(f"__{s}" in c for s in SISTER_SOURCES)]
    dis_cols = [c for c in train_clean.columns if any(suf in c for suf in
                ["__mean_all", "__std_all", "__range_all", "__bias_vs_mean"])]
    must_have_plus_clim = must_have + ["clim"]
    all_cols = list(dict.fromkeys(top15 + must_have_plus_clim + extra_lag_names + sister_cols + dis_cols))
    final_cols = [c for c in all_cols if c in train_clean.columns and c in valid_f.columns]
    print(f"Features: {len(final_cols)}")

    for c in final_cols:
        if train_clean[c].isna().any() or valid_f[c].isna().any():
            med = train_clean[c].median()
            if pd.isna(med): med = 0.0
            train_clean[c] = train_clean[c].fillna(med)
            valid_f[c] = valid_f[c].fillna(med)

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
        "objective": "regression_l1",  # MAE loss for residual (quantile τ=0.5 on residual = MAE)
        "metric": "mae",
        "verbose": -1, "num_threads": 0, "device": "cpu", "seed": 42, **bp,
    }
    hps.pop("alpha", None)  # remove quantile param

    sw = np.ones(len(train_clean))  # no sw[q1] fix
    rep4 = (train_clean[REPAIR_COL] == 4).values
    rep5 = (train_clean[REPAIR_COL] == 5).values
    sw[rep4] *= 1.2; sw[rep5] *= 1.2

    X = train_clean[final_cols].values
    y_resid = train_clean["y_resid"].values
    y_true = train_clean[TARGET_COL].values
    clim_arr = train_clean["clim"].values
    n_avail = train_clean["n_avail"].values
    groups = train_clean["month_key"].values
    cat_idx = [final_cols.index("sector_8")] if "sector_8" in final_cols else "auto"

    gkf = GroupKFold(n_splits=12)
    oof = np.zeros(len(train_clean))  # final prediction y = pred_resid + clim
    fold_nmae, fold_q1, feat_imp, models = [], [], np.zeros(len(final_cols)), []
    for fold, (tr, te) in enumerate(gkf.split(X, y_resid, groups)):
        ds_tr = lgb.Dataset(X[tr], label=y_resid[tr], weight=sw[tr], categorical_feature=cat_idx)
        ds_te = lgb.Dataset(X[te], label=y_resid[te], weight=sw[te], reference=ds_tr, categorical_feature=cat_idx)
        m = lgb.train(hps, ds_tr, num_boost_round=n_boost, valid_sets=[ds_te],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred_resid = m.predict(X[te], num_iteration=m.best_iteration)
        pred_full = pred_resid + clim_arr[te]
        pred_mw = np.clip(pred_full, 0.0, n_avail[te] * TURBINE_RATED_MW)
        oof[te] = pred_mw
        fold_nmae.append(nmae(y_true[te], pred_mw))
        is_q1_te = train_clean["dt"].iloc[te].dt.month.isin([1, 2, 3]).values
        if is_q1_te.sum() > 50:
            fold_q1.append(nmae(y_true[te][is_q1_te], pred_mw[is_q1_te]))
        feat_imp += m.feature_importance("gain")
        models.append(m)
        print(f"  fold {fold:02d}: nMAE={fold_nmae[-1]:.3f}%")

    cv_mean = float(np.mean(fold_nmae)); cv_std = float(np.std(fold_nmae))
    cv_q1 = float(np.mean(fold_q1)) if fold_q1 else cv_mean
    print(f"\n[EXP-072] CV = {cv_mean:.4f}% +/- {cv_std:.4f}% Q1={cv_q1:.4f}%")
    print(f"vs EXP-021 (8.238 / 8.242): {8.238-cv_mean:+.4f} / {8.242-cv_q1:+.4f}")
    print(f"vs EXP-069 (8.209 / 8.336): {8.209-cv_mean:+.4f} / {8.336-cv_q1:+.4f}")

    Xv = valid_f[final_cols].values
    clim_v = valid_f["clim"].values
    preds_resid = np.mean([m.predict(Xv, num_iteration=m.best_iteration) for m in models], axis=0)
    preds_full = preds_resid + clim_v
    pred_mw_v = np.clip(preds_full, 0.0, valid_f["n_avail"].values * TURBINE_RATED_MW)
    pd.DataFrame({"dt": valid_f["dt"].values, "pred_mw": pred_mw_v,
                  "p_phys": valid_f["p_phys"].values, "n_avail": valid_f["n_avail"].values,
                  "clim": clim_v}).to_parquet(EXP_DIR / "valid_pred.parquet")
    pd.DataFrame({"dt": train_clean["dt"].values, "y_true": y_true, "oof_pred": oof,
                  "clim": clim_arr}).to_parquet(EXP_DIR / "oof.parquet")

    summary = {
        "exp_id": "EXP-072", "name": "target_decomp_climatology_residual",
        "n_features": len(final_cols),
        "climatology_period": "Q1 2024+2025",
        "n_clim_entries": len(clim),
        "cv_mean_nmae": cv_mean, "cv_std_nmae": cv_std,
        "cv_q1_only_mean_nmae": cv_q1,
        "elapsed_sec": time.time() - t0,
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Done: {EXP_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
