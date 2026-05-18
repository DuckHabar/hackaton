"""
EXP-076: Wake/array losses Jensen model (fresh_eyes #4 missing physics fix).

Идея: 26 turbines в плотном расположении, при longitudinal wind wake loss до 10-15%.
Jensen single-wake deficit: η = (1 − 0.5·(1−√(1−Ct))²)^N_in_wake
- Ct ≈ 0.85 для cube-region (8-12 m/s)
- N_in_wake ≈ зависит от sector_8 (longitudinal vs perpendicular)

Crude model:
- Per sector wake efficiency: [1.0, 0.95, 0.90, 0.85, 0.90, 0.95, 1.0, 0.95]
  (8 секторов: N, NE, E, SE, S, SW, W, NW)
- ws_eff = ws_120 * efficiency_sector
- p_phys_wake = power_curve(ws_eff) * n_avail

Add p_phys_wake + wake_efficiency_sector as features.
EXP-069 setup (spatial) + wake features.
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

# Import EXP-069 spatial helpers
sys.path.insert(0, str(Path("/home/duck/wind_hackathon/pipeline/q1_task")))
from exp069_spatial import (
    add_spatial_features, load_spatial,
    SPATIAL_POINTS, add_extra_lags_base, add_disagreement_features,
    EXTRA_LAG_HOURS, EXTRA_LAG_COLS,
)

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-076"
EXP_DIR.mkdir(parents=True, exist_ok=True)

SISTER_SOURCES = ["gfs", "icon", "arpege", "knmi", "dmi"]

# Wake efficiency per sector (8 sectors N..NW, crude estimate based on typical farm layout)
# Assumption: prevailing wind NE-E gives most data, layout likely N-S или NE-SW.
# Sectors NE, SW (longitudinal) -> highest wake -> 0.85
# Sectors E, W (cross-array) -> 0.90
# Sectors N, S (orthogonal) -> 0.95
# Diagonals NW, SE -> 0.92
WAKE_EFF_8 = {
    0: 0.95,  # N
    1: 0.85,  # NE - longitudinal, max wake
    2: 0.90,  # E
    3: 0.92,  # SE
    4: 0.95,  # S
    5: 0.85,  # SW - longitudinal
    6: 0.90,  # W
    7: 0.92,  # NW
}


def add_wake_features(df, pc):
    """Add wake_eff (per sector_8) + ws_eff_wake + p_phys_wake."""
    d = df.copy()
    if "sector_8" not in d.columns:
        return d
    sec = d["sector_8"].fillna(0).astype(int).values
    eff = np.array([WAKE_EFF_8.get(int(s), 0.95) for s in sec])
    d["wake_eff_sector"] = eff
    # ws_eff considering wake
    ws_col = "ws_corrected_120" if "ws_corrected_120" in d.columns else "wind_speed_120m"
    if ws_col in d.columns:
        ws_eff = d[ws_col].values * eff
        d["ws_eff_wake"] = ws_eff
        # p_phys via power curve on ws_eff
        p_per = np.interp(ws_eff, pc["wind_speed"].values, pc["value"].values) / 1e6
        if "n_avail" in d.columns:
            d["p_phys_wake"] = p_per * d["n_avail"].values
    return d


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
    "wake_eff_sector", "ws_eff_wake", "p_phys_wake",  # NEW
]
TOP_FEATS = list(dict.fromkeys(top15 + must_have))


def main():
    t0 = time.time()
    print("[EXP-076] prepare features ...")
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

    # Spatial features
    spatial = load_spatial(SPATIAL_POINTS)
    train_f = add_spatial_features(train_f, spatial)
    valid_f = add_spatial_features(valid_f, spatial)

    # Wake features
    train_f = add_wake_features(train_f, pc)
    valid_f = add_wake_features(valid_f, pc)
    print(f"[EXP-076] wake_eff_sector unique: {sorted(train_f['wake_eff_sector'].unique())}")
    print(f"  p_phys_wake stats: mean={train_f['p_phys_wake'].mean():.2f}, p_phys mean={train_f['p_phys'].mean():.2f}")

    n_repair_col = "Кол-во_ВЭУ_в_ремонте"
    train_f_filt = train_f[train_f[n_repair_col] <= 5].copy()

    extra_lag_names = [f"{c}_xlag_{('p' if h > 0 else 'm')}{abs(h)}"
                       for c in EXTRA_LAG_COLS for h in EXTRA_LAG_HOURS]
    sister_cols = [c for c in train_f.columns if any(f"__{s}" in c for s in SISTER_SOURCES)]
    dis_cols = [c for c in train_f.columns if any(suf in c for suf in
                ["__mean_all", "__std_all", "__range_all", "__bias_vs_mean"])]
    spatial_cols = [c for c in train_f.columns if c.startswith("spatial_")]
    all_cols = list(dict.fromkeys(TOP_FEATS + extra_lag_names + sister_cols + dis_cols + spatial_cols))
    final_cols = [c for c in all_cols if c in train_f_filt.columns]
    print(f"[EXP-076] features: {len(final_cols)}")

    train_clean = train_f_filt.dropna(subset=[TARGET_COL]).copy()
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
        "objective": "quantile", "alpha": 0.5, "metric": "quantile",
        "verbose": -1, "num_threads": 0, "device": "cpu", "seed": 42, **bp,
    }

    sw = np.ones(len(train_clean))
    rep4 = (train_clean[n_repair_col] == 4).values
    rep5 = (train_clean[n_repair_col] == 5).values
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
    print(f"\n[EXP-076] CV = {cv_mean:.4f}% +/- {cv_std:.4f}% Q1={cv_q1:.4f}%")
    print(f"vs EXP-069 (8.209 / 8.336): {8.209-cv_mean:+.4f} / {8.336-cv_q1:+.4f}")

    Xv = valid_f[final_cols].values
    preds = np.mean([m.predict(Xv, num_iteration=m.best_iteration) for m in models], axis=0)
    pred_mw_v = np.clip(preds, 0.0, valid_f["n_avail"].values * TURBINE_RATED_MW)
    pd.DataFrame({"dt": valid_f["dt"].values, "pred_mw": pred_mw_v,
                  "p_phys": valid_f["p_phys"].values, "n_avail": valid_f["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")
    pd.DataFrame({"dt": train_clean["dt"].values, "y_true": y, "oof_pred": oof}).to_parquet(EXP_DIR / "oof.parquet")

    imp = sorted(zip(final_cols, feat_imp), key=lambda kv: -kv[1])
    print("\n=== Wake feature importance ===")
    for f, g in imp:
        if f in ["wake_eff_sector", "ws_eff_wake", "p_phys_wake"]:
            print(f"  {f}: {g:.0f}")

    summary = {
        "exp_id": "EXP-076", "name": "wake_jensen_features",
        "n_features": len(final_cols),
        "wake_efficiencies": WAKE_EFF_8,
        "cv_mean_nmae": cv_mean, "cv_std_nmae": cv_std,
        "cv_q1_only_mean_nmae": cv_q1,
        "elapsed_sec": time.time() - t0,
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Done: {EXP_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
