"""
EXP-070b: NGBoost heteroscedastic σ as feature -> second-stage LGBM.

Идея: NGBoost outputs (μ, σ) jointly. Pass σ(X) as feature
to second-stage LGBM. This adds "uncertainty axis" orthogonal to bias.

Pipeline:
1. Train NGBoost on subset of features (most informative ~30) - too many = slow.
2. Get sigma(X) for train + valid.
3. Add as feature to EXP-021 setup.
4. Refit LGBM.

NGBoost CPU is SLOW (5-10x LGBM). Limit n_estimators=200, only key features.
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae, smooth_power_curve_ti
from lgbm_q50 import prepare_features
from nwp_bias_correction import apply_bias, add_p_phys_corrected, estimate_bias
from sister_features import add_sister_features

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-070b"
EXP_DIR.mkdir(parents=True, exist_ok=True)

SISTER_SOURCES = ["gfs", "icon", "arpege", "knmi", "dmi"]

# Top key features for NGBoost (compact, fast)
NGB_KEY_FEATS = [
    "wind_speed_120m", "wind_speed_80m",
    "ws_corrected_120", "p_phys_corrected", "p_phys",
    "wind_gusts_10m", "alpha_80_120", "rho_air",
    "rews_corr", "ws_at_84",
    "wind_speed_120m__gfs", "wind_speed_120m__icon", "wind_speed_120m__arpege",
    "wind_speed_120m__knmi", "wind_speed_120m__dmi",
    "wind_gusts_10m__gfs", "wind_gusts_10m__icon",
    "n_avail", "available_frac", "month", "hour_of_day",
]


def main():
    t0 = time.time()
    print("[EXP-070b] prepare features + sister (no extra lags) ...")
    train_f, valid_f = prepare_features()
    pc = smooth_power_curve_ti(ti=0.20, rated_speed=10.0)
    bt, _ = estimate_bias(train_f, pc)
    train_f = apply_bias(train_f, bt, mode="mean"); train_f = add_p_phys_corrected(train_f, pc)
    valid_f = apply_bias(valid_f, bt, mode="mean"); valid_f = add_p_phys_corrected(valid_f, pc)
    train_f, valid_f = add_sister_features(train_f, valid_f, sources=SISTER_SOURCES)

    n_repair_col = "Кол-во_ВЭУ_в_ремонте"
    train_f = train_f[train_f[n_repair_col] <= 5].copy()
    train_clean = train_f.dropna(subset=[TARGET_COL]).copy()

    # Subset to NGB key features (drop missing) + fillna
    ngb_feats = [c for c in NGB_KEY_FEATS if c in train_clean.columns and c in valid_f.columns]
    print(f"NGB features: {len(ngb_feats)}")
    for c in ngb_feats:
        med = train_clean[c].median()
        if pd.isna(med): med = 0.0
        train_clean[c] = train_clean[c].fillna(med)
        valid_f[c] = valid_f[c].fillna(med)

    # === NGBoost: predict (μ, σ) ===
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from sklearn.tree import DecisionTreeRegressor
    print(f"[EXP-070b] training NGBoost ({len(train_clean)} rows, {len(ngb_feats)} feats, n_est=150) ...")
    ngb = NGBRegressor(
        Dist=Normal, n_estimators=150,
        learning_rate=0.05, verbose=False,
        Base=DecisionTreeRegressor(max_depth=4, min_samples_leaf=200),
        random_state=42,
    )
    t1 = time.time()
    ngb.fit(train_clean[ngb_feats].values, train_clean[TARGET_COL].values)
    print(f"  NGBoost fit in {time.time()-t1:.1f}s")

    # Predict sigma(X) for train (OOF approx - use full-fit transduction; for honest OOF would need CV)
    train_dist = ngb.pred_dist(train_clean[ngb_feats].values)
    valid_dist = ngb.pred_dist(valid_f[ngb_feats].values)
    train_clean["ngb_mu"] = train_dist.loc
    train_clean["ngb_sigma"] = train_dist.scale
    valid_f["ngb_mu"] = valid_dist.loc
    valid_f["ngb_sigma"] = valid_dist.scale
    print(f"  ngb_sigma stats: mean={train_clean['ngb_sigma'].mean():.3f}, std={train_clean['ngb_sigma'].std():.3f}")

    # === Use ngb_mu/sigma as features in EXP-021 setup ===
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
        "ngb_sigma",  # EXP-070bb: drop ngb_mu (leak), keep sigma only  # NEW
    ]
    sister_cols = [c for c in train_clean.columns if any(f"__{s}" in c for s in SISTER_SOURCES)]
    final_cols = list(dict.fromkeys(top15 + must_have + sister_cols))
    final_cols = [c for c in final_cols if c in train_clean.columns and c in valid_f.columns]
    print(f"[EXP-070b] LGBM features: {len(final_cols)}")

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
    print(f"\n[EXP-070b] CV = {cv_mean:.4f}% +/- {cv_std:.4f}% Q1={cv_q1:.4f}%")
    print(f"vs EXP-021 (8.238 / 8.242): {8.238-cv_mean:+.4f} / {8.242-cv_q1:+.4f}")

    Xv = valid_f[final_cols].values
    preds = np.mean([m.predict(Xv, num_iteration=m.best_iteration) for m in models], axis=0)
    pred_mw_v = np.clip(preds, 0.0, valid_f["n_avail"].values * TURBINE_RATED_MW)
    pd.DataFrame({"dt": valid_f["dt"].values, "pred_mw": pred_mw_v,
                  "p_phys": valid_f["p_phys"].values, "n_avail": valid_f["n_avail"].values,
                  "ngb_mu": valid_f["ngb_mu"].values, "ngb_sigma": valid_f["ngb_sigma"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")
    pd.DataFrame({"dt": train_clean["dt"].values, "y_true": y, "oof_pred": oof,
                  "ngb_mu": train_clean["ngb_mu"].values, "ngb_sigma": train_clean["ngb_sigma"].values}).to_parquet(EXP_DIR / "oof.parquet")

    # NGB feature importance
    imp = sorted(zip(final_cols, feat_imp), key=lambda kv: -kv[1])
    print("\n=== ngb_mu/ngb_sigma importance ===")
    for f, g in imp:
        if f in ["ngb_mu", "ngb_sigma"]:
            print(f"  {f}: {g:.0f}")

    summary = {
        "exp_id": "EXP-070b", "name": "ngboost_sigma_feature",
        "n_features": len(final_cols),
        "cv_mean_nmae": cv_mean, "cv_std_nmae": cv_std,
        "cv_q1_only_mean_nmae": cv_q1,
        "elapsed_sec": time.time() - t0,
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Done: {EXP_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
