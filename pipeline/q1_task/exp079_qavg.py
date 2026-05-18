"""
EXP-079: Quantile averaging τ∈{0.40, 0.45, 0.50, 0.55, 0.60} blend (contrarian #3).

Train 5 LGBM с разными tau, average predictions. Idea: для bimodal distribution
average близких quantiles даёт robust estimate "регрессии к центру плотности".

Less risk than single tau=0.5. EXP-021 setup base.
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
EXP_DIR = ROOT / "experiments/active/EXP-079"
EXP_DIR.mkdir(parents=True, exist_ok=True)

SISTER_SOURCES = ["gfs", "icon", "arpege", "knmi", "dmi"]
TAUS = [0.40, 0.45, 0.50, 0.55, 0.60]

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


def main():
    t0 = time.time()
    print("[EXP-079] prepare features ...")
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
    oof_per_tau = {tau: np.zeros(len(train_clean)) for tau in TAUS}
    valid_preds_per_tau = {tau: [] for tau in TAUS}

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        for tau in TAUS:
            hps = {
                "objective": "quantile", "alpha": tau, "metric": "quantile",
                "verbose": -1, "num_threads": 0, "device": "cpu", "seed": 42, **bp,
            }
            ds_tr = lgb.Dataset(X[tr], label=y[tr], weight=sw[tr], categorical_feature=cat_idx)
            ds_te = lgb.Dataset(X[te], label=y[te], weight=sw[te], reference=ds_tr, categorical_feature=cat_idx)
            m = lgb.train(hps, ds_tr, num_boost_round=n_boost, valid_sets=[ds_te],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            pred = m.predict(X[te], num_iteration=m.best_iteration)
            oof_per_tau[tau][te] = np.clip(pred, 0.0, n_avail[te] * TURBINE_RATED_MW)
            valid_preds_per_tau[tau].append(m.predict(valid_f[final_cols].values, num_iteration=m.best_iteration))
        print(f"  fold {fold:02d} done")

    # Average OOF across taus
    oof_avg = np.mean([oof_per_tau[t] for t in TAUS], axis=0)
    nmae_per_tau = {t: float(nmae(y, oof_per_tau[t])) for t in TAUS}
    nmae_avg = float(nmae(y, oof_avg))
    print(f"\nPer-tau OOF nMAE:")
    for t, n in nmae_per_tau.items():
        print(f"  τ={t}: {n:.4f}%")
    print(f"  AVG: {nmae_avg:.4f}%")

    # Q1 2025 only
    q1_2025 = (train_clean["dt"].dt.year == 2025) & train_clean["dt"].dt.month.isin([1, 2, 3])
    nmae_avg_q1_2025 = float(nmae(y[q1_2025.values], oof_avg[q1_2025.values]))
    print(f"  AVG Q1 2025: {nmae_avg_q1_2025:.4f}%")

    # Valid pred avg
    valid_pred_avg_per_tau = {t: np.mean(valid_preds_per_tau[t], axis=0) for t in TAUS}
    valid_pred_avg = np.mean([valid_pred_avg_per_tau[t] for t in TAUS], axis=0)
    pred_mw_v = np.clip(valid_pred_avg, 0.0, valid_f["n_avail"].values * TURBINE_RATED_MW)
    pd.DataFrame({"dt": valid_f["dt"].values, "pred_mw": pred_mw_v,
                  "p_phys": valid_f["p_phys"].values, "n_avail": valid_f["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")
    pd.DataFrame({"dt": train_clean["dt"].values, "y_true": y, "oof_pred": oof_avg}).to_parquet(EXP_DIR / "oof.parquet")

    summary = {
        "exp_id": "EXP-079", "name": "quantile_averaging_5tau",
        "taus": TAUS, "n_features": len(final_cols),
        "nmae_per_tau": nmae_per_tau,
        "cv_mean_nmae": nmae_avg,
        "cv_q1_only_mean_nmae": nmae_avg_q1_2025,
        "elapsed_sec": time.time() - t0,
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Done: {EXP_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
