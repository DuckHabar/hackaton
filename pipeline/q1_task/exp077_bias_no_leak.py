"""
EXP-077: bias correction inside fold (fresh-eyes #2 fix - no leak).

Текущий EXP-021 setup имеет:
- estimate_bias(train_f, pc) на ВСЁМ train ДО fold split.
- ws_corrected_120 = ws_raw + bias_table[(sector, month)]
- bias_table[(S, M)] был fitted на rows с теми же (S, M) -> если эти rows попадают в test fold -> INFO LEAK target -> fold test prediction.

Fix: estimate_bias внутри fold loop, fit ТОЛЬКО на train_fold, apply на test_fold.

Compare with EXP-021 (potentially leaky) baseline 8.2378.

Если honest CV becomes >8.5 - мы понимаем что previously CV was overstated.
Это может объяснить почему 9 экспериментов подряд CV win -> LB regression.
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
from nwp_bias_correction import estimate_bias, apply_bias, add_p_phys_corrected
from sister_features import add_sister_features

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-077"
EXP_DIR.mkdir(parents=True, exist_ok=True)

SISTER_SOURCES = ["gfs", "icon", "arpege", "knmi", "dmi"]

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
    print("[EXP-077] prepare features (NO global bias correction yet) ...")
    train_f, valid_f = prepare_features()
    pc = smooth_power_curve_ti(ti=0.20, rated_speed=10.0)
    # СКИПАЕМ global estimate_bias - будем делать в fold loop

    # Но для add_p_phys_corrected нужен ws_corrected_120 -> создадим placeholder
    train_f["ws_corrected_120"] = train_f["wind_speed_120m"]  # initialize to raw
    train_f["ws_corrected_120_corr"] = train_f["wind_speed_120m"]
    valid_f["ws_corrected_120"] = valid_f["wind_speed_120m"]
    valid_f["ws_corrected_120_corr"] = valid_f["wind_speed_120m"]
    # add_p_phys_corrected nedded for some features
    train_f = add_p_phys_corrected(train_f, pc)
    valid_f = add_p_phys_corrected(valid_f, pc)

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
    print(f"[EXP-077] features: {len(final_cols)}")

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

    y = train_clean[TARGET_COL].values
    n_avail = train_clean["n_avail"].values
    groups = train_clean["month_key"].values
    cat_idx = [final_cols.index("sector_8")] if "sector_8" in final_cols else "auto"

    gkf = GroupKFold(n_splits=12)
    oof = np.zeros(len(train_clean))
    fold_nmae, fold_q1, feat_imp, models = [], [], np.zeros(len(final_cols)), []
    bias_tables_per_fold = []
    valid_preds_per_fold = []

    # Pre-compute mapping from train_clean index to row for bias fitting
    train_clean_idx = train_clean.index
    train_for_bias_full = train_f_filt.loc[train_clean_idx].copy()

    for fold, (tr, te) in enumerate(gkf.split(train_clean, y, groups)):
        # === FIT BIAS ON TR FOLD ONLY ===
        train_fold_for_bias = train_for_bias_full.iloc[tr].copy()
        # estimate_bias uses TARGET_COL
        bias_table, _ = estimate_bias(train_fold_for_bias, pc)
        # Apply bias to BOTH tr and te subsets
        tr_sub = train_clean.iloc[tr].copy()
        te_sub = train_clean.iloc[te].copy()
        tr_sub = apply_bias(tr_sub, bias_table, mode="mean")
        te_sub = apply_bias(te_sub, bias_table, mode="mean")
        tr_sub = add_p_phys_corrected(tr_sub, pc)
        te_sub = add_p_phys_corrected(te_sub, pc)
        # Recompute ws_corrected_120_xlag in extra_lag_names ? - depends on lag features
        # For honest cv: drop lag features that depend on ws_corrected_120 (they're fitted on full)
        # OR re-build lags on tr+te. Easiest: drop those derived from ws_corrected_120 for this fold.
        excluded_lag_cols = [c for c in final_cols if "ws_corrected_120_xlag" in c]
        # NOTE: simpler: keep them as-is (they were computed on global combined; not target-derived)
        # The lag columns themselves don't use target; only ws_corrected_120 base does.
        # So only the BASE column "ws_corrected_120" needs re-applying.

        # Now also bias for valid (for final pred - we'll fit on full train + all folds)
        # Save bias_table per fold for later (we'll use ensemble of fold biases for valid)
        bias_tables_per_fold.append(bias_table)

        # Get feature arrays (re-extract since we updated ws_corrected_120)
        X_tr = tr_sub[final_cols].values
        X_te = te_sub[final_cols].values

        ds_tr = lgb.Dataset(X_tr, label=y[tr], weight=sw[tr], categorical_feature=cat_idx)
        ds_te = lgb.Dataset(X_te, label=y[te], weight=sw[te], reference=ds_tr, categorical_feature=cat_idx)
        m = lgb.train(hps, ds_tr, num_boost_round=n_boost, valid_sets=[ds_te],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred = m.predict(X_te, num_iteration=m.best_iteration)
        pred_mw = np.clip(pred, 0.0, n_avail[te] * TURBINE_RATED_MW)
        oof[te] = pred_mw
        fold_nmae.append(nmae(y[te], pred_mw))
        is_q1_te = train_clean["dt"].iloc[te].dt.month.isin([1, 2, 3]).values
        if is_q1_te.sum() > 50:
            fold_q1.append(nmae(y[te][is_q1_te], pred_mw[is_q1_te]))
        feat_imp += m.feature_importance("gain")
        models.append(m)

        # Apply bias to valid for this fold (we'll average later)
        v_sub = valid_f.copy()
        v_sub = apply_bias(v_sub, bias_table, mode="mean")
        v_sub = add_p_phys_corrected(v_sub, pc)
        Xv = v_sub[final_cols].values
        pv = m.predict(Xv, num_iteration=m.best_iteration)
        valid_preds_per_fold.append(pv)
        print(f"  fold {fold:02d}: nMAE={fold_nmae[-1]:.3f}%")

    cv_mean = float(np.mean(fold_nmae)); cv_std = float(np.std(fold_nmae))
    cv_q1 = float(np.mean(fold_q1)) if fold_q1 else cv_mean
    print(f"\n[EXP-077] HONEST CV = {cv_mean:.4f}% +/- {cv_std:.4f}% Q1={cv_q1:.4f}%")
    print(f"vs EXP-021 (8.238 - possibly leaky): {8.238-cv_mean:+.4f}")

    # Valid pred = average across folds
    preds_v = np.mean(valid_preds_per_fold, axis=0)
    pred_mw_v = np.clip(preds_v, 0.0, valid_f["n_avail"].values * TURBINE_RATED_MW)
    pd.DataFrame({"dt": valid_f["dt"].values, "pred_mw": pred_mw_v,
                  "p_phys": valid_f["p_phys"].values, "n_avail": valid_f["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")
    pd.DataFrame({"dt": train_clean["dt"].values, "y_true": y, "oof_pred": oof}).to_parquet(EXP_DIR / "oof.parquet")

    summary = {
        "exp_id": "EXP-077", "name": "bias_no_leak_fold_aware",
        "n_features": len(final_cols),
        "cv_mean_nmae": cv_mean, "cv_std_nmae": cv_std,
        "cv_q1_only_mean_nmae": cv_q1,
        "elapsed_sec": time.time() - t0,
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Done: {EXP_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
