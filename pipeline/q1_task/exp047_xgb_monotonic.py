"""
EXP-047: XGBoost Quantile (τ=0.5) + monotone_constraints на EXP-021 features.

LGBM не поддерживает monotone_constraints с quantile/regression_l1 (только MSE),
но XGBoost 2.1.3 поддерживает `reg:quantileerror` + monotone_constraints.

Hypothesis: physics-grounded constraint (wind_speed -> +, n_repair -> -, p_phys -> +)
стабилизирует и даёт diversity от LGBM.

Same setup как EXP-021:
- 5 sister NWP (gfs+icon+arpege+knmi+dmi) + disagreement features
- TOP_FEATS + extra lags + sister + disagreement
- GroupKFold/12 by month_key
- n_repair ≤ 5 filter
- sw Q1×1.5, rep4/5×1.2

Difference: XGBoost вместо LGBM + monotone constraints.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae, smooth_power_curve_ti
from lgbm_q50 import prepare_features, compute_segments
from nwp_bias_correction import apply_bias, add_p_phys_corrected, estimate_bias
from sister_features import add_sister_features

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-047"
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
N_REPAIR_COL = "Кол-во_ВЭУ_в_ремонте"


def add_extra_lags_base(df_all):
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


def add_disagreement_features(df, sources):
    key_vars = ["wind_speed_120m", "wind_speed_80m", "wind_gusts_10m"]
    for var in key_vars:
        cols = [var]
        for s in sources:
            c = f"{var}__{s}"
            if c in df.columns:
                cols.append(c)
        if len(cols) < 2:
            continue
        df[f"{var}__mean_all"] = df[cols].mean(axis=1)
        df[f"{var}__std_all"] = df[cols].std(axis=1)
        df[f"{var}__range_all"] = df[cols].max(axis=1) - df[cols].min(axis=1)
        for c in cols:
            tag = c.replace(var, "").lstrip("_") or "ecmwf"
            df[f"{var}__bias_vs_mean__{tag}"] = df[c] - df[f"{var}__mean_all"]
    return df


def build_monotone(final_cols):
    """Monotone tuple: +1 for ws/power features, -1 for repair, 0 otherwise."""
    pos_substr = ("wind_speed_120m", "wind_speed_80m", "wind_speed_180m", "wind_speed_10m",
                  "ws_corrected_120", "ws_at_84", "p_phys", "wind_gusts_10m",
                  "n_avail", "rews_simple", "rews_corr",
                  "__mean_all", "__gfs", "__icon", "__arpege", "__knmi", "__dmi")
    neg_substr = (N_REPAIR_COL,)
    constraints = []
    for c in final_cols:
        if any(s in c for s in neg_substr):
            constraints.append(-1)
        elif any(s in c for s in pos_substr):
            # Only positive monotone if no negation/disagreement (xlag, bias) modifiers
            if "bias_vs_mean" in c or "_std_all" in c or "_range_all" in c:
                constraints.append(0)
            elif "__corr" in c:
                constraints.append(0)
            else:
                constraints.append(1)
        else:
            constraints.append(0)
    return constraints


def main():
    t0 = time.time()
    print("[EXP-047] prepare features ...")
    train_f, valid_f = prepare_features()
    pc = smooth_power_curve_ti(ti=0.20, rated_speed=10.0)
    bt, _ = estimate_bias(train_f, pc)
    train_f = apply_bias(train_f, bt, mode="mean")
    train_f = add_p_phys_corrected(train_f, pc)
    valid_f = apply_bias(valid_f, bt, mode="mean")
    valid_f = add_p_phys_corrected(valid_f, pc)

    combined = pd.concat([train_f.assign(_src="train"), valid_f.assign(_src="valid")], ignore_index=True)
    combined = add_extra_lags_base(combined)
    train_f = combined[combined["_src"] == "train"].drop(columns=["_src"]).copy()
    valid_f = combined[combined["_src"] == "valid"].drop(columns=["_src"]).copy()

    print(f"[EXP-047] adding 5 sister sources ...")
    train_f, valid_f = add_sister_features(train_f, valid_f, sources=SISTER_SOURCES)

    print(f"[EXP-047] adding disagreement features ...")
    train_f = add_disagreement_features(train_f, SISTER_SOURCES)
    valid_f = add_disagreement_features(valid_f, SISTER_SOURCES)
    print(f"[EXP-047] after disagreement: train {train_f.shape}, valid {valid_f.shape}")

    n_repair = train_f[N_REPAIR_COL]
    train_f_filtered = train_f[n_repair <= 5].copy()

    extra_lag_names_base = [f"{c}_xlag_{('p' if h > 0 else 'm')}{abs(h)}"
                            for c in EXTRA_LAG_COLS for h in EXTRA_LAG_HOURS]
    sister_cols = [c for c in train_f.columns if any(f"__{s}" in c for s in SISTER_SOURCES)]
    dis_cols = [c for c in train_f.columns if "__mean_all" in c or "__std_all" in c
                or "__range_all" in c or "__bias_vs_mean" in c]
    all_cols = list(dict.fromkeys(TOP_FEATS + extra_lag_names_base + sister_cols + dis_cols
                                  + [N_REPAIR_COL]))
    final_cols = [c for c in all_cols if c in train_f_filtered.columns]
    print(f"[EXP-047] features: {len(final_cols)} (sister={len(sister_cols)}, dis={len(dis_cols)})")

    train_clean = train_f_filtered.dropna(subset=[TARGET_COL]).copy()
    for c in final_cols:
        if train_clean[c].isna().any() or valid_f[c].isna().any():
            med = train_clean[c].median()
            if pd.isna(med):
                med = 0.0
            train_clean[c] = train_clean[c].fillna(med)
            valid_f[c] = valid_f[c].fillna(med)

    constraints = build_monotone(final_cols)
    n_pos = sum(1 for x in constraints if x == 1)
    n_neg = sum(1 for x in constraints if x == -1)
    print(f"[EXP-047] monotone: +{n_pos}, -{n_neg}, 0×{len(constraints)-n_pos-n_neg}")

    sw = np.ones(len(train_clean))
    is_q1 = train_clean["dt"].dt.month.isin([1, 2, 3]).values
    sw[is_q1] *= 1.5
    rep4 = (train_clean[N_REPAIR_COL] == 4).values
    rep5 = (train_clean[N_REPAIR_COL] == 5).values
    sw[rep4] *= 1.2
    sw[rep5] *= 1.2

    X = train_clean[final_cols].values
    y = train_clean[TARGET_COL].values
    n_avail = train_clean["n_avail"].values
    groups = train_clean["month_key"].values

    # XGBoost params
    xgb_params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": 0.5,
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.7,
        "min_child_weight": 50,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "seed": 42,
        "monotone_constraints": tuple(constraints),
    }
    N_BOOST = 2000

    gkf = GroupKFold(n_splits=12)
    oof = np.zeros(len(train_clean))
    fold_nmae = []
    fold_q1 = []
    feat_imp = np.zeros(len(final_cols))
    models = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        dtr = xgb.DMatrix(X[tr], label=y[tr], weight=sw[tr], feature_names=final_cols)
        dte = xgb.DMatrix(X[te], label=y[te], weight=sw[te], feature_names=final_cols)
        m = xgb.train(
            xgb_params, dtr, num_boost_round=N_BOOST,
            evals=[(dte, "val")], early_stopping_rounds=50, verbose_eval=0,
        )
        pred = m.predict(dte, iteration_range=(0, m.best_iteration + 1))
        pred_mw = np.clip(pred, 0.0, n_avail[te] * TURBINE_RATED_MW)
        oof[te] = pred_mw
        fold_nmae.append(nmae(y[te], pred_mw))
        is_q1_te = train_clean["dt"].iloc[te].dt.month.isin([1, 2, 3]).values
        if is_q1_te.sum() > 50:
            fold_q1.append(nmae(y[te][is_q1_te], pred_mw[is_q1_te]))
        fimp = m.get_score(importance_type="gain")
        for k, v in fimp.items():
            if k in final_cols:
                feat_imp[final_cols.index(k)] += v
        models.append(m)
        print(f"  fold {fold:02d}: nMAE={fold_nmae[-1]:.3f}% (best_it={m.best_iteration})")

    cv_mean = float(np.mean(fold_nmae))
    cv_std = float(np.std(fold_nmae))
    cv_q1 = float(np.mean(fold_q1)) if fold_q1 else cv_mean
    print(f"\n[EXP-047] CV = {cv_mean:.4f}% +/- {cv_std:.4f}% Q1={cv_q1:.4f}%")
    print(f"vs EXP-021 (8.247/8.381) | EXP-024 blend (8.238/-): {8.247-cv_mean:+.4f} / {8.381-cv_q1:+.4f}")

    Xv = valid_f[final_cols].values
    dval = xgb.DMatrix(Xv, feature_names=final_cols)
    preds = np.mean([m.predict(dval, iteration_range=(0, m.best_iteration + 1)) for m in models], axis=0)
    pred_mw_v = np.clip(preds, 0.0, valid_f["n_avail"].values * TURBINE_RATED_MW)
    pd.DataFrame({"dt": valid_f["dt"].values, "pred_mw": pred_mw_v,
                  "p_phys": valid_f["p_phys"].values,
                  "n_avail": valid_f["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")
    pd.DataFrame({"dt": train_clean["dt"].values, "y_true": y,
                  "oof_pred": oof}).to_parquet(EXP_DIR / "oof.parquet")

    seg = compute_segments(y, oof, train_clean)
    imp = sorted(zip(final_cols, feat_imp), key=lambda kv: -kv[1])[:30]
    print(f"\nTop 30 features by gain:")
    for f, g in imp:
        print(f"  {f}: {g:.0f}")

    summary = {
        "exp_id": "EXP-047", "name": "xgb_quantile_monotonic_5src",
        "n_features": len(final_cols), "n_train_clean": len(train_clean),
        "sister_sources": SISTER_SOURCES,
        "monotone_pos": n_pos, "monotone_neg": n_neg,
        "xgb_params": {k: v for k, v in xgb_params.items() if k != "monotone_constraints"},
        "cv_mean_nmae": round(cv_mean, 4), "cv_std_nmae": round(cv_std, 4),
        "cv_q1_only_mean_nmae": round(cv_q1, 4),
        "fold_nmae": [round(x, 4) for x in fold_nmae],
        "fold_q1_nmae": [round(x, 4) for x in fold_q1],
        "top30_features": [(f, round(float(g), 1)) for f, g in imp],
        "segments": seg,
        "wallclock_sec": round(time.time() - t0, 1),
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\n[EXP-047] summary: {EXP_DIR / 'summary.json'}")
    print(f"[EXP-047] wallclock {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
