"""
EXP-009: train reshape под valid distribution.

Insight из аудита сегмента rep_6 (2026-05-16):
- n_repair константа в пределах месяца (это месячная метка плана ремонта).
- rep_6 = только May 2024, rep_7 = May 2022 + May 2023.
- Valid Q1 2026 содержит только rep_3 и rep_4.

Стратегия EXP-009:
- Брать EXP-007 setup (pruned features + extra lags ±1..±12 для top sigs).
- Поверх: drop из train все месяцы где n_repair >= 5 (избавляемся от out-of-Q1 режима).
- Sample weights: rep_3 ×1.0, rep_4 ×1.2 (отражает valid).
- Q1 sw снижаем с 2.5 до 1.5 (уже фокусируемся через filter).

Что измеряем:
- cv_mean_nmae на отфильтрованном train (subset матчит valid distribution).
- cv_q1_only - внутри Q1 segment OOF.
- LB после submit.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae, smooth_power_curve_ti
from lgbm_q50 import prepare_features, get_feature_cols, compute_segments
from nwp_bias_correction import apply_bias, add_p_phys_corrected, estimate_bias

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-009"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Top features из EXP-005 + must-have (как в EXP-007)
exp005_summary = json.load(open(ROOT / "experiments/active/EXP-005/summary.json"))
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
    print("[1/5] Prepare features...")
    train_f, valid_f = prepare_features()
    pc = smooth_power_curve_ti(ti=0.20, rated_speed=10.0)
    bt, _ = estimate_bias(train_f, pc)
    train_f = apply_bias(train_f, bt, mode="mean")
    train_f = add_p_phys_corrected(train_f, pc)
    valid_f = apply_bias(valid_f, bt, mode="mean")
    valid_f = add_p_phys_corrected(valid_f, pc)

    # Combined extra lags
    combined = pd.concat([train_f.assign(_src="train"), valid_f.assign(_src="valid")], ignore_index=True)
    combined = add_extra_lags(combined)
    train_f = combined[combined["_src"] == "train"].drop(columns=["_src"]).copy()
    valid_f = combined[combined["_src"] == "valid"].drop(columns=["_src"]).copy()

    # === EXP-009 SPECIFIC: filter train ===
    n_repair_col = "Кол-во_ВЭУ_в_ремонте"
    if n_repair_col not in train_f.columns:
        # try alternative name from features.py
        for alt in ["n_repair", "n_repair_raw"]:
            if alt in train_f.columns:
                n_repair_col = alt
                break
    n_repair = train_f[n_repair_col]
    n_before = len(train_f)
    train_f_filtered = train_f[n_repair <= 4].copy()
    n_after = len(train_f_filtered)
    print(f"[2/5] Train reshape: {n_before} -> {n_after} (dropped {n_before-n_after} rows with n_repair>=5)")

    # Verify valid не задет
    valid_nr = valid_f[n_repair_col].value_counts().sort_index()
    print(f"  Valid n_repair distribution: {dict(valid_nr)}")

    # Verify dropped: какие месяцы убрали?
    dropped = train_f[n_repair >= 5]
    dropped_months = sorted(dropped["dt"].dt.strftime("%Y-%m").unique().tolist())
    print(f"  Dropped months ({len(dropped_months)}): {dropped_months}")

    # === ===

    extra_lag_names = [f"{c}_xlag_{('p' if h > 0 else 'm')}{abs(h)}"
                       for c in EXTRA_LAG_COLS for h in EXTRA_LAG_HOURS]
    all_cols = list(dict.fromkeys(TOP_FEATS + extra_lag_names))
    final_cols = [c for c in all_cols if c in train_f_filtered.columns]
    print(f"[3/5] Final feature count: {len(final_cols)}")

    train_clean = train_f_filtered.dropna(subset=[TARGET_COL]).copy()
    print(f"  After dropna target: {len(train_clean)} rows")
    for c in final_cols:
        if train_clean[c].isna().any():
            med = train_clean[c].median()
            train_clean[c] = train_clean[c].fillna(med)
            valid_f[c] = valid_f[c].fillna(med)

    # Tuned HPs из EXP-004 (Optuna best)
    study = optuna.load_study(
        study_name="exp004_lgbm_q1",
        storage="sqlite:////home/duck/wind_hackathon/experiments/active/EXP-004/optuna.db",
    )
    bp = study.best_params.copy()
    sw_q1_default = bp.pop("sw_q1_factor", 2.0)
    n_boost = bp.pop("n_boost", 2000)
    hps = {
        "objective": "quantile", "alpha": 0.5, "metric": "quantile",
        "verbose": -1, "num_threads": 0, "device": "cpu", "seed": 42,
        **bp,
    }

    # === EXP-009 sw: ниже Q1-weight, выше rep_4 ===
    sw_q1 = 1.5  # снижаем т.к. уже отфильтровали train
    sw = np.ones(len(train_clean))
    is_q1 = train_clean["dt"].dt.month.isin([1, 2, 3]).values
    sw[is_q1] *= sw_q1
    rep4 = (train_clean[n_repair_col] == 4).values
    sw[rep4] *= 1.2  # rep_4 weight up (соотв. valid)
    print(f"  Sample weights: mean={sw.mean():.3f}, max={sw.max():.3f}")

    # CV fit
    print("[4/5] CV fit (GroupKFold by month, 12 splits)...")
    X = train_clean[final_cols].values
    y = train_clean[TARGET_COL].values
    n_avail = train_clean["n_avail"].values
    groups = train_clean["month_key"].values
    cat_idx = [final_cols.index("sector_8")] if "sector_8" in final_cols else "auto"

    gkf = GroupKFold(n_splits=12)
    oof = np.zeros(len(train_clean))
    fold_nmae = []
    fold_q1 = []
    feat_imp = np.zeros(len(final_cols))
    models = []
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
        print(f"  fold {fold:02d}: nMAE={fold_nmae[-1]:.3f}%  iter={m.best_iteration}")

    cv_mean = float(np.mean(fold_nmae))
    cv_std = float(np.std(fold_nmae))
    cv_q1 = float(np.mean(fold_q1)) if fold_q1 else cv_mean
    cv_q1_std = float(np.std(fold_q1)) if fold_q1 else 0

    seg = compute_segments(train_clean[TARGET_COL].values, oof, train_clean)
    imp = sorted(zip(final_cols, feat_imp), key=lambda kv: -kv[1])[:25]

    print(f"\n[5/5] EXP-009 CV mean = {cv_mean:.4f}% ± {cv_std:.4f}%   Q1-only = {cv_q1:.4f}%")
    print(f"vs EXP-007 (9.281 / 8.783): {9.281 - cv_mean:+.4f} / {8.783 - cv_q1:+.4f}")
    print(f"vs EXP-005 (9.209 / 8.737): {9.209 - cv_mean:+.4f} / {8.737 - cv_q1:+.4f}")

    # Predict valid
    Xv = valid_f[final_cols].values
    preds = np.zeros(len(valid_f))
    for m in models:
        preds += m.predict(Xv, num_iteration=m.best_iteration)
    preds /= len(models)
    pred_mw_v = np.clip(preds, 0.0, valid_f["n_avail"].values * TURBINE_RATED_MW)

    valid_out = pd.DataFrame({"dt": valid_f["dt"].values, "pred_mw": pred_mw_v,
                              "p_phys": valid_f["p_phys"].values, "n_avail": valid_f["n_avail"].values})
    valid_out.to_parquet(EXP_DIR / "valid_pred.parquet", index=False)

    oof_out = pd.DataFrame({"dt": train_clean["dt"].values, "y_true": train_clean[TARGET_COL].values,
                            "oof_pred": oof})
    oof_out.to_parquet(EXP_DIR / "oof.parquet", index=False)

    summary = {
        "exp_id": "EXP-009",
        "name": "train_reshape_under_valid_dist",
        "n_features": len(final_cols),
        "n_train_before": n_before,
        "n_train_after": n_after,
        "n_train_clean": len(train_clean),
        "dropped_months": dropped_months,
        "sw_q1_factor": sw_q1,
        "sw_rep4_factor": 1.2,
        "n_boost": n_boost,
        "tuned_hps": hps,
        "cv_mean_nmae": round(cv_mean, 4),
        "cv_std_nmae": round(cv_std, 4),
        "cv_q1_only_mean_nmae": round(cv_q1, 4),
        "cv_q1_only_std_nmae": round(cv_q1_std, 4),
        "fold_nmae": [round(x, 3) for x in fold_nmae],
        "fold_nmae_q1_only": [round(x, 3) for x in fold_q1],
        "segments": seg,
        "top25_importance": [{"feat": f, "gain": round(float(g), 1)} for f, g in imp],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\nSummary: {EXP_DIR / 'summary.json'}")
    print(f"Time: {summary['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
