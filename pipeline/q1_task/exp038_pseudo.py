"""
EXP-038: Pseudo-labeling top-30% confident Q1 2026 valid -> add to train, retrain EXP-021 setup.

Идея: domain adaptation против distribution shift Q1 2026.

Шаги:
1. Прогоняем EXP-021 + EXP-024 (best blend) на valid Q1 2026 -> 2 предсказания.
2. Считаем uncertainty: |EXP-021_pred - EXP-024_pred| и std среди 4 base моделей.
3. Top-30% наиболее confident (низкая uncertainty) - pseudo labels из EXP-024 prediction.
4. Добавляем pseudo-labeled rows в train с weight 0.3 (low confidence vs real label).
5. Переобучаем EXP-021 setup на расширенном train, predict на valid.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae, smooth_power_curve_ti
from lgbm_q50 import prepare_features, compute_segments
from nwp_bias_correction import apply_bias, add_p_phys_corrected, estimate_bias
from sister_features import add_sister_features

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-038"
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

BASE_SOURCES = ["EXP-019", "EXP-020", "EXP-021", "EXP-024"]


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
    train_f["dt"] = pd.to_datetime(train_f["dt"])
    valid_f["dt"] = pd.to_datetime(valid_f["dt"])

    # Загрузим pre-computed predictions от 4 base models на valid
    base_preds = {}
    for name in BASE_SOURCES:
        v = pd.read_parquet(ROOT / f"experiments/archive/promoted/{name}/valid_pred.parquet")
        v["dt"] = pd.to_datetime(v["dt"])
        base_preds[name] = v[["dt", "pred_mw"]].rename(columns={"pred_mw": f"pred_{name}"})

    valid_preds_df = base_preds["EXP-024"].copy()
    for n in ["EXP-019", "EXP-020", "EXP-021"]:
        valid_preds_df = valid_preds_df.merge(base_preds[n], on="dt", how="inner")

    # Uncertainty: std across 4 base models per row
    pred_arr = valid_preds_df[[f"pred_{n}" for n in BASE_SOURCES]].values
    valid_preds_df["uncertainty"] = pred_arr.std(axis=1)
    valid_preds_df["pseudo_target"] = valid_preds_df["pred_EXP-024"]  # use blend as label

    # Top-30% confident (lowest uncertainty)
    threshold = np.quantile(valid_preds_df["uncertainty"], 0.30)
    confident_mask = valid_preds_df["uncertainty"] <= threshold
    print(f"Uncertainty threshold (30%-ile): {threshold:.3f} МВт")
    print(f"Confident rows: {confident_mask.sum()} / {len(valid_preds_df)}")

    confident_dts = valid_preds_df.loc[confident_mask, ["dt", "pseudo_target"]].copy()

    # Берём confident_dts subset из valid_f (имеет все features) + добавляем pseudo target
    valid_for_pseudo = valid_f[valid_f["dt"].isin(confident_dts["dt"])].copy()
    # Strip TARGET_COL if exists в valid_f (например NaN-column от prepare_features)
    if TARGET_COL in valid_for_pseudo.columns:
        valid_for_pseudo = valid_for_pseudo.drop(columns=[TARGET_COL])
    # Map pseudo_target to TARGET_COL directly
    pseudo_map = confident_dts.set_index("dt")["pseudo_target"].to_dict()
    valid_for_pseudo[TARGET_COL] = valid_for_pseudo["dt"].map(pseudo_map)
    valid_for_pseudo["_pseudo"] = True
    print(f"valid_for_pseudo: {len(valid_for_pseudo)}")

    # Train: rep<=5
    n_repair_col = "Кол-во_ВЭУ_в_ремонте"
    train_f_filtered = train_f[train_f[n_repair_col] <= 5].copy()
    train_f_filtered["_pseudo"] = False

    # Concat extended train
    common_cols = sorted(set(train_f_filtered.columns) & set(valid_for_pseudo.columns))
    # Гарантируем что TARGET_COL и ключевые есть
    for must in [TARGET_COL, "_pseudo", "dt", "month_key", n_repair_col, "n_avail"]:
        if must in train_f_filtered.columns and must in valid_for_pseudo.columns and must not in common_cols:
            common_cols.append(must)
    print(f"TARGET_COL in train_f_filtered={TARGET_COL in train_f_filtered.columns}, in valid_for_pseudo={TARGET_COL in valid_for_pseudo.columns}, in common={TARGET_COL in common_cols}")
    train_ext = pd.concat([
        train_f_filtered[common_cols],
        valid_for_pseudo[common_cols],
    ], ignore_index=True)
    print(f"Extended train: {len(train_ext)} (real={len(train_f_filtered)}, pseudo={len(valid_for_pseudo)})")
    print(f"TARGET_COL in train_ext: {TARGET_COL in train_ext.columns}")

    extra_lag_names_base = [f"{c}_xlag_{('p' if h > 0 else 'm')}{abs(h)}"
                            for c in EXTRA_LAG_COLS for h in EXTRA_LAG_HOURS]
    sister_cols = [c for c in train_ext.columns if any(f"__{s}" in c for s in SISTER_SOURCES)]
    dis_cols = [c for c in train_ext.columns if "__mean_all" in c or "__std_all" in c or "__range_all" in c]
    all_cols = list(dict.fromkeys(TOP_FEATS + extra_lag_names_base + sister_cols + dis_cols))
    final_cols = [c for c in all_cols if c in train_ext.columns]
    print(f"[EXP-038] {len(final_cols)} features ...", flush=True)

    train_clean = train_ext.dropna(subset=[TARGET_COL]).copy()
    train_clean = train_clean.copy()
    for c in final_cols:
        if train_clean[c].isna().any() or valid_f[c].isna().any():
            med = train_clean[c].median()
            if pd.isna(med): med = 0.0
            train_clean[c] = train_clean[c].fillna(med)
            valid_f[c] = valid_f[c].fillna(med)

    study = optuna.load_study(
        study_name="exp004_lgbm_q1",
        storage="sqlite:////home/duck/wind_hackathon/experiments/archive/rejected/duplicate/EXP-004/optuna.db",
    )
    bp = study.best_params.copy()
    bp.pop("sw_q1_factor", None)
    n_boost = bp.pop("n_boost", 2000)
    bp["feature_fraction"] = 0.65
    bp["min_data_in_leaf"] = 250
    hps = {"objective": "quantile", "alpha": 0.5, "metric": "quantile",
           "verbose": -1, "num_threads": 0, "device": "cpu", "seed": 42, **bp}

    sw = np.ones(len(train_clean))
    is_q1 = train_clean["dt"].dt.month.isin([1, 2, 3]).values
    sw[is_q1] *= 1.5
    rep4 = (train_clean[n_repair_col] == 4).values
    rep5 = (train_clean[n_repair_col] == 5).values
    sw[rep4] *= 1.2; sw[rep5] *= 1.2
    is_pseudo = train_clean["_pseudo"].values
    sw[is_pseudo] *= 0.3  # downweight pseudo labels

    print(f"Sample weights: real_mean={sw[~is_pseudo].mean():.3f}, pseudo_mean={sw[is_pseudo].mean():.3f}")

    X = train_clean[final_cols].values
    y = train_clean[TARGET_COL].values
    n_avail = train_clean["n_avail"].values
    # для CV groupkfold по month_key - pseudo rows из Q1 2026 = "2026-01"/"2026-02"/"2026-03" groups
    groups = train_clean["month_key"].values
    cat_idx = [final_cols.index("sector_8")] if "sector_8" in final_cols else "auto"

    # Только real rows используем для metric calculation (pseudo не реальные)
    is_real_for_metric = (~is_pseudo)

    gkf = GroupKFold(n_splits=12)
    oof = np.zeros(len(train_clean))
    fold_nmae = []; fold_q1 = []; models = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        # Не оцениваем metric на pseudo rows
        te_real = te[is_real_for_metric[te]]
        ds_tr = lgb.Dataset(X[tr], label=y[tr], weight=sw[tr], categorical_feature=cat_idx)
        ds_te = lgb.Dataset(X[te], label=y[te], weight=sw[te], reference=ds_tr, categorical_feature=cat_idx)
        m = lgb.train(hps, ds_tr, num_boost_round=n_boost, valid_sets=[ds_te],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred = m.predict(X[te], num_iteration=m.best_iteration)
        pred_mw = np.clip(pred, 0.0, n_avail[te] * TURBINE_RATED_MW)
        oof[te] = pred_mw
        if len(te_real) > 50:
            fold_nmae.append(nmae(y[te_real], pred_mw[is_real_for_metric[te]]))
        is_q1_te = train_clean["dt"].iloc[te].dt.month.isin([1, 2, 3]).values & is_real_for_metric[te]
        if is_q1_te.sum() > 50:
            fold_q1.append(nmae(y[te][is_q1_te], pred_mw[is_q1_te]))
        models.append(m)
        print(f"  fold {fold:02d}: nMAE={fold_nmae[-1]:.3f}% iters={m.best_iteration}", flush=True)

    cv_mean = float(np.mean(fold_nmae)); cv_std = float(np.std(fold_nmae))
    cv_q1 = float(np.mean(fold_q1)) if fold_q1 else cv_mean
    print(f"\n[EXP-038] pseudo CV = {cv_mean:.4f}% Q1={cv_q1:.4f}%", flush=True)
    print(f"vs EXP-021 (8.242 / 8.351): {8.242-cv_mean:+.4f} / {8.351-cv_q1:+.4f}", flush=True)

    Xv = valid_f[final_cols].values
    preds = np.mean([m.predict(Xv, num_iteration=m.best_iteration) for m in models], axis=0)
    pred_mw_v = np.clip(preds, 0.0, valid_f["n_avail"].values * TURBINE_RATED_MW)
    pd.DataFrame({"dt": valid_f["dt"].values, "pred_mw": pred_mw_v,
                  "p_phys": valid_f["p_phys"].values, "n_avail": valid_f["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")
    # OOF только real
    oof_real_dt = train_clean.loc[is_real_for_metric, "dt"].values
    oof_real_y = y[is_real_for_metric]
    oof_real_pred = oof[is_real_for_metric]
    pd.DataFrame({"dt": oof_real_dt, "y_true": oof_real_y, "oof_pred": oof_real_pred}).to_parquet(EXP_DIR / "oof.parquet")

    summary = {
        "exp_id": "EXP-038", "name": "pseudo_label_top30_confident_valid",
        "n_features": len(final_cols), "n_train_real": int((~is_pseudo).sum()), "n_pseudo": int(is_pseudo.sum()),
        "uncertainty_threshold": round(float(threshold), 3),
        "pseudo_weight": 0.3,
        "cv_mean_nmae": round(cv_mean, 4), "cv_std_nmae": round(cv_std, 4),
        "cv_q1_only_mean_nmae": round(cv_q1, 4),
        "fold_nmae": [round(x, 3) for x in fold_nmae],
        "fold_nmae_q1_only": [round(x, 3) for x in fold_q1],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Summary: {EXP_DIR / 'summary.json'}, time {summary['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
