"""
EXP-036: 2-level stacking. Meta-LGBM поверх OOFs от EXP-019/020/021/024.

Идея: brand-new model learns optimal NON-LINEAR combination of base predictions
plus context features (hour_of_day, month, n_repair, sector_8, ws_120m).
Учится понимать "когда какая модель лучше" (e.g. EXP-019 better в Q4, EXP-024 в Q1).

Avoiding overfit: малая модель (depth 4, leaves 16, min_data 500),
GroupKFold по month_key.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae, smooth_power_curve_ti
from lgbm_q50 import prepare_features
from nwp_bias_correction import apply_bias, add_p_phys_corrected, estimate_bias

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-036"
EXP_DIR.mkdir(parents=True, exist_ok=True)

BASE_SOURCES = [
    ("EXP-019", ROOT / "experiments/archive/promoted/EXP-019"),
    ("EXP-020", ROOT / "experiments/archive/promoted/EXP-020"),
    ("EXP-021", ROOT / "experiments/archive/promoted/EXP-021"),
    ("EXP-024", ROOT / "experiments/archive/promoted/EXP-024"),
]

CONTEXT_FEATS = ["wind_speed_120m", "wind_speed_80m", "ws_corrected_120",
                 "p_phys_corrected", "n_avail", "hour_of_day", "month",
                 "sector_8", "rho_air", "alpha_80_120", "temperature_80m",
                 "Кол-во_ВЭУ_в_ремонте"]


def main():
    t0 = time.time()
    print("[EXP-036] preparing context features ...", flush=True)
    train_f, valid_f = prepare_features()
    pc = smooth_power_curve_ti(ti=0.20, rated_speed=10.0)
    bt, _ = estimate_bias(train_f, pc)
    train_f = apply_bias(train_f, bt, mode="mean"); train_f = add_p_phys_corrected(train_f, pc)
    valid_f = apply_bias(valid_f, bt, mode="mean"); valid_f = add_p_phys_corrected(valid_f, pc)

    # Загрузим OOF + valid preds от base моделей
    oofs = {}; valids = {}
    for name, path in BASE_SOURCES:
        o = pd.read_parquet(path / "oof.parquet")
        v = pd.read_parquet(path / "valid_pred.parquet")
        o["dt"] = pd.to_datetime(o["dt"])
        v["dt"] = pd.to_datetime(v["dt"])
        oofs[name] = o; valids[name] = v
        print(f"  {name}: oof={len(o)}, valid={len(v)}", flush=True)

    # Merge train base context + OOFs
    train_f["dt"] = pd.to_datetime(train_f["dt"])
    valid_f["dt"] = pd.to_datetime(valid_f["dt"])
    n_repair_col = "Кол-во_ВЭУ_в_ремонте"

    # Filter train: rep<=5, dropna target
    train_f = train_f[train_f[n_repair_col] <= 5].dropna(subset=[TARGET_COL]).copy()

    # Merge OOF predictions
    df_tr = train_f[["dt", TARGET_COL, "month_key"] + CONTEXT_FEATS].copy()
    df_tr = df_tr.loc[:, ~df_tr.columns.duplicated()]
    for name in [s[0] for s in BASE_SOURCES]:
        o = oofs[name][["dt", "oof_pred"]].rename(columns={"oof_pred": f"oof_{name}"})
        df_tr = df_tr.merge(o, on="dt", how="inner")
    print(f"\nTrain after merge: {len(df_tr)}", flush=True)

    val_cols = [c for c in (["dt", "n_avail"] + CONTEXT_FEATS) if c in valid_f.columns]
    df_val = valid_f[val_cols].copy()
    df_val = df_val.loc[:, ~df_val.columns.duplicated()]
    for name in [s[0] for s in BASE_SOURCES]:
        vp = valids[name][["dt", "pred_mw"]].rename(columns={"pred_mw": f"pred_{name}"})
        df_val = df_val.merge(vp, on="dt", how="inner")
    print(f"Valid after merge: {len(df_val)}", flush=True)

    meta_oof_cols = [f"oof_{s[0]}" for s in BASE_SOURCES]
    meta_val_cols = [f"pred_{s[0]}" for s in BASE_SOURCES]
    final_feats = meta_oof_cols + CONTEXT_FEATS

    # Fillna для context
    for c in CONTEXT_FEATS:
        if df_tr[c].isna().any() or df_val[c].isna().any():
            med = df_tr[c].median()
            if pd.isna(med): med = 0.0
            df_tr[c] = df_tr[c].fillna(med)
            df_val[c] = df_val[c].fillna(med)

    # Сделаем валидационные значения в том же порядке (rename pred->oof для unified scoring)
    rename_map = {f"pred_{s[0]}": f"oof_{s[0]}" for s in BASE_SOURCES}
    df_val_aligned = df_val.rename(columns=rename_map)

    hps = {
        "objective": "regression_l1", "metric": "mae",
        "verbose": -1, "num_threads": 0, "device": "cpu", "seed": 42,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 5,
        "min_data_in_leaf": 200,
        "lambda_l1": 1.0,
        "lambda_l2": 5.0,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 3,
    }
    n_boost = 2000

    sw = np.ones(len(df_tr))
    is_q1 = df_tr["dt"].dt.month.isin([1, 2, 3]).values
    sw[is_q1] *= 1.5
    rep4 = (df_tr[n_repair_col] == 4).values
    rep5 = (df_tr[n_repair_col] == 5).values
    sw[rep4] *= 1.2; sw[rep5] *= 1.2

    X = df_tr[final_feats].values
    y = df_tr[TARGET_COL].values
    n_avail = df_tr["n_avail"].values if "n_avail" in df_tr.columns else None
    if n_avail is None:
        n_avail = train_f.set_index("dt").loc[df_tr["dt"]]["n_avail"].values
    groups = df_tr["month_key"].values
    cat_idx = [final_feats.index("sector_8")] if "sector_8" in final_feats else "auto"

    gkf = GroupKFold(n_splits=12)
    oof = np.zeros(len(df_tr))
    fold_nmae = []; fold_q1 = []; models = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        ds_tr = lgb.Dataset(X[tr], label=y[tr], weight=sw[tr], categorical_feature=cat_idx)
        ds_te = lgb.Dataset(X[te], label=y[te], weight=sw[te], reference=ds_tr, categorical_feature=cat_idx)
        m = lgb.train(hps, ds_tr, num_boost_round=n_boost, valid_sets=[ds_te],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        pred = m.predict(X[te], num_iteration=m.best_iteration)
        pred_mw = np.clip(pred, 0.0, n_avail[te] * TURBINE_RATED_MW)
        oof[te] = pred_mw
        fold_nmae.append(nmae(y[te], pred_mw))
        is_q1_te = df_tr["dt"].iloc[te].dt.month.isin([1, 2, 3]).values
        if is_q1_te.sum() > 50:
            fold_q1.append(nmae(y[te][is_q1_te], pred_mw[is_q1_te]))
        models.append(m)
        print(f"  fold {fold:02d}: nMAE={fold_nmae[-1]:.3f}% iters={m.best_iteration}", flush=True)

    cv_mean = float(np.mean(fold_nmae)); cv_std = float(np.std(fold_nmae))
    cv_q1 = float(np.mean(fold_q1)) if fold_q1 else cv_mean
    print(f"\n[EXP-036] stacking CV = {cv_mean:.4f}% Q1={cv_q1:.4f}%", flush=True)
    print(f"vs EXP-024 (7.3122 LB / 8.2378 CV): {8.2378-cv_mean:+.4f}", flush=True)

    # Predict on valid
    Xv = df_val_aligned[final_feats].values
    nv_avail = df_val["n_avail"].values
    preds = np.mean([m.predict(Xv, num_iteration=m.best_iteration) for m in models], axis=0)
    pred_mw_v = np.clip(preds, 0.0, nv_avail * TURBINE_RATED_MW)
    pd.DataFrame({"dt": df_val["dt"].values, "pred_mw": pred_mw_v,
                  "n_avail": nv_avail}).to_parquet(EXP_DIR / "valid_pred.parquet")
    pd.DataFrame({"dt": df_tr["dt"].values, "y_true": y, "oof_pred": oof}).to_parquet(EXP_DIR / "oof.parquet")

    # Top-5 features
    imp = sorted(zip(final_feats, models[0].feature_importance("gain")), key=lambda kv: -kv[1])[:10]
    print(f"\nTop10 feat importance (fold 0):")
    for f, g in imp: print(f"  {f}: {g:.0f}")

    summary = {
        "exp_id": "EXP-036", "name": "stacking_2lvl_meta_lgbm",
        "n_features": len(final_feats), "n_train": len(df_tr),
        "base_models": [s[0] for s in BASE_SOURCES],
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
