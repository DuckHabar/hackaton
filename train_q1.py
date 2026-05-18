"""обучение модели прогноза q1 (sister 5 источников + калибровка по бинам).

вход: data/processed/train_with_physics.parquet, valid_with_physics.parquet
      data/processed/nwp_sister/{gfs,icon,arpege,knmi,dmi}.parquet
      model/q1/optuna.db (заранее подобранные гиперпараметры)
      model/q1/exp005_summary.json (отсортированные по важности признаки)
выход: model/q1/lgbm_q1.txt          (бустер)
       model/q1/oof.parquet           (предсказания на отложенных кусках)
       model/q1/valid_pred.parquet    (предсказания на проверочной части)
       model/q1/bias_map.json         (поправка по бинам ветра)
       model/q1/feature_columns.json  (порядок признаков)
       model/q1/meta.json             (метаданные)
"""

import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "pipeline" / "q1_task"))
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae, smooth_power_curve_ti
from lgbm_q50 import prepare_features, compute_segments
from nwp_bias_correction import apply_bias, add_p_phys_corrected, estimate_bias
from sister_features import add_sister_features

MODEL_DIR = ROOT / "model" / "q1"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OPTUNA_DB = MODEL_DIR / "optuna.db"
EXP005_FEATS = MODEL_DIR / "exp005_summary.json"

BINS = [0, 3, 5, 7, 9, 11, 13, 16, 30]
RATED = 90.09

exp005_summary = json.load(open(EXP005_FEATS))
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


def fit_bias_by_bin(y_true, pred, ws, min_n=30):
    bins = np.digitize(ws, bins=BINS[1:-1])
    out = {}
    for b in sorted(set(bins.tolist())):
        mask = bins == b
        if mask.sum() >= min_n:
            out[int(b)] = float(np.median(y_true[mask] - pred[mask]))
        else:
            out[int(b)] = 0.0
    return out


def main():
    t0 = time.time()
    print("подготовка признаков")
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

    print(f"подключаем пять соседних источников {SISTER_SOURCES}")
    train_f, valid_f = add_sister_features(train_f, valid_f, sources=SISTER_SOURCES)
    print(f"после: обучение {train_f.shape}, проверка {valid_f.shape}")

    n_repair_col = "Кол-во_ВЭУ_в_ремонте"
    n_repair = train_f[n_repair_col]
    train_f_filtered = train_f[n_repair <= 5].copy()

    extra_lag_names_base = [f"{c}_xlag_{('p' if h > 0 else 'm')}{abs(h)}"
                            for c in EXTRA_LAG_COLS for h in EXTRA_LAG_HOURS]
    sister_cols = [c for c in train_f.columns if any(f"__{s}" in c for s in SISTER_SOURCES)]
    all_cols = list(dict.fromkeys(TOP_FEATS + extra_lag_names_base + sister_cols))
    final_cols = [c for c in all_cols if c in train_f_filtered.columns]
    print(f"всего признаков: {len(final_cols)} (из них соседних: {len(sister_cols)})")

    train_clean = train_f_filtered.dropna(subset=[TARGET_COL]).copy()
    for c in final_cols:
        if train_clean[c].isna().any() or valid_f[c].isna().any():
            med = train_clean[c].median()
            if pd.isna(med):
                med = 0.0
            train_clean[c] = train_clean[c].fillna(med)
            valid_f[c] = valid_f[c].fillna(med)

    study = optuna.load_study(
        study_name="exp004_lgbm_q1",
        storage=f"sqlite:///{OPTUNA_DB}",
    )
    bp = study.best_params.copy()
    bp.pop("sw_q1_factor", None)
    n_boost = bp.pop("n_boost", 2000)
    bp["feature_fraction"] = 0.75
    bp.setdefault("min_data_in_leaf", 200)
    hps = {
        "objective": "quantile", "alpha": 0.5, "metric": "quantile",
        "verbose": -1, "num_threads": 0, "device": "cpu", "seed": 42,
        **bp,
    }

    sw = np.ones(len(train_clean))
    is_q1 = train_clean["dt"].dt.month.isin([1, 2, 3]).values
    sw[is_q1] *= 1.5
    rep4 = (train_clean[n_repair_col] == 4).values
    rep5 = (train_clean[n_repair_col] == 5).values
    sw[rep4] *= 1.2
    sw[rep5] *= 1.2

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
        print(f"  фолд {fold:02d}: nMAE={fold_nmae[-1]:.3f}%")

    cv_mean = float(np.mean(fold_nmae))
    cv_std = float(np.std(fold_nmae))
    cv_q1 = float(np.mean(fold_q1)) if fold_q1 else cv_mean
    print(f"\nсредняя CV-ошибка: {cv_mean:.4f}% +/- {cv_std:.4f}%, только Q1: {cv_q1:.4f}%")

    print("предсказание на проверочной части")
    Xv = valid_f[final_cols].values
    preds = np.mean([m.predict(Xv, num_iteration=m.best_iteration) for m in models], axis=0)
    pred_mw_v = np.clip(preds, 0.0, valid_f["n_avail"].values * TURBINE_RATED_MW)

    ws_train = train_clean["wind_speed_120m"].values
    is_cal = (train_clean["dt"].dt.month.isin([1, 2, 3])
              & train_clean["dt"].dt.year.isin([2023, 2024, 2025])).values
    bias_map = fit_bias_by_bin(y[is_cal], oof[is_cal], ws_train[is_cal])

    best_model = models[int(np.argmin(fold_nmae))]
    best_model.save_model(str(MODEL_DIR / "lgbm_q1.txt"))
    pd.DataFrame({"dt": valid_f["dt"].values, "pred_mw": pred_mw_v,
                  "p_phys": valid_f["p_phys"].values,
                  "n_avail": valid_f["n_avail"].values,
                  "ws_120m": valid_f["wind_speed_120m"].values}).to_parquet(MODEL_DIR / "valid_pred.parquet")
    pd.DataFrame({"dt": train_clean["dt"].values, "y_true": y, "oof_pred": oof}).to_parquet(MODEL_DIR / "oof.parquet")
    (MODEL_DIR / "bias_map.json").write_text(
        json.dumps({str(k): v for k, v in bias_map.items()}, ensure_ascii=False, indent=2)
    )
    (MODEL_DIR / "feature_columns.json").write_text(
        json.dumps(final_cols, ensure_ascii=False, indent=2)
    )

    seg = compute_segments(y, oof, train_clean)
    imp = sorted(zip(final_cols, feat_imp), key=lambda kv: -kv[1])[:40]
    summary = {
        "name": "sister_5src_gfs_icon_arpege_knmi_dmi",
        "n_features": len(final_cols),
        "n_train_clean": len(train_clean),
        "sister_sources": SISTER_SOURCES,
        "n_sister_features": len(sister_cols),
        "cv_mean_nmae": round(cv_mean, 4),
        "cv_std_nmae": round(cv_std, 4),
        "cv_q1_only_mean_nmae": round(cv_q1, 4),
        "fold_nmae": [round(x, 3) for x in fold_nmae],
        "fold_nmae_q1_only": [round(x, 3) for x in fold_q1],
        "segments": seg,
        "top40_importance": [{"feat": f, "gain": round(float(g), 1)} for f, g in imp],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\nготово. артефакты в {MODEL_DIR}, время {time.time() - t0:.1f} сек")


if __name__ == "__main__":
    main()
