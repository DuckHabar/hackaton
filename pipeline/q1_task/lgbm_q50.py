"""
EXP-002 = lgbm_q50_baseline_q1.

LightGBM quantile τ=0.5 (proxy MAE) на физ-фичах + bidirectional NWP lags ±6ч.
Учитывает выводы distribution-shifts и контр-гипотез:
- target normalize через y / n_avail (per-turbine) - устраняет repair shift
- sample_weight для Q1 строк ×2
- direct vs residual ablation
- year НЕ как фича (отложили на importance-check)
- ws_at_84 = ws_80 × (84/80)^α
- 8-секторная категория

CV: GroupKFold(12) по month_key + дополнительный Q1-only fold-mean.

Hard cap по available capacity (n_avail × 3.465 МВт) применяется к финальной prediction.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
from common.physics_features import (
    TARGET_COL, REPAIR_COL, DT_COL,
    TURBINE_RATED_MW, N_TURBINES, P_INST_MW,
    nmae,
)
TRAIN_PQ = ROOT / "data/processed/train_with_physics.parquet"
VALID_PQ = ROOT / "data/processed/valid_with_physics.parquet"
EXP_DIR = ROOT / "experiments/active/EXP-002"
PRED_OOF = EXP_DIR / "oof.parquet"
PRED_VALID = EXP_DIR / "valid_pred.parquet"
SUMMARY = EXP_DIR / "summary.json"
LOG = ROOT / "logs/exp002.log"

# HEFTcom24 winner HPs
HPS = dict(
    objective="quantile",
    alpha=0.5,
    metric="quantile",
    num_leaves=500,
    learning_rate=0.2,
    max_depth=6,
    min_data_in_leaf=200,
    lambda_l1=40.0,
    lambda_l2=80.0,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    seed=42,
    verbose=-1,
    num_threads=0,  # all cores
    device="cpu",
)
N_BOOST = 2000
EARLY_STOP = 50

# Lags для NWP (разрешено для Q1: NWP - это forecast, не target)
LAG_HOURS = [-6, -3, -1, 1, 3, 6]
LAG_BASE_COLS = ["wind_speed_120m", "wind_gusts_10m", "ws_corr_120",
                 "wd_120_sin", "wd_120_cos", "alpha_80_120"]
ROLL_WINDOWS = [3, 6, 12]
ROLL_COLS = ["wind_speed_120m", "ws_corr_120"]

SECTOR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


# Feature engineering
def build_sector(wd_deg):
    """8-секторная категория. Возвращает int 0..7."""
    # wd_deg ∈ [0, 360]. Сектор N = [-22.5, 22.5], NE=[22.5, 67.5], ...
    shifted = (wd_deg + 22.5) % 360
    return np.clip((shifted // 45).astype(np.int8), 0, 7)


def add_ws_at_84(df):
    """REWS на hub 84м через power-law от ws_80m с локальным α."""
    alpha = df["alpha_80_120"].clip(lower=-1.0, upper=1.5)
    df["ws_at_84"] = df["wind_speed_80m"].clip(lower=0.1) * (84.0 / 80.0) ** alpha
    df["ws_at_84_corr"] = df["ws_at_84"] * (df["rho_air"] / 1.225) ** (1.0 / 3.0)
    return df


def add_ws_180_fallback(df):
    """Fallback для ws_180m / wd_180m NaN - extrapolate от ws_120m по shear."""
    alpha = df["alpha_80_120"].clip(lower=-1.0, upper=1.5)
    fallback = df["wind_speed_120m"].clip(lower=0.1) * (180.0 / 120.0) ** alpha
    has_180 = df["wind_speed_180m"].notna()
    df["has_ws_180"] = has_180.astype(np.int8)
    df["wind_speed_180m"] = df["wind_speed_180m"].fillna(fallback)
    df["ws_corr_180"] = df["ws_corr_180"].fillna(
        fallback * (df["rho_air"] / 1.225) ** (1.0 / 3.0)
    )
    # wd_180 NaN - fallback wd_120
    for col in ["wind_direction_180m", "wd_180_deg", "wd_180_sin", "wd_180_cos"]:
        if col in df.columns:
            if col.endswith("_sin"):
                df[col] = df[col].fillna(df["wd_120_sin"])
            elif col.endswith("_cos"):
                df[col] = df[col].fillna(df["wd_120_cos"])
            else:
                df[col] = df[col].fillna(df["wind_direction_120m" if col != "wd_180_deg" else "wd_120_deg"])
    # alpha_10_180 fallback to alpha_80_120
    if "alpha_10_180" in df.columns:
        df["alpha_10_180"] = df["alpha_10_180"].replace([np.inf, -np.inf], np.nan).fillna(alpha)
    # rews_simple fallback (берём ws_80/ws_120/fallback180)
    weights = np.array([1.0 / abs(80 - 84), 1.0 / abs(120 - 84), 1.0 / abs(180 - 84)])
    weights = weights / weights.sum()
    df["rews_simple"] = df["rews_simple"].fillna(
        weights[0] * df["wind_speed_80m"] +
        weights[1] * df["wind_speed_120m"] +
        weights[2] * df["wind_speed_180m"]
    )
    df["rews_corr"] = df["rews_corr"].fillna(
        df["rews_simple"] * (df["rho_air"] / 1.225) ** (1.0 / 3.0)
    )
    return df


def add_lag_features(df_all):
    """
    df_all = concat([train, valid]) отсортированный по dt без пропусков.
    Применяем shift(±h) - это безопасно т.к. NWP заранее известен на весь горизонт.
    """
    df_all = df_all.sort_values("dt").reset_index(drop=True)
    for col in LAG_BASE_COLS:
        if col not in df_all.columns:
            continue
        for h in LAG_HOURS:
            sign = "p" if h > 0 else "m"
            df_all[f"{col}_lag_{sign}{abs(h)}"] = df_all[col].shift(-h)
    # Rolling mean/std/max на ws_120 (только trailing - это безопасно)
    for col in ROLL_COLS:
        if col not in df_all.columns:
            continue
        for w in ROLL_WINDOWS:
            df_all[f"{col}_rmean_{w}"] = df_all[col].rolling(w, min_periods=1).mean()
            df_all[f"{col}_rstd_{w}"] = df_all[col].rolling(w, min_periods=1).std().fillna(0)
            df_all[f"{col}_rmax_{w}"] = df_all[col].rolling(w, min_periods=1).max()
        # ewma(ws³) - «cleaner» power signal
        df_all[f"{col}_ewma_ws3_6"] = (df_all[col] ** 3).ewm(span=6, min_periods=1).mean()
    return df_all


def prepare_features():
    print("Loading train/valid parquets...")
    train = pd.read_parquet(TRAIN_PQ)
    valid = pd.read_parquet(VALID_PQ)
    train["dt"] = pd.to_datetime(train[DT_COL])
    valid["dt"] = pd.to_datetime(valid[DT_COL])
    train["_src"] = "train"
    valid["_src"] = "valid"
    valid[TARGET_COL] = np.nan
    valid["p_per_turbine"] = np.nan
    if "month_key" not in valid.columns:
        valid["month_key"] = valid["dt"].dt.to_period("M").astype(str)
    if "year" not in valid.columns:
        valid["year"] = valid["dt"].dt.year

    combined = pd.concat([train, valid], ignore_index=True)
    combined = add_ws_180_fallback(combined)
    combined = add_ws_at_84(combined)

    # sector_8 из wd_120_deg
    combined["sector_8"] = build_sector(combined["wd_120_deg"])

    # bidirectional NWP lags + rolling
    combined = add_lag_features(combined)
    print(f"After lag/rolling: shape={combined.shape}")

    # Назад split
    train_f = combined[combined["_src"] == "train"].copy()
    valid_f = combined[combined["_src"] == "valid"].copy()
    return train_f, valid_f


def get_feature_cols(df):
    """Финальный список feature columns. ИСКЛЮЧАЕМ: target, dt, datetime, _src,
    month_key (используется как group), year (не как фича на старте),
    p_per_turbine (используется как target_norm, не как фича)."""
    exclude = {
        TARGET_COL, "p_per_turbine", "_src", "dt", DT_COL,
        "month_key", "year",
        "ws_bin", "sector", "quarter",  # из physics baseline, не в финальной модели
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == "O":
            continue
        cols.append(c)
    return cols


# CV training
def fit_cv(train_f, feature_cols, mode="direct", n_splits=12):
    """
    mode: "direct" - target = y (raw МВт)
          "direct_perturb" - target = y / n_avail (per-turbine), pred ×n_avail
          "residual" - target = y - p_phys, pred = p_phys + resid_hat
          "residual_perturb" - target = (y - p_phys) / n_avail, pred = p_phys + resid_hat * n_avail
    """
    X = train_f[feature_cols].values
    n_avail = train_f["n_avail"].values
    p_phys = train_f["p_phys"].values
    y_raw = train_f[TARGET_COL].values

    if mode == "direct":
        y = y_raw
    elif mode == "direct_perturb":
        y = y_raw / np.maximum(n_avail, 1)
    elif mode == "residual":
        y = y_raw - p_phys
    elif mode == "residual_perturb":
        y = (y_raw - p_phys) / np.maximum(n_avail, 1)
    else:
        raise ValueError(mode)

    # Sample weight: Q1 строки ×2
    sw = np.where(train_f["dt"].dt.month.isin([1, 2, 3]).values, 2.0, 1.0)

    groups = train_f["month_key"].values
    gkf = GroupKFold(n_splits=n_splits)

    oof = np.zeros(len(train_f), dtype=np.float64)
    fold_nmae = []
    fold_nmae_q1 = []
    fold_best_iter = []
    feat_importance = np.zeros(len(feature_cols), dtype=np.float64)
    models = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        ds_tr = lgb.Dataset(X[tr], label=y[tr], weight=sw[tr])
        ds_te = lgb.Dataset(X[te], label=y[te], weight=sw[te], reference=ds_tr)
        model = lgb.train(
            HPS, ds_tr,
            num_boost_round=N_BOOST,
            valid_sets=[ds_te],
            callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(0)],
            categorical_feature=[feature_cols.index("sector_8")]
            if "sector_8" in feature_cols else "auto",
        )
        pred = model.predict(X[te], num_iteration=model.best_iteration)

        # Inverse transform
        if mode == "direct_perturb":
            pred_mw = pred * n_avail[te]
        elif mode == "residual":
            pred_mw = p_phys[te] + pred
        elif mode == "residual_perturb":
            pred_mw = p_phys[te] + pred * n_avail[te]
        else:
            pred_mw = pred

        # Capacity cap
        pred_mw = np.clip(pred_mw, 0.0, n_avail[te] * TURBINE_RATED_MW)

        oof[te] = pred_mw
        fnmae = nmae(y_raw[te], pred_mw)
        fold_nmae.append(fnmae)

        # Q1-only внутри этого тестового fold
        is_q1 = train_f["dt"].iloc[te].dt.month.isin([1, 2, 3]).values
        if is_q1.sum() > 50:
            fold_nmae_q1.append(nmae(y_raw[te][is_q1], pred_mw[is_q1]))

        fold_best_iter.append(model.best_iteration)
        feat_importance += model.feature_importance(importance_type="gain")
        models.append(model)
        print(f"  fold {fold:02d}: nMAE={fnmae:.3f}%  best_iter={model.best_iteration}")

    return {
        "oof": oof,
        "fold_nmae": fold_nmae,
        "fold_nmae_q1_only": fold_nmae_q1,
        "fold_best_iter": fold_best_iter,
        "feat_importance": feat_importance / n_splits,
        "models": models,
    }


def compute_segments(y_true, y_pred, df_meta):
    """Сегментный анализ как в EXP-001b."""
    seg = {}
    seg["quarter"] = {}
    for q in [1, 2, 3, 4]:
        mask = ((df_meta["dt"].dt.month - 1) // 3 + 1).values == q
        if mask.sum() > 50:
            seg["quarter"][f"Q{q}"] = nmae(y_true[mask], y_pred[mask])
    seg["hour_window"] = {}
    for lo, hi, name in [(0, 5, "night"), (6, 11, "morning"), (12, 17, "day"), (18, 23, "evening")]:
        mask = (df_meta["hour_of_day"] >= lo).values & (df_meta["hour_of_day"] <= hi).values
        if mask.sum() > 50:
            seg["hour_window"][name] = nmae(y_true[mask], y_pred[mask])
    seg["wind_regime"] = {}
    for lo, hi, name in [(0, 5, "low"), (5, 12, "normal"), (12, 15, "near_rated"), (15, 100, "high_cutout")]:
        mask = (df_meta["wind_speed_120m"] >= lo).values & (df_meta["wind_speed_120m"] < hi).values
        if mask.sum() > 50:
            seg["wind_regime"][name] = nmae(y_true[mask], y_pred[mask])
    seg["repair"] = {}
    for r in sorted(df_meta[REPAIR_COL].unique()):
        mask = (df_meta[REPAIR_COL] == r).values
        if mask.sum() > 100:
            seg["repair"][f"rep_{int(r)}"] = nmae(y_true[mask], y_pred[mask])
    seg["sector"] = {}
    for i, s in enumerate(SECTOR_NAMES):
        mask = (df_meta["sector_8"] == i).values
        if mask.sum() > 100:
            seg["sector"][s] = nmae(y_true[mask], y_pred[mask])
    return seg


def predict_valid(valid_f, models, feature_cols, mode):
    """Усреднение предсказаний с k моделей CV (bagging-style)."""
    X = valid_f[feature_cols].values
    p_phys = valid_f["p_phys"].values
    n_avail = valid_f["n_avail"].values

    preds = np.zeros(len(valid_f), dtype=np.float64)
    for m in models:
        preds += m.predict(X, num_iteration=m.best_iteration)
    preds /= len(models)

    if mode == "direct_perturb":
        pred_mw = preds * n_avail
    elif mode == "residual":
        pred_mw = p_phys + preds
    elif mode == "residual_perturb":
        pred_mw = p_phys + preds * n_avail
    else:
        pred_mw = preds

    return np.clip(pred_mw, 0.0, n_avail * TURBINE_RATED_MW)


# Main
def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train_f, valid_f = prepare_features()
    feature_cols = get_feature_cols(train_f)
    n_feat = len(feature_cols)
    print(f"\nFeatures ({n_feat}): {feature_cols[:10]}... (показаны первые 10)")
    print(f"Train shape: {train_f.shape}, Valid shape: {valid_f.shape}")

    # Дроп строк с NaN в физ-фичах (если есть) - для чистоты
    n_before = len(train_f)
    train_clean = train_f.dropna(subset=[TARGET_COL])
    # NaN в features заполнить медианой train (чтобы не дропать строки целиком -
    # 21% с ws_180 fallback'утся, но если что-то ещё)
    for c in feature_cols:
        if train_clean[c].isna().any():
            med = train_clean[c].median()
            train_clean[c] = train_clean[c].fillna(med)
            valid_f[c] = valid_f[c].fillna(med)
    print(f"Train clean: {len(train_clean)} (dropped {n_before - len(train_clean)} NaN target rows)")

    summary = {
        "exp_id": "EXP-002",
        "name": "lgbm_q50_baseline_q1",
        "n_features": n_feat,
        "feature_cols": feature_cols,
        "hps": HPS,
        "n_boost": N_BOOST,
        "early_stop": EARLY_STOP,
        "modes": {},
    }

    # Ablation по 4 modes
    modes = ["direct", "direct_perturb", "residual", "residual_perturb"]
    best_mode = None
    best_cv = float("inf")
    best_oof = None
    best_models = None

    for mode in modes:
        print(f"\n=== Mode: {mode} ===")
        result = fit_cv(train_clean, feature_cols, mode=mode, n_splits=12)
        cv_mean = float(np.mean(result["fold_nmae"]))
        cv_std = float(np.std(result["fold_nmae"]))
        cv_q1 = float(np.mean(result["fold_nmae_q1_only"])) if result["fold_nmae_q1_only"] else None
        cv_q1_std = float(np.std(result["fold_nmae_q1_only"])) if result["fold_nmae_q1_only"] else None

        # Segments
        seg = compute_segments(
            train_clean[TARGET_COL].values, result["oof"], train_clean
        )

        # Top-5 importance
        imp = sorted(
            zip(feature_cols, result["feat_importance"]),
            key=lambda kv: -kv[1],
        )
        top_imp = [{"feat": f, "gain": round(float(g), 1)} for f, g in imp[:15]]

        summary["modes"][mode] = {
            "cv_mean_nmae": round(cv_mean, 4),
            "cv_std_nmae": round(cv_std, 4),
            "cv_q1_only_mean_nmae": round(cv_q1, 4) if cv_q1 is not None else None,
            "cv_q1_only_std_nmae": round(cv_q1_std, 4) if cv_q1_std is not None else None,
            "fold_nmae": [round(x, 3) for x in result["fold_nmae"]],
            "fold_nmae_q1_only": [round(x, 3) for x in result["fold_nmae_q1_only"]],
            "fold_best_iter": result["fold_best_iter"],
            "segments": seg,
            "top15_importance": top_imp,
        }

        print(f"  CV mean: {cv_mean:.3f}% ± {cv_std:.3f}%")
        if cv_q1 is not None:
            print(f"  CV Q1-only: {cv_q1:.3f}% ± {cv_q1_std:.3f}%")
        print(f"  Best segment: {min(((s, b, v) for s, items in seg.items() for b, v in items.items()), key=lambda t: t[2])}")
        print(f"  Worst segment: {max(((s, b, v) for s, items in seg.items() for b, v in items.items()), key=lambda t: t[2])}")
        print(f"  Top-5 imp: {top_imp[:5]}")

        if cv_mean < best_cv:
            best_cv = cv_mean
            best_mode = mode
            best_oof = result["oof"]
            best_models = result["models"]

    print(f"\n=== BEST MODE: {best_mode}  CV={best_cv:.3f}% ===")
    summary["best_mode"] = best_mode
    summary["best_cv_nmae"] = round(best_cv, 4)

    # Predict valid
    print(f"Predicting valid через bagging k={len(best_models)} моделей...")
    valid_pred = predict_valid(valid_f, best_models, feature_cols, best_mode)
    valid_out = pd.DataFrame({
        "dt": valid_f["dt"].values,
        "pred_mw": valid_pred,
        "p_phys": valid_f["p_phys"].values,
        "n_avail": valid_f["n_avail"].values,
    })
    valid_out.to_parquet(PRED_VALID, index=False)
    print(f"valid pred: mean={valid_pred.mean():.2f} МВт, q05={np.quantile(valid_pred, 0.05):.2f}, "
          f"q95={np.quantile(valid_pred, 0.95):.2f}")

    # OOF dump
    oof_out = pd.DataFrame({
        "dt": train_clean["dt"].values,
        "y_true": train_clean[TARGET_COL].values,
        "oof_pred": best_oof,
        "p_phys": train_clean["p_phys"].values,
    })
    oof_out.to_parquet(PRED_OOF, index=False)

    summary["elapsed_sec"] = round(time.time() - t0, 1)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\nSummary saved: {SUMMARY}")
    print(f"OOF: {PRED_OOF}, valid pred: {PRED_VALID}")
    print(f"Total time: {summary['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
