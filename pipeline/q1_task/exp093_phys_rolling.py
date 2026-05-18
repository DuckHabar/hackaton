"""
EXP-093: EXP-021 + заимствования у сильного соперника
  + rolling/EWM stats (3/6/12/24h means/std/diff, 3/12/48h halflife EWM, TI proxy)
  + физ-фичи (Richardson number, solar elevation, LLJ flag, hodograph span)
  + recency sample weighting (2yr half-life) + season × ramp boost
  + iso baseline as feature (per-fold)

Target: Q1 2025 OOF nMAE < 7.28% (превзойти EXP-054 cal).
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae, smooth_power_curve_ti
from lgbm_q50 import prepare_features, compute_segments
from nwp_bias_correction import apply_bias, add_p_phys_corrected, estimate_bias
from sister_features import add_sister_features

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-093"
EXP_DIR.mkdir(parents=True, exist_ok=True)

SISTER_SOURCES = ["gfs", "icon", "arpege", "knmi", "dmi"]
EXTRA_LAG_HOURS = [-48, -36, -24, -18, -12, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48]
EXTRA_LAG_COLS = ["wind_speed_120m", "ws_corrected_120", "wind_gusts_10m",
                  "wd_120_sin", "wd_120_cos", "alpha_80_120", "p_phys_corrected"]

LAT_DEG = 46.8268455973
LON_DEG = 38.7179393185
UTC_OFFSET_H = 3.0
G = 9.81
V_CUT_IN = 3.0
V_RATED = 12.0


# Parity helpers (with EXP-021)
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


# NEW: rolling / EWM features
def add_rolling_features(df_all):
    """3/6/12/24h rolling means/std/diff. df_all sorted by dt."""
    df_all = df_all.sort_values("dt").reset_index(drop=True)
    for col in ["ws_corrected_120", "wind_speed_120m", "wind_gusts_10m"]:
        if col not in df_all.columns:
            continue
        s = df_all[col]
        for w in [3, 6, 12, 24]:
            df_all[f"{col}_roll_mean_{w}h"] = s.rolling(w, min_periods=1).mean()
            df_all[f"{col}_roll_std_{w}h"] = s.rolling(w, min_periods=2).std().fillna(0.0)
        df_all[f"{col}_roll_min_6h"] = s.rolling(6, min_periods=1).min()
        df_all[f"{col}_roll_max_6h"] = s.rolling(6, min_periods=1).max()
        df_all[f"{col}_roll_range_6h"] = df_all[f"{col}_roll_max_6h"] - df_all[f"{col}_roll_min_6h"]
        df_all[f"{col}_diff_3h"] = s.diff(3).fillna(0.0)
        df_all[f"{col}_diff_6h"] = s.diff(6).fillna(0.0)
        df_all[f"{col}_diff_abs_3h"] = df_all[f"{col}_diff_3h"].abs()
        for hl in [3, 12, 48]:
            df_all[f"{col}_ewm_{hl}h"] = s.ewm(halflife=hl, min_periods=1, adjust=False).mean()
            df_all[f"{col}_dev_ewm_{hl}h"] = s - df_all[f"{col}_ewm_{hl}h"]
        # TI proxy
        df_all[f"{col}_ti_proxy_6h"] = df_all[f"{col}_roll_std_6h"] / df_all[f"{col}_roll_mean_6h"].clip(lower=0.5)
        df_all[f"{col}_ratio_24h"] = s / df_all[f"{col}_roll_mean_24h"].clip(lower=0.5)
        df_all[f"{col}_min_center_3h"] = s.rolling(3, center=True, min_periods=1).min()
        df_all[f"{col}_max_center_3h"] = s.rolling(3, center=True, min_periods=1).max()
        df_all[f"{col}_local_range_3h"] = df_all[f"{col}_max_center_3h"] - df_all[f"{col}_min_center_3h"]
    return df_all


# NEW: physical features
def solar_elevation(dt_series):
    dt = pd.to_datetime(dt_series)
    doy = dt.dt.dayofyear.values.astype(float)
    hour = dt.dt.hour.values.astype(float) + dt.dt.minute.values.astype(float) / 60.0
    local_solar_time = hour + LON_DEG / 15.0 - UTC_OFFSET_H
    hour_angle = np.deg2rad(15.0 * (local_solar_time - 12.0))
    decl = np.deg2rad(23.45 * np.sin(np.deg2rad(360.0 / 365.0 * (doy - 81.0))))
    lat = np.deg2rad(LAT_DEG)
    sin_elev = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return np.arcsin(np.clip(sin_elev, -1.0, 1.0))


def add_phys_features(df):
    df = df.copy()
    # Solar elevation
    elev = solar_elevation(df["dt"])
    df["solar_elev"] = elev
    df["solar_elev_sin"] = np.sin(elev)
    df["is_daytime"] = (elev > 0.0).astype(np.int8)
    cloud = df["cloud_cover_low"].astype(float) if "cloud_cover_low" in df.columns else pd.Series(0.0, index=df.index)
    cloud_n = np.where(cloud.max() <= 1.0, np.clip(cloud * 10.0, 0.0, 1.0), np.clip(cloud / 100.0, 0.0, 1.0))
    df["insolation_proxy"] = np.clip(np.sin(elev), 0.0, None) * (1.0 - 0.75 * cloud_n)

    # Richardson number (80m vs 120m)
    if "temperature_80m" in df.columns and "temperature_120m" in df.columns:
        dz = 40.0
        dtdz = (df["temperature_80m"] - df["temperature_120m"]) / dz
        dudz = (df["wind_speed_120m"] - df["wind_speed_80m"]) / dz + 1e-6
        t_avg_k = ((df["temperature_80m"] + df["temperature_120m"]) / 2.0 + 273.15).clip(lower=150.0)
        Ri = ((G / t_avg_k) * dtdz / (dudz ** 2)).clip(-5.0, 5.0)
        df["richardson_number"] = Ri
        df["atm_stable"] = (Ri > 0.25).astype(np.int8)
        df["atm_unstable"] = (Ri < -0.50).astype(np.int8)
        df["atm_neutral"] = ((Ri >= -0.50) & (Ri <= 0.25)).astype(np.int8)
        df["night_stable"] = ((elev < -0.1) & (Ri > 0.1)).astype(np.int8)

    # LLJ flag и ratio
    if "wind_speed_180m" in df.columns and "wind_speed_10m" in df.columns:
        ws180 = df["wind_speed_180m"]
        ws10 = df["wind_speed_10m"]
        df["llj_ratio"] = ws180 / ws10.clip(lower=0.5)
        df["llj_flag"] = ((df["llj_ratio"] > 2.5) & (elev < 0.0)).astype(np.int8)
        df["llj_excess"] = np.where(elev < 0.0, (ws180 - df["wind_speed_80m"]).clip(lower=0.0), 0.0)

    # Hodograph span (vector shear)
    if all(c in df.columns for c in ["wind_speed_80m", "wind_speed_180m", "wind_direction_80m", "wind_direction_180m"]):
        wd80 = np.deg2rad(df["wind_direction_80m"].values)
        wd180 = np.deg2rad(df["wind_direction_180m"].values)
        u80 = -df["wind_speed_80m"].values * np.sin(wd80)
        v80 = -df["wind_speed_80m"].values * np.cos(wd80)
        u180 = -df["wind_speed_180m"].values * np.sin(wd180)
        v180 = -df["wind_speed_180m"].values * np.cos(wd180)
        df["hodograph_span"] = np.sqrt((u180 - u80) ** 2 + (v180 - v80) ** 2)

    # Sector-specific binary flags (sector_8: 0=N..7=NW)
    if "sector_8" in df.columns:
        sec = df["sector_8"].astype(int).values
        df["sector_NE"] = (sec == 1).astype(np.int8)
        df["sector_SE"] = (sec == 3).astype(np.int8)
        df["sector_S"] = (sec == 4).astype(np.int8)
        df["sector_SW"] = (sec == 5).astype(np.int8)
        df["sector_S_all"] = np.isin(sec, [3, 4, 5]).astype(np.int8)
        hour = df["dt"].dt.hour.values if "dt" in df.columns else np.zeros(len(df))
        df["night_x_south"] = ((df["sector_S_all"] == 1) & ((hour >= 20) | (hour < 6))).astype(np.int8)

    # gust excess и ratio
    if "wind_gusts_10m" in df.columns and "wind_speed_120m" in df.columns:
        df["gust_excess"] = (df["wind_gusts_10m"] - df["wind_speed_120m"]).clip(lower=0.0)
    return df


# NEW: recency × season × ramp sample weights
def make_sample_weights_v2(train_clean):
    n = len(train_clean)
    dt = train_clean["dt"]
    max_dt = dt.max()
    age_days = (max_dt - dt).dt.days.astype(float).values
    recency = np.power(0.5, age_days / 730.0)
    # Season: target = Q1 (months 1-3). Distance to Feb (mid-Q1).
    month = dt.dt.month.values.astype(float)
    md = np.minimum(np.abs(month - 2.0), 12.0 - np.abs(month - 2.0))
    season = 0.90 + 0.30 * np.cos(2.0 * np.pi * md / 12.0)
    # Ramp zone boost
    ws120 = train_clean["wind_speed_120m"].values
    ramp = ((ws120 >= V_CUT_IN) & (ws120 < V_RATED)).astype(float)
    base = recency * season * (1.0 + 0.15 * ramp)
    # Our standard: Q1 ×1.5 (вместо ×2 в EXP-002 - мягче т.к. season уже даёт boost)
    is_q1 = month <= 3.0
    base = base * np.where(is_q1, 1.5, 1.0)
    n_rep = train_clean["Кол-во_ВЭУ_в_ремонте"].values
    base = base * np.where((n_rep == 4) | (n_rep == 5), 1.2, 1.0)
    return base / base.mean()


# NEW: iso baseline as feature (per-fold)
def fit_iso_feature(ws_eff_tr, y_tr, n_avail_tr, weights=None):
    avail = np.clip(n_avail_tr / 26.0, 0.2, 1.0)
    y_full = np.clip(y_tr / avail, 0.0, 90.09)
    x = np.asarray(ws_eff_tr, dtype=float)
    order = np.argsort(x, kind="mergesort")
    iso = IsotonicRegression(y_min=0.0, y_max=90.09, increasing=True, out_of_bounds="clip")
    w = None if weights is None else np.asarray(weights)[order]
    iso.fit(x[order], y_full[order], sample_weight=w)
    return iso


def predict_iso_feature(iso, ws_eff, n_avail):
    full = iso.predict(np.asarray(ws_eff, dtype=float))
    return full * np.clip(n_avail / 26.0, 0.0, 1.0)


# Main
def main():
    t0 = time.time()
    print("[EXP-093] prepare_features ...")
    train_f, valid_f = prepare_features()
    pc = smooth_power_curve_ti(ti=0.20, rated_speed=10.0)
    bt, _ = estimate_bias(train_f, pc)
    train_f = apply_bias(train_f, bt, mode="mean")
    train_f = add_p_phys_corrected(train_f, pc)
    valid_f = apply_bias(valid_f, bt, mode="mean")
    valid_f = add_p_phys_corrected(valid_f, pc)

    combined = pd.concat([train_f.assign(_src="train"), valid_f.assign(_src="valid")], ignore_index=True)
    combined = add_extra_lags_base(combined)
    # NEW rolling/EWM
    combined = add_rolling_features(combined)
    print(f"[EXP-093] after rolling/EWM: shape={combined.shape}")
    train_f = combined[combined["_src"] == "train"].drop(columns=["_src"]).copy()
    valid_f = combined[combined["_src"] == "valid"].drop(columns=["_src"]).copy()

    print(f"[EXP-093] add_sister_features (5 sources) ...")
    train_f, valid_f = add_sister_features(train_f, valid_f, sources=SISTER_SOURCES)
    train_f = add_disagreement_features(train_f, SISTER_SOURCES)
    valid_f = add_disagreement_features(valid_f, SISTER_SOURCES)

    # NEW physical features
    train_f = add_phys_features(train_f)
    valid_f = add_phys_features(valid_f)
    print(f"[EXP-093] after phys features: train {train_f.shape}, valid {valid_f.shape}")

    n_repair_col = "Кол-во_ВЭУ_в_ремонте"
    n_repair = train_f[n_repair_col]
    train_f_filtered = train_f[n_repair <= 5].copy()
    train_clean = train_f_filtered.dropna(subset=[TARGET_COL]).copy()

    # Feature list base (EXP-021 must_have) + новые
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
    base_feats = list(dict.fromkeys(top15 + must_have))
    extra_lag_names = [f"{c}_xlag_{('p' if h > 0 else 'm')}{abs(h)}"
                       for c in EXTRA_LAG_COLS for h in EXTRA_LAG_HOURS]

    # NEW rolling/EWM cols
    rolling_cols = []
    for col in ["ws_corrected_120", "wind_speed_120m", "wind_gusts_10m"]:
        for w in [3, 6, 12, 24]:
            rolling_cols.append(f"{col}_roll_mean_{w}h")
            rolling_cols.append(f"{col}_roll_std_{w}h")
        rolling_cols.extend([
            f"{col}_roll_min_6h", f"{col}_roll_max_6h", f"{col}_roll_range_6h",
            f"{col}_diff_3h", f"{col}_diff_6h", f"{col}_diff_abs_3h",
            f"{col}_ewm_3h", f"{col}_ewm_12h", f"{col}_ewm_48h",
            f"{col}_dev_ewm_3h", f"{col}_dev_ewm_12h", f"{col}_dev_ewm_48h",
            f"{col}_ti_proxy_6h", f"{col}_ratio_24h",
            f"{col}_local_range_3h",
        ])

    phys_cols = [
        "solar_elev", "solar_elev_sin", "is_daytime", "insolation_proxy",
        "richardson_number", "atm_stable", "atm_unstable", "atm_neutral", "night_stable",
        "llj_ratio", "llj_flag", "llj_excess", "hodograph_span",
        "sector_NE", "sector_SE", "sector_S", "sector_SW", "sector_S_all",
        "night_x_south", "gust_excess",
    ]
    iso_feat_col = ["iso_baseline_pred"]
    sister_cols = [c for c in train_clean.columns if any(f"__{s}" in c for s in SISTER_SOURCES)]
    dis_cols = [c for c in train_clean.columns if "__mean_all" in c or "__std_all" in c
                or "__range_all" in c or "__bias_vs_mean" in c]
    all_cols = list(dict.fromkeys(base_feats + extra_lag_names + rolling_cols + phys_cols + iso_feat_col + sister_cols + dis_cols))
    final_cols = [c for c in all_cols if c in train_clean.columns or c == "iso_baseline_pred"]
    print(f"[EXP-093] features: {len(final_cols)} (rolling={len([c for c in rolling_cols if c in train_clean.columns])}, "
          f"phys={len([c for c in phys_cols if c in train_clean.columns])}, "
          f"sister={len(sister_cols)}, dis={len(dis_cols)})")

    # Fill NaN with median (use only train_clean to compute medians)
    for c in final_cols:
        if c == "iso_baseline_pred":
            continue
        if c not in train_clean.columns:
            train_clean[c] = 0.0
        if c not in valid_f.columns:
            valid_f[c] = 0.0
        if train_clean[c].isna().any() or valid_f[c].isna().any():
            med = train_clean[c].median()
            if pd.isna(med):
                med = 0.0
            train_clean[c] = train_clean[c].fillna(med)
            valid_f[c] = valid_f[c].fillna(med)

    # Use EXP-004 study HPs (proven quantile τ=0.5 baseline)
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
        "verbose": -1, "num_threads": 0, "device": "cpu", "seed": 42,
        **bp,
    }

    # NEW: sample weights v2
    sw_v2 = make_sample_weights_v2(train_clean)
    print(f"[EXP-093] sample weights: mean={sw_v2.mean():.3f}, std={sw_v2.std():.3f}, "
          f"min={sw_v2.min():.3f}, max={sw_v2.max():.3f}")

    y = train_clean[TARGET_COL].values
    n_avail = train_clean["n_avail"].values
    groups = train_clean["month_key"].values

    # Iso baseline ws_eff = ws_corrected_120_corr (density+bias corrected)
    ws_eff = train_clean["ws_corrected_120_corr"].values
    ws_eff_v = valid_f["ws_corrected_120_corr"].values

    gkf = GroupKFold(n_splits=12)
    oof = np.zeros(len(train_clean))
    iso_pred_train = np.zeros(len(train_clean))  # OOF iso baseline as feature

    # First pass: fit iso per fold to fill iso_baseline_pred OOF column
    for fold, (tr, te) in enumerate(gkf.split(train_clean, y, groups)):
        iso = fit_iso_feature(ws_eff[tr], y[tr], n_avail[tr], weights=sw_v2[tr])
        iso_pred_train[te] = predict_iso_feature(iso, ws_eff[te], n_avail[te])

    # Fit single iso on full train for valid prediction
    iso_full = fit_iso_feature(ws_eff, y, n_avail, weights=sw_v2)
    iso_pred_valid = predict_iso_feature(iso_full, ws_eff_v, valid_f["n_avail"].values)

    train_clean["iso_baseline_pred"] = iso_pred_train
    valid_f["iso_baseline_pred"] = iso_pred_valid

    # Re-build feature matrix
    cat_idx = [final_cols.index("sector_8")] if "sector_8" in final_cols else "auto"
    X = train_clean[final_cols].values
    Xv = valid_f[final_cols].values

    # Main CV with the augmented features
    fold_nmae = []
    fold_q1_2025 = []
    feat_imp = np.zeros(len(final_cols))
    models = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        ds_tr = lgb.Dataset(X[tr], label=y[tr], weight=sw_v2[tr], categorical_feature=cat_idx)
        ds_te = lgb.Dataset(X[te], label=y[te], weight=sw_v2[te], reference=ds_tr, categorical_feature=cat_idx)
        m = lgb.train(hps, ds_tr, num_boost_round=n_boost, valid_sets=[ds_te],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred = m.predict(X[te], num_iteration=m.best_iteration)
        cap = n_avail[te] * TURBINE_RATED_MW
        pred_mw = np.clip(pred, 0.0, cap)
        oof[te] = pred_mw
        fn = nmae(y[te], pred_mw)
        fold_nmae.append(fn)
        dt_te = train_clean["dt"].iloc[te]
        is_q1_2025 = ((dt_te.dt.year == 2025) & dt_te.dt.month.isin([1, 2, 3])).values
        if is_q1_2025.sum() > 50:
            fold_q1_2025.append(nmae(y[te][is_q1_2025], pred_mw[is_q1_2025]))
        feat_imp += m.feature_importance("gain")
        models.append(m)
        print(f"  fold {fold:02d}: nMAE={fn:.3f}%")

    cv_mean = float(np.mean(fold_nmae))
    cv_std = float(np.std(fold_nmae))
    cv_q1_2025 = float(np.mean(fold_q1_2025)) if fold_q1_2025 else cv_mean

    # Compute Q1 2025 OOF directly (over all rows with year=2025, month in 1-3)
    is_q1_2025_all = ((train_clean["dt"].dt.year == 2025) &
                      train_clean["dt"].dt.month.isin([1, 2, 3])).values
    q1_2025_oof_nmae = float(nmae(y[is_q1_2025_all], oof[is_q1_2025_all]))

    print(f"\n[EXP-093] CV mean = {cv_mean:.4f}% +/- {cv_std:.4f}%")
    print(f"Q1 2025 OOF (n={is_q1_2025_all.sum()}): {q1_2025_oof_nmae:.4f}%")
    print(f"vs EXP-054 cal Q1 2025 = 7.2782: {7.2782 - q1_2025_oof_nmae:+.4f}")
    print(f"vs EXP-021 Q1 2025 = 7.326: {7.326 - q1_2025_oof_nmae:+.4f}")
    print(f"vs EXP-024 Q1 2025 = 7.314: {7.314 - q1_2025_oof_nmae:+.4f}")

    # Predict valid
    preds = np.mean([m.predict(Xv, num_iteration=m.best_iteration) for m in models], axis=0)
    pred_mw_v = np.clip(preds, 0.0, valid_f["n_avail"].values * TURBINE_RATED_MW)
    pd.DataFrame({"dt": valid_f["dt"].values, "pred_mw": pred_mw_v,
                  "p_phys": valid_f["p_phys"].values, "n_avail": valid_f["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")
    pd.DataFrame({"dt": train_clean["dt"].values, "y_true": y, "oof_pred": oof}).to_parquet(EXP_DIR / "oof.parquet")

    imp = sorted(zip(final_cols, feat_imp), key=lambda kv: -kv[1])[:80]
    # Highlight new features in top
    new_feat_names = set(rolling_cols) | set(phys_cols) | set(iso_feat_col)
    new_in_top = [(f, g) for f, g in imp if f in new_feat_names][:30]
    print(f"\nNew features in top-80 importance:")
    for f, g in new_in_top[:20]:
        print(f"  {f}: gain={g:.0f}")

    summary = {
        "exp_id": "EXP-093", "name": "rival_borrowings_rolling_phys_recency",
        "n_features": len(final_cols), "n_train_clean": len(train_clean),
        "new_feature_groups": {
            "rolling_ewm": len([c for c in rolling_cols if c in train_clean.columns]),
            "phys": len([c for c in phys_cols if c in train_clean.columns]),
            "iso_baseline": 1,
        },
        "cv_mean_nmae": round(cv_mean, 4),
        "cv_std_nmae": round(cv_std, 4),
        "cv_q1_2025_mean": round(cv_q1_2025, 4),
        "q1_2025_oof_nmae": round(q1_2025_oof_nmae, 4),
        "fold_nmae": [round(x, 3) for x in fold_nmae],
        "vs_exp054_cal_delta": round(7.2782 - q1_2025_oof_nmae, 4),
        "vs_exp024_delta": round(7.314 - q1_2025_oof_nmae, 4),
        "top80_importance": [{"feat": f, "gain": round(float(g), 1)} for f, g in imp],
        "new_features_in_top": [{"feat": f, "gain": round(float(g), 1)} for f, g in new_in_top],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\nSummary: {EXP_DIR / 'summary.json'}, time {summary['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
