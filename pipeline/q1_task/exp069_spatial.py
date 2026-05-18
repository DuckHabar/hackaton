"""
EXP-069: spatial NWP percentile features (HEFTcom24 GEB winner trick).

Идея: вместо single point NWP, использовать 9 nearby points -> percentile/stat features.
Win expected: 0.4-0.8% nMAE (GEB winner получил 6-11% improvement).

Setup: EXP-021 baseline + ~20 spatial features.

Spatial features per (var, hour):
1. ws_120: mean, max, min, p25, p75, std across 9 points
2. wind_gusts_10m: mean, max, min, std
3. wind_direction_120m: circular std (proxy for frontal passage)
4. divergence: ((u_E - u_W)/dx + (v_N - v_S)/dy) where u = ws*sin(wd), v = ws*cos(wd)
5. vorticity: (v_E - v_W)/dx - (u_N - u_S)/dy

Save: oof, valid_pred, summary.json. Готово для blend в EXP-054 ensemble.
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
from lgbm_q50 import prepare_features, compute_segments
from nwp_bias_correction import apply_bias, add_p_phys_corrected, estimate_bias
from sister_features import add_sister_features

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-069"
EXP_DIR.mkdir(parents=True, exist_ok=True)
SPATIAL_DIR = ROOT / "data/processed/spatial_nwp"

SISTER_SOURCES = ["gfs", "icon", "arpege", "knmi", "dmi"]
SPATIAL_POINTS = ["C", "N", "S", "E", "W", "NE", "NW", "SE", "SW"]

# 0.1 deg ≈ 11.1 km at lat 46.83. Use dx=dy=11000 m for divergence/vorticity.
DX = DY = 11000.0


def load_spatial(points):
    out = {}
    for p in points:
        df = pd.read_parquet(SPATIAL_DIR / f"{p}.parquet")
        df["dt"] = pd.to_datetime(df["dt"])
        out[p] = df
    return out


def add_spatial_features(df, spatial_dfs):
    """Adds spatial features merged on dt."""
    d = df.copy()
    d["dt"] = pd.to_datetime(d["dt"])

    # Stack ws_120 across all points
    ws_arrays, gust_arrays = [], []
    wd_sin_arrays, wd_cos_arrays = [], []
    point_to_uv = {}
    for p, sf in spatial_dfs.items():
        sf_sub = sf[["dt", "wind_speed_120m", "wind_gusts_10m", "wind_direction_120m"]].copy()
        sf_sub = sf_sub.rename(columns={
            "wind_speed_120m": f"_ws120_{p}",
            "wind_gusts_10m": f"_gust_{p}",
            "wind_direction_120m": f"_wd120_{p}",
        })
        d = d.merge(sf_sub, on="dt", how="left")
        ws_arrays.append(d[f"_ws120_{p}"].values)
        gust_arrays.append(d[f"_gust_{p}"].values)
        wd = d[f"_wd120_{p}"].fillna(180.0).values
        wd_sin_arrays.append(np.sin(np.deg2rad(wd)))
        wd_cos_arrays.append(np.cos(np.deg2rad(wd)))
        # u = ws * sin(wd_rad), v = ws * cos(wd_rad)
        u = d[f"_ws120_{p}"].values * np.sin(np.deg2rad(wd))
        v = d[f"_ws120_{p}"].values * np.cos(np.deg2rad(wd))
        point_to_uv[p] = (u, v)

    ws_stack = np.array(ws_arrays)
    gust_stack = np.array(gust_arrays)

    d["spatial_ws120_mean"] = np.nanmean(ws_stack, axis=0)
    d["spatial_ws120_max"] = np.nanmax(ws_stack, axis=0)
    d["spatial_ws120_min"] = np.nanmin(ws_stack, axis=0)
    d["spatial_ws120_std"] = np.nanstd(ws_stack, axis=0)
    d["spatial_ws120_range"] = d["spatial_ws120_max"] - d["spatial_ws120_min"]
    d["spatial_ws120_p25"] = np.nanpercentile(ws_stack, 25, axis=0)
    d["spatial_ws120_p75"] = np.nanpercentile(ws_stack, 75, axis=0)
    d["spatial_ws120_iqr"] = d["spatial_ws120_p75"] - d["spatial_ws120_p25"]

    d["spatial_gust_mean"] = np.nanmean(gust_stack, axis=0)
    d["spatial_gust_max"] = np.nanmax(gust_stack, axis=0)
    d["spatial_gust_std"] = np.nanstd(gust_stack, axis=0)

    # Circular std for wd: 1 - |mean(sin)| - |mean(cos)| (proxy)
    wd_sin_mean = np.mean(wd_sin_arrays, axis=0)
    wd_cos_mean = np.mean(wd_cos_arrays, axis=0)
    d["spatial_wd_coherence"] = np.sqrt(wd_sin_mean**2 + wd_cos_mean**2)  # 1=all agree, 0=random

    # Divergence: ∂u/∂x + ∂v/∂y
    # ∂u/∂x ≈ (u_E - u_W) / (2*DX)
    # ∂v/∂y ≈ (v_N - v_S) / (2*DY)
    uE, vE = point_to_uv["E"]; uW, vW = point_to_uv["W"]
    uN, vN = point_to_uv["N"]; uS, vS = point_to_uv["S"]
    du_dx = (uE - uW) / (2 * DX)
    dv_dy = (vN - vS) / (2 * DY)
    d["spatial_divergence"] = du_dx + dv_dy

    # Vorticity (vertical): ∂v/∂x - ∂u/∂y
    dv_dx = (vE - vW) / (2 * DX)
    du_dy = (uN - uS) / (2 * DY)
    d["spatial_vorticity"] = dv_dx - du_dy

    # Strain rate magnitude: sqrt((du/dx - dv/dy)^2 + (du/dy + dv/dx)^2)
    norm_strain = (du_dx - dv_dy)
    shear_strain = (du_dy + dv_dx)
    d["spatial_strain"] = np.sqrt(norm_strain**2 + shear_strain**2)

    # Cleanup tmp columns
    drop_cols = [c for c in d.columns if c.startswith("_ws120_") or c.startswith("_gust_") or c.startswith("_wd120_")]
    d = d.drop(columns=drop_cols)
    return d


# Same setup as exp021
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
            if c in df.columns:
                cols.append(c)
        if len(cols) < 2: continue
        df[f"{var}__mean_all"] = df[cols].mean(axis=1)
        df[f"{var}__std_all"] = df[cols].std(axis=1)
        df[f"{var}__range_all"] = df[cols].max(axis=1) - df[cols].min(axis=1)
        for c in cols:
            tag = c.replace(var, "").lstrip("_") or "ecmwf"
            df[f"{var}__bias_vs_mean__{tag}"] = df[c] - df[f"{var}__mean_all"]
    return df


def main():
    t0 = time.time()
    print("[EXP-069] prepare features ...")
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

    print(f"[EXP-069] loading spatial NWP 9 points ...")
    spatial = load_spatial(SPATIAL_POINTS)
    train_f = add_spatial_features(train_f, spatial)
    valid_f = add_spatial_features(valid_f, spatial)
    print(f"  +spatial: train {train_f.shape}, valid {valid_f.shape}")

    n_repair_col = "Кол-во_ВЭУ_в_ремонте"
    train_f_filt = train_f[train_f[n_repair_col] <= 5].copy()

    extra_lag_names = [f"{c}_xlag_{('p' if h > 0 else 'm')}{abs(h)}"
                       for c in EXTRA_LAG_COLS for h in EXTRA_LAG_HOURS]
    sister_cols = [c for c in train_f.columns if any(f"__{s}" in c for s in SISTER_SOURCES)]
    dis_cols = [c for c in train_f.columns if any(suf in c for suf in
                ["__mean_all", "__std_all", "__range_all", "__bias_vs_mean"])]
    spatial_cols = [c for c in train_f.columns if c.startswith("spatial_")]
    print(f"  spatial features added: {len(spatial_cols)}")
    print(f"  spatial cols: {spatial_cols}")
    all_cols = list(dict.fromkeys(TOP_FEATS + extra_lag_names + sister_cols + dis_cols + spatial_cols))
    final_cols = [c for c in all_cols if c in train_f_filt.columns]
    print(f"[EXP-069] features: {len(final_cols)} (spatial={len(spatial_cols)})")

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
    is_q1 = train_clean["dt"].dt.month.isin([1, 2, 3]).values
    # Decision: use NO sw[q1]*=1.5 (fresh-eyes #2 confirmed)
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
    print(f"\n[EXP-069] CV = {cv_mean:.4f}% +/- {cv_std:.4f}% Q1={cv_q1:.4f}%")
    print(f"vs EXP-021 (8.238 / 8.242): {8.238-cv_mean:+.4f} / {8.242-cv_q1:+.4f}")
    print(f"vs EXP-062 A_no_sw (8.229 / 8.356): {8.229-cv_mean:+.4f} / {8.356-cv_q1:+.4f}")

    Xv = valid_f[final_cols].values
    preds = np.mean([m.predict(Xv, num_iteration=m.best_iteration) for m in models], axis=0)
    pred_mw_v = np.clip(preds, 0.0, valid_f["n_avail"].values * TURBINE_RATED_MW)
    pd.DataFrame({"dt": valid_f["dt"].values, "pred_mw": pred_mw_v,
                  "p_phys": valid_f["p_phys"].values, "n_avail": valid_f["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")
    pd.DataFrame({"dt": train_clean["dt"].values, "y_true": y, "oof_pred": oof}).to_parquet(EXP_DIR / "oof.parquet")

    # Feature importance for spatial cols
    imp = sorted(zip(final_cols, feat_imp), key=lambda kv: -kv[1])
    print("\n=== Top-10 features overall ===")
    for f, g in imp[:10]:
        print(f"  {f}: {g:.0f}")
    print("\n=== Spatial features importance ===")
    for f, g in imp:
        if f in spatial_cols:
            print(f"  {f}: {g:.0f}")

    summary = {
        "exp_id": "EXP-069", "name": "spatial_nwp_percentile_features",
        "n_features": len(final_cols),
        "spatial_features": spatial_cols,
        "cv_mean_nmae": cv_mean, "cv_std_nmae": cv_std,
        "cv_q1_only_mean_nmae": cv_q1,
        "elapsed_sec": time.time() - t0,
        "feature_importance_top20": [{"feat": f, "gain": float(g)} for f, g in imp[:20]],
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Done: {EXP_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
