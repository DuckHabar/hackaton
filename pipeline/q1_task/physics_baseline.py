"""
Этап 1: physics audit + physics baseline.

Запускает EDA по физике станции, строит physics baseline через
sg_3_4_132 power curve + density correction + capacity cap,
выдаёт CV nMAE без ML на time-based fold (GroupKFold по месяцу).

Все находки сохраняются в JSON ~/wind_hackathon/notes/physics_findings.json,
оттуда читаются для записи человекочитаемого notes/physics_findings.md.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Подключаем общий модуль (pipeline/common/physics_features.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.physics_features import (
    add_physics_features,
    smooth_power_curve_ti,
    predict_p_physical_total,
    nmae,
    TARGET_COL,
    REPAIR_COL,
    DT_COL,
    P_INST_MW,
    N_TURBINES,
    TURBINE_RATED_MW,
)

ROOT = Path("~/wind_hackathon").expanduser()
DATA = ROOT / "data/raw/train_dataset.csv"
OUT_JSON = ROOT / "notes/physics_findings.json"
OUT_PARQUET = ROOT / "data/processed/train_with_physics.parquet"
OUT_PARQUET_VALID = ROOT / "data/processed/valid_with_physics.parquet"
VALID = ROOT / "data/raw/valid_features.csv"


def load_train():
    df = pd.read_csv(DATA)
    df["dt"] = pd.to_datetime(df[DT_COL])
    df = df.sort_values("dt").reset_index(drop=True)
    df["year"] = df["dt"].dt.year
    df["month_key"] = df["dt"].dt.to_period("M").astype(str)
    return df


def load_valid():
    df = pd.read_csv(VALID)
    df["dt"] = pd.to_datetime(df[DT_COL])
    df = df.sort_values("dt").reset_index(drop=True)
    return df


def summary_dict(s):
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "q25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "q75": float(s.quantile(0.75)),
    }


def main():
    out = {}

    print("[1/9] Loading train...")
    df = load_train()
    out["shape"] = list(df.shape)
    out["date_range"] = [str(df["dt"].min()), str(df["dt"].max())]
    out["target_stats"] = summary_dict(df[TARGET_COL])
    out["repair_dist"] = {
        int(k): int(v) for k, v in df[REPAIR_COL].value_counts().sort_index().items()
    }

    print("[2/9] Adding physics features...")
    df = add_physics_features(df)
    df["p_per_turbine"] = df[TARGET_COL] / df["n_avail"]
    df["p_per_turbine"] = df["p_per_turbine"].clip(lower=0.0, upper=TURBINE_RATED_MW * 1.05)

    print("[3/9] Power curve audit (bins по ws_120m)...")
    ws_bins = [0, 3, 5, 8, 10, 12, 15, 20, 25, 50]
    df["ws_bin"] = pd.cut(df["wind_speed_120m"], bins=ws_bins)
    pc_stats = (
        df.groupby("ws_bin", observed=True)["p_per_turbine"]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    out["power_curve_by_ws_bin"] = {
        str(k): {kk: float(vv) for kk, vv in v.items()}
        for k, v in pc_stats.to_dict(orient="index").items()
    }
    # Effective rated power (95-й перцентиль)
    out["p_per_turbine_q95_by_bin"] = (
        df.groupby("ws_bin", observed=True)["p_per_turbine"]
        .quantile(0.95)
        .round(3)
        .to_dict()
    )
    out["p_per_turbine_q95_by_bin"] = {
        str(k): float(v) for k, v in out["p_per_turbine_q95_by_bin"].items()
    }

    print("[4/9] Air density...")
    rho = df["rho_air"]
    out["rho_air"] = summary_dict(rho)
    mask_rated = (df["wind_speed_120m"] >= 8) & (df["wind_speed_120m"] <= 13)
    out["rho_p_corr_in_rated_window"] = float(
        rho[mask_rated].corr(df.loc[mask_rated, "p_per_turbine"])
    )
    df_q = df.assign(rho_q=pd.qcut(rho, 5, labels=False, duplicates="drop"))
    out["rho_quintile_mean_p_per_turbine"] = (
        df_q.groupby("rho_q")["p_per_turbine"].mean().round(3).to_dict()
    )
    out["rho_quintile_mean_p_per_turbine"] = {
        int(k): float(v) for k, v in out["rho_quintile_mean_p_per_turbine"].items()
    }
    # Эффект density correction: разница nMAE с/без density_correction
    out["ws_corr_120_minus_ws_120_stats"] = summary_dict(
        df["ws_corr_120"] - df["wind_speed_120m"]
    )

    print("[5/9] Wind shear α...")
    alpha = df["alpha_80_120"].replace([np.inf, -np.inf], np.nan).dropna()
    out["alpha_80_120"] = summary_dict(alpha)
    out["alpha_by_hour"] = (
        df.groupby(df["dt"].dt.hour)["alpha_80_120"].mean().round(3).to_dict()
    )
    out["alpha_by_hour"] = {int(k): float(v) for k, v in out["alpha_by_hour"].items()}
    out["alpha_by_month"] = (
        df.groupby("month")["alpha_80_120"].mean().round(3).to_dict()
    )
    out["alpha_by_month"] = {int(k): float(v) for k, v in out["alpha_by_month"].items()}
    # Категории shear
    out["shear_class_dist"] = (
        pd.cut(
            df["alpha_80_120"],
            bins=[-1, 0.1, 0.2, 0.3, 5],
            labels=["unstable", "near_neutral", "stable", "very_stable"],
        )
        .value_counts(normalize=True)
        .round(3)
        .to_dict()
    )
    out["shear_class_dist"] = {str(k): float(v) for k, v in out["shear_class_dist"].items()}

    print("[6/9] Yaw sectors...")
    sec_bins = np.arange(0, 361, 45)
    sec_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    df["sector"] = pd.cut(
        df["wd_120_deg"], bins=sec_bins, labels=sec_labels, include_lowest=True
    )
    sec_stats = df.groupby("sector", observed=True).agg(
        mean_p_per_turbine=("p_per_turbine", "mean"),
        count=("p_per_turbine", "count"),
        mean_ws_120=("wind_speed_120m", "mean"),
    ).round(3)
    out["yaw_sector_stats"] = {
        str(k): {kk: float(vv) for kk, vv in v.items()}
        for k, v in sec_stats.to_dict(orient="index").items()
    }

    print("[7/9] Icing, TI, breeze...")
    icing = df["icing_risk"].astype(bool)
    out["icing_hours_total"] = int(icing.sum())
    out["icing_hours_share"] = float(icing.mean())
    if icing.sum() > 100:
        out["icing_mean_p_per_turbine"] = float(df.loc[icing, "p_per_turbine"].mean())
        cold_no_ice = (df["temperature_80m"] >= -5) & (df["temperature_80m"] <= 1) & ~icing
        if cold_no_ice.sum() > 100:
            out["no_icing_cold_mean_p_per_turbine"] = float(
                df.loc[cold_no_ice, "p_per_turbine"].mean()
            )
    out["ti_proxy"] = summary_dict(df["ti_proxy"])
    out["breeze_by_hour"] = (
        df.groupby(df["dt"].dt.hour)["breeze_idx"].mean().round(3).to_dict()
    )
    out["breeze_by_hour"] = {int(k): float(v) for k, v in out["breeze_by_hour"].items()}

    print("[8/9] Building physics baseline...")
    pc_smooth = smooth_power_curve_ti(ti=0.15)
    # Sample точек для дампа
    out["smoothed_power_curve_per_turbine_kw"] = {
        float(round(ws, 2)): float(round(p / 1000.0, 1))
        for ws, p in zip(pc_smooth["wind_speed"].values[::2], pc_smooth["value"].values[::2])
    }

    df["p_phys"] = predict_p_physical_total(df, pc_smooth, ws_col="ws_corr_120")
    df["p_phys_no_density"] = predict_p_physical_total(
        df, pc_smooth, ws_col="wind_speed_120m"
    )
    # Alt: на rews_corr (rotor-equivalent)
    df["p_phys_rews"] = predict_p_physical_total(df, pc_smooth, ws_col="rews_corr")

    out["nmae_phys_overall"] = nmae(df[TARGET_COL].values, df["p_phys"].values)
    out["nmae_phys_no_density_overall"] = nmae(
        df[TARGET_COL].values, df["p_phys_no_density"].values
    )
    out["nmae_phys_rews_overall"] = nmae(df[TARGET_COL].values, df["p_phys_rews"].values)

    print("[9/9] CV nMAE через GroupKFold по month_key...")
    # GroupKFold по уникальным месяцам (48 групп -> n_splits до 12)
    n_groups = df["month_key"].nunique()
    n_splits = min(12, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    cv_folds = []
    for tr, te in gkf.split(df, groups=df["month_key"]):
        cv_folds.append(nmae(df[TARGET_COL].iloc[te].values, df["p_phys"].iloc[te].values))
    out["nmae_phys_cv_mean"] = float(np.mean(cv_folds))
    out["nmae_phys_cv_std"] = float(np.std(cv_folds))
    out["nmae_phys_cv_folds"] = [round(float(x), 3) for x in cv_folds]
    out["nmae_phys_cv_n_splits"] = n_splits

    # Time-based expanding fold: fit на годе K, тест K+1
    out["nmae_phys_by_year_test"] = {}
    for yr in sorted(df["year"].unique()):
        mask = df["year"] == yr
        out["nmae_phys_by_year_test"][int(yr)] = nmae(
            df.loc[mask, TARGET_COL].values, df.loc[mask, "p_phys"].values
        )

    # Сегментный анализ по error_segments_list.yaml
    print("        + сегменты...")
    seg = {}

    # quarter
    df["quarter"] = (df["dt"].dt.month - 1) // 3 + 1
    seg["quarter"] = {
        f"Q{q}": nmae(
            df.loc[df["quarter"] == q, TARGET_COL].values,
            df.loc[df["quarter"] == q, "p_phys"].values,
        )
        for q in sorted(df["quarter"].unique())
    }

    # hour_window
    bins_h = [(0, 5, "night"), (6, 11, "morning"), (12, 17, "day"), (18, 23, "evening")]
    seg["hour_window"] = {}
    for lo, hi, name in bins_h:
        mask = (df["hour_of_day"] >= lo) & (df["hour_of_day"] <= hi)
        seg["hour_window"][name] = nmae(
            df.loc[mask, TARGET_COL].values, df.loc[mask, "p_phys"].values
        )

    # wind_regime
    bins_w = [
        (0, 5, "low"),
        (5, 12, "normal"),
        (12, 15, "near_rated"),
        (15, 100, "high_cutout"),
    ]
    seg["wind_regime"] = {}
    for lo, hi, name in bins_w:
        mask = (df["wind_speed_120m"] >= lo) & (df["wind_speed_120m"] < hi)
        if mask.sum() > 0:
            seg["wind_regime"][name] = nmae(
                df.loc[mask, TARGET_COL].values, df.loc[mask, "p_phys"].values
            )

    # repair_bucket
    seg["repair"] = {}
    for r in sorted(df[REPAIR_COL].unique()):
        mask = df[REPAIR_COL] == r
        if mask.sum() > 100:
            seg["repair"][f"rep_{int(r)}"] = nmae(
                df.loc[mask, TARGET_COL].values, df.loc[mask, "p_phys"].values
            )

    # temperature_bucket
    bins_t = [(-50, -1, "icing_zone"), (-1, 5, "cold"), (5, 20, "mild"), (20, 100, "warm")]
    seg["temperature"] = {}
    for lo, hi, name in bins_t:
        mask = (df["temperature_80m"] >= lo) & (df["temperature_80m"] < hi)
        if mask.sum() > 100:
            seg["temperature"][name] = nmae(
                df.loc[mask, TARGET_COL].values, df.loc[mask, "p_phys"].values
            )

    # shear_bucket
    seg["shear"] = {}
    bins_s = [(-5, 0.1, "low"), (0.1, 0.3, "med"), (0.3, 5, "high")]
    for lo, hi, name in bins_s:
        mask = (df["alpha_80_120"] >= lo) & (df["alpha_80_120"] < hi)
        if mask.sum() > 100:
            seg["shear"][name] = nmae(
                df.loc[mask, TARGET_COL].values, df.loc[mask, "p_phys"].values
            )

    # sector
    seg["sector"] = {}
    for s_lbl in sec_labels:
        mask = df["sector"] == s_lbl
        if mask.sum() > 100:
            seg["sector"][s_lbl] = nmae(
                df.loc[mask, TARGET_COL].values, df.loc[mask, "p_phys"].values
            )

    out["nmae_phys_by_segment"] = seg

    # Топ-3 слепых пятна
    flat = []
    for seg_name, items in seg.items():
        for bucket, v in items.items():
            flat.append((seg_name, bucket, v))
    flat.sort(key=lambda t: -t[2])
    out["top3_blind_spots"] = [
        {"segment": s, "bucket": b, "nmae": round(v, 3)} for s, b, v in flat[:5]
    ]

    print("[+] Сохраняю processed parquet для последующих ML экспериментов...")
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = [
        c for c in df.columns
        if c not in ("ws_bin", "sector", "quarter")
    ]
    df[keep_cols].to_parquet(OUT_PARQUET, index=False)
    print(f"    train_with_physics: {OUT_PARQUET}  rows={len(df)}")

    # Аналогично для valid (без target/p_phys)
    print("[+] Обработка valid_features.csv...")
    v = load_valid()
    v = add_physics_features(v)
    v["p_phys"] = predict_p_physical_total(v, pc_smooth, ws_col="ws_corr_120")
    v.to_parquet(OUT_PARQUET_VALID, index=False)
    print(f"    valid_with_physics: {OUT_PARQUET_VALID}  rows={len(v)}")

    # Распределение physics-preds на valid (sanity-check distribution shift)
    out["valid_p_phys_stats"] = summary_dict(v["p_phys"])
    out["valid_rho_air_stats"] = summary_dict(v["rho_air"])
    out["valid_alpha_stats"] = summary_dict(
        v["alpha_80_120"].replace([np.inf, -np.inf], np.nan).dropna()
    )

    # Сохраняем
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print()
    print(f"=== physics baseline DONE ===")
    print(f"Артефакт: {OUT_JSON}")
    print(f"Physics baseline overall nMAE   = {out['nmae_phys_overall']:.3f}%")
    print(f"Physics baseline (no density)   = {out['nmae_phys_no_density_overall']:.3f}%")
    print(f"Physics baseline (REWS corr)    = {out['nmae_phys_rews_overall']:.3f}%")
    print(f"Physics baseline CV mean nMAE   = {out['nmae_phys_cv_mean']:.3f}% (std {out['nmae_phys_cv_std']:.3f})")
    print(f"Top blind spots:")
    for bs in out["top3_blind_spots"][:3]:
        print(f"  {bs['segment']}/{bs['bucket']}: {bs['nmae']:.3f}%")


if __name__ == "__main__":
    main()
