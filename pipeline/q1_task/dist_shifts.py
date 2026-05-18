"""
Этап 2: distribution shifts train vs valid (Q1 2026).

Считаем KS-statistic и PSI для каждой фичи в двух разрезах:
  1. train (все 4 года) vs valid Q1 2026 - общий shift
  2. train (только Q1 за 2022-2025) vs valid Q1 2026 - сезонный shift (правильнее)

Анализируем и сырые Open-Meteo фичи, и физ-фичи из этапа 1.

Выход:
  - notes/dist_shifts.json - таблицы по всем фичам
  - notes/dist_shifts.md заполняется отдельно по этим данным.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path("~/wind_hackathon/pipeline").expanduser()))
from common.physics_features import TARGET_COL, REPAIR_COL, DT_COL
ROOT = Path("~/wind_hackathon").expanduser()
TRAIN_PQ = ROOT / "data/processed/train_with_physics.parquet"
VALID_PQ = ROOT / "data/processed/valid_with_physics.parquet"
OUT_JSON = ROOT / "notes/dist_shifts.json"

# Фичи для проверки (numerical only)
RAW_FEATURES = [
    "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
    "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m",
    "wind_gusts_10m",
    "temperature_80m", "temperature_120m",
    "pressure_msl",
    "rain", "showers", "snowfall", "cloud_cover_low",
    "month", "hour_of_day",
    REPAIR_COL,
]
PHYSICS_FEATURES = [
    "rho_air",
    "ws_corr_10", "ws_corr_80", "ws_corr_120", "ws_corr_180",
    "alpha_80_120", "alpha_10_180",
    "rews_simple", "rews_corr",
    "ri_bulk_80_120",
    "breeze_idx",
    "ti_proxy",
    "icing_risk",
    "available_frac", "n_avail",
    "wd_10_deg", "wd_80_deg", "wd_120_deg", "wd_180_deg",
    "wd_10_sin", "wd_10_cos",
    "wd_80_sin", "wd_80_cos",
    "wd_120_sin", "wd_120_cos",
    "wd_180_sin", "wd_180_cos",
]


def psi(expected, actual, n_bins=10):
    """
    Population Stability Index. Бинируем по квантилям expected, считаем shift.
    Возвращает скаляр PSI (>0.25 - significant shift).
    """
    expected = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna()
    actual = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna()
    if len(expected) < 50 or len(actual) < 50:
        return float("nan")
    # Бины по квантилям expected
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = expected.quantile(quantiles).values
    # Уникальные edges (если константная переменная - drop)
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_pct = (e_counts + 1) / (e_counts.sum() + len(e_counts))  # smoothed
    a_pct = (a_counts + 1) / (a_counts.sum() + len(a_counts))
    psi_val = float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
    return psi_val


def psi_label(p):
    if np.isnan(p):
        return "nan"
    if p < 0.10:
        return "stable"
    if p < 0.25:
        return "moderate"
    return "significant"


def compare_distributions(train_s, valid_s, name):
    """Возвращает dict с KS-D, p-value, PSI, basic stats для одной фичи."""
    a = pd.to_numeric(train_s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.to_numeric(valid_s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(a) < 50 or len(b) < 50:
        return None
    D, p = ks_2samp(a, b)
    p_val = psi(a, b)
    return {
        "feature": name,
        "ks_D": round(float(D), 4),
        "ks_p": round(float(p), 6),
        "psi": round(float(p_val), 4),
        "psi_label": psi_label(p_val),
        "n_train": int(len(a)),
        "n_valid": int(len(b)),
        "train_mean": round(float(a.mean()), 4),
        "valid_mean": round(float(b.mean()), 4),
        "train_std": round(float(a.std()), 4),
        "valid_std": round(float(b.std()), 4),
        "train_q05": round(float(a.quantile(0.05)), 4),
        "train_q95": round(float(a.quantile(0.95)), 4),
        "valid_q05": round(float(b.quantile(0.05)), 4),
        "valid_q95": round(float(b.quantile(0.95)), 4),
        "flag_ks": bool(D > 0.1),
        "flag_psi": bool(p_val > 0.25),
    }


def main():
    print("Loading data...")
    train = pd.read_parquet(TRAIN_PQ)
    valid = pd.read_parquet(VALID_PQ)
    train["dt"] = pd.to_datetime(train[DT_COL])
    valid["dt"] = pd.to_datetime(valid[DT_COL])

    # Train Q1 only (январь-март всех годов 2022-2025)
    train_q1 = train[train["dt"].dt.month.isin([1, 2, 3])].copy()
    print(f"train all: {len(train)}, train Q1 only: {len(train_q1)}, valid: {len(valid)}")

    all_features = RAW_FEATURES + [f for f in PHYSICS_FEATURES if f in train.columns and f in valid.columns]

    # 1) Общий shift train (all) vs valid Q1 2026
    print("\n[1/2] General shift: train (all 4 years) vs valid Q1 2026")
    general_rows = []
    for f in all_features:
        if f not in train.columns or f not in valid.columns:
            continue
        r = compare_distributions(train[f], valid[f], f)
        if r:
            general_rows.append(r)
    general_rows.sort(key=lambda r: -r["ks_D"])

    # 2) Сезонный shift train Q1 vs valid Q1
    print("[2/2] Seasonal shift: train Q1 only vs valid Q1 2026")
    seasonal_rows = []
    for f in all_features:
        if f not in train_q1.columns or f not in valid.columns:
            continue
        r = compare_distributions(train_q1[f], valid[f], f)
        if r:
            seasonal_rows.append(r)
    seasonal_rows.sort(key=lambda r: -r["ks_D"])

    # 3) Сравнение год за годом для valid-релевантных фичей (по Q1 каждого года)
    print("[+] Year-by-year Q1 means для ключевых фич:")
    yoy_features = ["wind_speed_120m", "temperature_80m", "pressure_msl",
                    "rho_air", "alpha_80_120", REPAIR_COL,
                    TARGET_COL if TARGET_COL in train.columns else "n_avail"]
    yoy = {}
    for f in yoy_features:
        if f not in train.columns and f not in valid.columns:
            continue
        if f in train.columns:
            t_q1_years = {}
            for yr in sorted(train["dt"].dt.year.unique()):
                m = (train["dt"].dt.year == yr) & train["dt"].dt.month.isin([1, 2, 3])
                if m.sum() > 100:
                    t_q1_years[int(yr)] = round(float(train.loc[m, f].mean()), 4)
            if f in valid.columns:
                t_q1_years[2026] = round(float(valid[f].mean()), 4)
            yoy[f] = t_q1_years

    # 4) Target sanity: распределение target по годам в Q1
    print("[+] Target Q1 means by year:")
    tgt_q1 = {}
    for yr in sorted(train["dt"].dt.year.unique()):
        m = (train["dt"].dt.year == yr) & train["dt"].dt.month.isin([1, 2, 3])
        if m.sum() > 100:
            tgt_q1[int(yr)] = {
                "mean": round(float(train.loc[m, TARGET_COL].mean()), 3),
                "median": round(float(train.loc[m, TARGET_COL].median()), 3),
                "std": round(float(train.loc[m, TARGET_COL].std()), 3),
                "n": int(m.sum()),
            }

    out = {
        "general_shift": general_rows,
        "seasonal_shift": seasonal_rows,
        "year_by_year_q1_means": yoy,
        "target_q1_by_year": tgt_q1,
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    # Печать топ-20 по каждому разрезу
    print("\n=== ОБЩИЙ SHIFT (train all vs valid Q1 2026), top-20 по KS-D ===")
    for r in general_rows[:20]:
        print(f"  {r['feature']:25s}  D={r['ks_D']:.3f}  PSI={r['psi']:.3f} [{r['psi_label']}]  "
              f"train μ={r['train_mean']:.2f} -> valid μ={r['valid_mean']:.2f}")

    print("\n=== СЕЗОННЫЙ SHIFT (train Q1 vs valid Q1 2026), top-20 по KS-D ===")
    for r in seasonal_rows[:20]:
        print(f"  {r['feature']:25s}  D={r['ks_D']:.3f}  PSI={r['psi']:.3f} [{r['psi_label']}]  "
              f"train μ={r['train_mean']:.2f} -> valid μ={r['valid_mean']:.2f}")

    # Сколько features флагнуто
    g_ks = sum(1 for r in general_rows if r["flag_ks"])
    g_psi = sum(1 for r in general_rows if r["flag_psi"])
    s_ks = sum(1 for r in seasonal_rows if r["flag_ks"])
    s_psi = sum(1 for r in seasonal_rows if r["flag_psi"])
    print(f"\nFlagged in general: KS>0.1 = {g_ks}, PSI>0.25 = {g_psi}  (из {len(general_rows)} фич)")
    print(f"Flagged in seasonal: KS>0.1 = {s_ks}, PSI>0.25 = {s_psi}  (из {len(seasonal_rows)} фич)")
    print(f"\nJSON saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
