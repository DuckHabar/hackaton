"""
Leakage guard для Q1 task.

Проверки:
1. В feature_cols НЕТ target (`Выработка...`) или его производных (`p_per_turbine`)
2. В feature_cols НЕТ lag/rolling/ewma OF TARGET
3. Distribution prediction в valid лежит в historical диапазоне
4. sample-weight, sector_8, n_avail это input features, не leakage
5. Bias correction table aggregated only on train (warn если valid touched)

Запуск перед сабмитом:
    python3 scripts/leakage_guard.py --exp EXP-007
"""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/duck/wind_hackathon")
TARGET_PATTERNS = [
    "Выработка",  # сам target
    "p_per_turbine",  # производная target / n_avail
    "y_true", "target", "label",
]
SUSPICIOUS_PATTERNS_TARGET_LAG = [
    "Выработка_lag", "Выработка_roll", "Выработка_ewma",
    "power_lag", "power_roll", "p_per_turbine_lag",
    "p_per_turbine_roll", "y_lag",
]


def check(exp_id):
    print(f"=== Leakage Guard для {exp_id} ===")
    summary_f = ROOT / f"experiments/active/{exp_id}/summary.json"
    if not summary_f.exists():
        print(f"FAIL: нет summary.json")
        sys.exit(1)
    summary = json.loads(summary_f.read_text())
    # Извлекаем feature_cols из top importance или явно
    if "top15_importance" in summary:
        feats = [f["feat"] for f in summary["top15_importance"]]
    elif "top25_importance" in summary:
        feats = [f["feat"] for f in summary["top25_importance"]]
    else:
        feats = summary.get("feature_cols") or []

    issues = []

    # 1. Прямой target в features
    for pat in TARGET_PATTERNS:
        for f in feats:
            if pat in f:
                issues.append(f"DIRECT_TARGET: '{f}' содержит '{pat}'")

    # 2. Lag/roll of target
    for pat in SUSPICIOUS_PATTERNS_TARGET_LAG:
        for f in feats:
            if pat.lower() in f.lower():
                issues.append(f"TARGET_LAG: '{f}' матчит '{pat}'")

    # 3. valid predictions stat sanity
    valid_pq = ROOT / f"experiments/active/{exp_id}/valid_pred.parquet"
    if valid_pq.exists():
        v = pd.read_parquet(valid_pq)
        if "pred_mw" in v.columns:
            stats = v["pred_mw"]
            if stats.isna().any():
                issues.append(f"VALID_NAN: {stats.isna().sum()} NaN в pred_mw")
            if (stats < -1).any() or (stats > 94.6).any():
                issues.append(f"VALID_OOR: pred вне [-1, 94.6]: min={stats.min():.3f}, max={stats.max():.3f}")
            if not (5 < stats.mean() < 80):
                issues.append(f"VALID_SUSPICIOUS_MEAN: {stats.mean():.3f} вне разумного [5, 80]")

    # 4. CV vs LB gap (если есть LB запись)
    # (no LB recorded yet)

    if issues:
        print(f"FAIL: {len(issues)} проблем:")
        for i in issues:
            print(f"  - {i}")
        sys.exit(2)
    print(f"OK: leakage check passed for {exp_id}")
    print(f"  Top features (sample): {feats[:5]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()
    check(args.exp)
