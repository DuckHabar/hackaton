"""
Проверка гипотезы: 2025 capacity drop объясняется ветром+ремонтом, а не оборудованием.

Fit: target ~ (ws_corr_120^3) × n_avail. Смотрим year-residual mean.
Если residual ≤ 1 МВт mean - гипотеза подтверждается, year-фича не нужна.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path("~/wind_hackathon/pipeline").expanduser()))
from common.physics_features import TARGET_COL
ROOT = Path("~/wind_hackathon").expanduser()
TRAIN = ROOT / "data/processed/train_with_physics.parquet"
OUT = ROOT / "notes/h05_year_residual.json"


def gpu_warmup():
    try:
        import torch
        if torch.cuda.is_available():
            a = torch.randn(2048, 2048, device="cuda")
            for _ in range(10):
                _ = a @ a
            torch.cuda.synchronize()
            return "ok"
    except Exception:
        pass
    return "skip"


def main():
    print("GPU:", gpu_warmup())
    df = pd.read_parquet(TRAIN)
    df["dt"] = pd.to_datetime(df["METEOFORECASTHOUR_OPENM_Datetime"])
    df["year"] = df["dt"].dt.year

    # Простая физ-модель: target ≈ k1 · ws_corr^3 · n_avail + k0
    X = np.column_stack([
        df["ws_corr_120"].fillna(df["wind_speed_120m"]).values ** 3,
        df["n_avail"].values,
        (df["ws_corr_120"].fillna(df["wind_speed_120m"]).values ** 3) * df["n_avail"].values,
    ])
    y = df[TARGET_COL].values

    # Fit на всём train (это диагностика, не CV)
    reg = LinearRegression().fit(X, y)
    pred = reg.predict(X)
    resid = y - pred

    out = {
        "model": "target ~ ws_corr_120^3 + n_avail + ws_corr_120^3*n_avail",
        "intercept": round(float(reg.intercept_), 4),
        "coefs": {
            "ws3": round(float(reg.coef_[0]), 6),
            "n_avail": round(float(reg.coef_[1]), 4),
            "ws3*n_avail": round(float(reg.coef_[2]), 6),
        },
        "residual_overall_mean": round(float(resid.mean()), 4),
        "residual_overall_std": round(float(resid.std()), 4),
        "residual_by_year": {},
        "target_mean_by_year": {},
    }

    for yr in sorted(df["year"].unique()):
        m = df["year"] == yr
        out["residual_by_year"][int(yr)] = {
            "mean_resid": round(float(resid[m].mean()), 4),
            "std_resid": round(float(resid[m].std()), 4),
            "n": int(m.sum()),
        }
        out["target_mean_by_year"][int(yr)] = round(float(y[m].mean()), 4)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print(f"Overall residual: mean={out['residual_overall_mean']}, std={out['residual_overall_std']}")
    print(f"\n--- Residual by year (МВт) ---")
    for yr, stats in out["residual_by_year"].items():
        print(f"  {yr}: mean_resid={stats['mean_resid']:+.3f}  std={stats['std_resid']:.3f}  n={stats['n']}")
    print()
    max_abs_year_resid = max(abs(s["mean_resid"]) for s in out["residual_by_year"].values())
    if max_abs_year_resid < 1.0:
        print(f"Гипотеза ПОДТВЕРЖДЕНА: max |year residual| = {max_abs_year_resid:.3f} МВт < 1.0")
        print("Year-as-feature НЕ нужен в EXP-002.")
    elif max_abs_year_resid < 2.5:
        print(f"Гипотеза НЕОДНОЗНАЧНА: max |year residual| = {max_abs_year_resid:.3f} МВт.")
        print("Year-фича опциональна, проверить через LightGBM importance.")
    else:
        print(f"Гипотеза ОТВЕРГНУТА: max |year residual| = {max_abs_year_resid:.3f} МВт > 2.5")
        print("Year-фича критична. Включать в EXP-002.")


if __name__ == "__main__":
    main()
