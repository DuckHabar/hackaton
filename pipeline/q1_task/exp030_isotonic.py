"""
EXP-030: Isotonic calibration EXP-024 valid predictions используя Q1 2025 OOF как holdout.

EXP-013 был неудачен (fit на full OOF -> overfit). EXP-030 правильнее:
fit isotonic ТОЛЬКО на Q1 2025 (Jan-Mar 2025), применить к valid Q1 2026.

Гипотеза: Q1 2025 - ближайший seasonal match к Q1 2026, calibration сместит bias.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-030"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Load EXP-024 OOF + valid
exp024_oof = pd.read_parquet("/home/duck/wind_hackathon/experiments/archive/promoted/EXP-024/oof.parquet")
exp024_val = pd.read_parquet("/home/duck/wind_hackathon/experiments/archive/promoted/EXP-024/valid_pred.parquet")
exp024_oof["dt"] = pd.to_datetime(exp024_oof["dt"])
exp024_val["dt"] = pd.to_datetime(exp024_val["dt"])

print(f"OOF rows: {len(exp024_oof)}, valid rows: {len(exp024_val)}")
print(f"OOF dt range: {exp024_oof['dt'].min()} ... {exp024_oof['dt'].max()}")

# Q1 2025 mask (use as holdout)
y = exp024_oof["y_true"].values
p = exp024_oof["oof_pred"].values

# Q1 2025: Jan-Mar 2025
is_q1_2025 = ((exp024_oof["dt"].dt.year == 2025) & (exp024_oof["dt"].dt.month.isin([1, 2, 3]))).values
# Q1 ALL years (для сравнения)
is_q1_all = exp024_oof["dt"].dt.month.isin([1, 2, 3]).values

print(f"Q1 2025 holdout: {is_q1_2025.sum()} rows")
print(f"Q1 all years: {is_q1_all.sum()} rows")
print(f"Base nMAE Q1 2025: {nmae(y[is_q1_2025], p[is_q1_2025]):.4f}")
print(f"Base nMAE Q1 all: {nmae(y[is_q1_all], p[is_q1_all]):.4f}")

# Variant A: fit isotonic ТОЛЬКО на Q1 2025
iso_q1_2025 = IsotonicRegression(out_of_bounds="clip", increasing=True)
iso_q1_2025.fit(p[is_q1_2025], y[is_q1_2025])
p_cal_a = iso_q1_2025.predict(p[is_q1_2025])
print(f"\n=== Calibration A (Q1 2025 only) ===")
print(f"  Calibrated Q1 2025 nMAE: {nmae(y[is_q1_2025], p_cal_a):.4f}")

# Variant B: fit on Q1 2024 + Q1 2025 (4-6 months) - more data
is_q1_24_25 = (((exp024_oof["dt"].dt.year == 2024) | (exp024_oof["dt"].dt.year == 2025)) &
               (exp024_oof["dt"].dt.month.isin([1, 2, 3]))).values
print(f"\nQ1 2024+2025 holdout: {is_q1_24_25.sum()} rows")
iso_q1_24_25 = IsotonicRegression(out_of_bounds="clip", increasing=True)
iso_q1_24_25.fit(p[is_q1_24_25], y[is_q1_24_25])
p_cal_b = iso_q1_24_25.predict(p[is_q1_24_25])
print(f"\n=== Calibration B (Q1 2024+2025) ===")
print(f"  Calibrated nMAE: {nmae(y[is_q1_24_25], p_cal_b):.4f}")

# Variant C: fit на полный Q1 all years (use as reference, may overfit)
iso_q1_all = IsotonicRegression(out_of_bounds="clip", increasing=True)
iso_q1_all.fit(p[is_q1_all], y[is_q1_all])
p_cal_c = iso_q1_all.predict(p[is_q1_all])
print(f"\n=== Calibration C (all Q1) ===")
print(f"  Calibrated nMAE: {nmae(y[is_q1_all], p_cal_c):.4f}")

# Test on Q4 2024 (out-of-fit holdout для Calibration A)
is_q4_2024 = ((exp024_oof["dt"].dt.year == 2024) & (exp024_oof["dt"].dt.month.isin([10, 11, 12]))).values
print(f"\n=== Out-of-fit test on Q4 2024 ({is_q4_2024.sum()} rows) ===")
print(f"  Base nMAE Q4 2024: {nmae(y[is_q4_2024], p[is_q4_2024]):.4f}")
print(f"  After Cal A: {nmae(y[is_q4_2024], iso_q1_2025.predict(p[is_q4_2024])):.4f}")
print(f"  After Cal B: {nmae(y[is_q4_2024], iso_q1_24_25.predict(p[is_q4_2024])):.4f}")

# Choose best calibration (Q1 2025 only - closest seasonal match)
# Apply to valid Q1 2026
v_p = exp024_val["pred_mw"].values
v_n_avail = exp024_val["n_avail"].values

v_p_cal_a = iso_q1_2025.predict(v_p)
v_p_cal_b = iso_q1_24_25.predict(v_p)

# Save BOTH variants для sub experimentation
v_p_cal_a = np.clip(v_p_cal_a, 0.0, v_n_avail * TURBINE_RATED_MW)
v_p_cal_b = np.clip(v_p_cal_b, 0.0, v_n_avail * TURBINE_RATED_MW)

print(f"\nValid base: mean={v_p.mean():.2f}, min={v_p.min():.2f}, max={v_p.max():.2f}")
print(f"Valid cal A: mean={v_p_cal_a.mean():.2f}, min={v_p_cal_a.min():.2f}, max={v_p_cal_a.max():.2f}")
print(f"Valid cal B: mean={v_p_cal_b.mean():.2f}, min={v_p_cal_b.min():.2f}, max={v_p_cal_b.max():.2f}")

# Use Variant A (most seasonal-specific) as main
pd.DataFrame({"dt": exp024_val["dt"].values, "pred_mw": v_p_cal_a,
              "pred_mw_uncal": v_p, "n_avail": v_n_avail,
              "pred_mw_cal_b": v_p_cal_b}).to_parquet(EXP_DIR / "valid_pred.parquet")

# Save calibrated OOF for blending later
p_cal_full = iso_q1_2025.predict(p)
pd.DataFrame({"dt": exp024_oof["dt"].values, "y_true": y, "oof_pred": p_cal_full,
              "oof_pred_uncal": p}).to_parquet(EXP_DIR / "oof.parquet")

# Metrics для summary
cv_overall = nmae(y, p_cal_full)
cv_q1 = nmae(y[is_q1_all], p_cal_full[is_q1_all])
print(f"\nFinal OOF (with Cal A applied to all OOF): overall {cv_overall:.4f}, Q1 {cv_q1:.4f}")

summary = {
    "exp_id": "EXP-030", "name": "isotonic_calibration_q1_2025_holdout_on_exp024",
    "base": "EXP-024 (LB 7.3122)",
    "calibration": "IsotonicRegression fit on Q1 2025 OOF only (~2160 rows)",
    "q1_2025_holdout_rows": int(is_q1_2025.sum()),
    "base_nmae_q1_2025": round(float(nmae(y[is_q1_2025], p[is_q1_2025])), 4),
    "cal_nmae_q1_2025": round(float(nmae(y[is_q1_2025], p_cal_a)), 4),
    "cv_mean_nmae": round(float(cv_overall), 4),
    "cv_q1_only_mean_nmae": round(float(cv_q1), 4),
}
(EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print(f"Summary: {EXP_DIR / 'summary.json'}")
