"""
EXP-013: Conformalized Quantile Regression (CQR) post-calibration на EXP-009 OOF.

Идея: CQR (Romano, Patterson, Candes 2019) калибрует prediction intervals (или point
predictions через offset) под conditional shift. Для нашей задачи: считаем residuals
(y_true - oof_pred) на train holdout, фитим isotonic regression от prediction -> residual_q50,
применяем shift к valid preds. Цель - bias absorb который CV не видит.

CV-only experiment, sub only если LB lift.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-013"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Load EXP-009 OOF + valid pred
exp009_oof = pd.read_parquet(ROOT / "experiments/active/EXP-009/oof.parquet")
exp009_valid = pd.read_parquet(ROOT / "experiments/active/EXP-009/valid_pred.parquet")

print(f"OOF rows: {len(exp009_oof)}, valid rows: {len(exp009_valid)}")
print(f"OOF nMAE base: {nmae(exp009_oof['y_true'].values, exp009_oof['oof_pred'].values):.4f}%")

# Calibration: residual = y_true - oof_pred
# Isotonic regression: y_true = f(oof_pred), monotonic
y = exp009_oof["y_true"].values
p = exp009_oof["oof_pred"].values

# Train isotonic monotonic mapping p -> y
iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
iso.fit(p, y)
p_cal = iso.predict(p)

base_nmae = nmae(y, p)
cal_nmae = nmae(y, p_cal)
print(f"OOF nMAE: base={base_nmae:.4f}% -> calibrated={cal_nmae:.4f}% ({cal_nmae-base_nmae:+.4f})")

# Q1 only
exp009_oof["dt"] = pd.to_datetime(exp009_oof["dt"])
is_q1 = exp009_oof["dt"].dt.month.isin([1, 2, 3]).values
print(f"Q1 OOF nMAE: base={nmae(y[is_q1], p[is_q1]):.4f}% -> cal={nmae(y[is_q1], p_cal[is_q1]):.4f}%")

# Apply to valid
v_p = exp009_valid["pred_mw"].values
v_n_avail = exp009_valid["n_avail"].values
v_p_cal = iso.predict(v_p)
v_p_cal = np.clip(v_p_cal, 0.0, v_n_avail * TURBINE_RATED_MW)

print(f"Valid pred: base mean={v_p.mean():.3f}, cal mean={v_p_cal.mean():.3f}, "
      f"base max={v_p.max():.3f}, cal max={v_p_cal.max():.3f}")

# Save submission
out = pd.DataFrame({
    "dt": exp009_valid["dt"].values, "pred_mw": v_p_cal,
    "pred_mw_uncal": v_p, "n_avail": v_n_avail,
})
out.to_parquet(EXP_DIR / "valid_pred.parquet")

# Also save OOF
oof_out = pd.DataFrame({
    "dt": exp009_oof["dt"].values, "y_true": y,
    "oof_pred": p_cal, "oof_pred_uncal": p,
})
oof_out.to_parquet(EXP_DIR / "oof.parquet")

# CV regions by oof_pred quantile (where does cal help)
bins = pd.cut(p, bins=[0, 5, 15, 30, 50, 90], labels=["very_low", "low", "med", "high", "very_high"])
print("\nCal impact by pred bin:")
for b in bins.unique():
    if pd.isna(b): continue
    mask = (bins == b).values
    if mask.sum() < 50: continue
    nb = nmae(y[mask], p[mask]); nc = nmae(y[mask], p_cal[mask])
    print(f"  {b}: n={mask.sum()}  base={nb:.3f}  cal={nc:.3f}  Δ={nc-nb:+.3f}")

summary = {
    "exp_id": "EXP-013", "name": "cqr_isotonic_calibration_on_exp009",
    "calibration_method": "IsotonicRegression",
    "base_cv_oof": round(base_nmae, 4),
    "cal_cv_oof": round(cal_nmae, 4),
    "delta_cv": round(cal_nmae - base_nmae, 4),
    "base_q1": round(float(nmae(y[is_q1], p[is_q1])), 4),
    "cal_q1": round(float(nmae(y[is_q1], p_cal[is_q1])), 4),
    "cv_mean_nmae": round(cal_nmae, 4),  # для submit gate
    "cv_q1_only_mean_nmae": round(float(nmae(y[is_q1], p_cal[is_q1])), 4),
}
(EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print(f"\nSummary: {EXP_DIR / 'summary.json'}")
