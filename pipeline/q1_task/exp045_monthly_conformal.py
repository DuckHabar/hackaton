"""
EXP-045: monthly conformal calibration. Per Jan/Feb/Mar 2025 -> Jan/Feb/Mar 2026.

Гипотеза: Jan 2025 ближе к Jan 2026 чем смешанный Q1.
Fit bin biases отдельно для каждого месяца, apply matching месяцу valid.

Опасность: меньше данных per month (~720 ч), bins могут быть пустыми.
Используем coarser 6 bins для надёжности.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae
from lgbm_q50 import prepare_features

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-045"
EXP_DIR.mkdir(parents=True, exist_ok=True)

print("[EXP-045] loading features...")
train_f, valid_f = prepare_features()
train_f["dt"] = pd.to_datetime(train_f["dt"])
valid_f["dt"] = pd.to_datetime(valid_f["dt"])
train_ws = train_f[["dt", "wind_speed_120m"]]
valid_ws = valid_f[["dt", "wind_speed_120m", "n_avail"]]

oof = pd.read_parquet(ROOT / "experiments/archive/promoted/EXP-024/oof.parquet")
val = pd.read_parquet(ROOT / "experiments/archive/promoted/EXP-024/valid_pred.parquet")
oof["dt"] = pd.to_datetime(oof["dt"])
val["dt"] = pd.to_datetime(val["dt"])
oof = oof.merge(train_ws, on="dt", how="left")
val = val.merge(valid_ws[["dt", "wind_speed_120m"]], on="dt", how="left")
oof["wind_speed_120m"] = oof["wind_speed_120m"].fillna(oof["wind_speed_120m"].median())
val["wind_speed_120m"] = val["wind_speed_120m"].fillna(val["wind_speed_120m"].median())

EDGES_6 = [0, 4, 7, 9, 11, 14, 30]

def fit_bins(oof_subset, edges, min_n=30):
    bias = {}; counts = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i+1]
        m = (oof_subset["wind_speed_120m"] >= lo) & (oof_subset["wind_speed_120m"] < hi)
        if m.sum() < min_n:
            bias[(lo, hi)] = 0.0
        else:
            resid = oof_subset.loc[m, "y_true"].values - oof_subset.loc[m, "oof_pred"].values
            bias[(lo, hi)] = float(np.median(resid))
        counts[(lo, hi)] = int(m.sum())
    return bias, counts

# Fit per-month на Q1 2025
bias_per_month = {}
counts_per_month = {}
for month in [1, 2, 3]:
    mask = (oof["dt"].dt.year == 2025) & (oof["dt"].dt.month == month)
    subset = oof[mask].copy()
    bias_per_month[month], counts_per_month[month] = fit_bins(subset, EDGES_6, min_n=20)
    print(f"\nMonth {month} 2025 ({len(subset)} rows):")
    for (lo, hi), b in bias_per_month[month].items():
        print(f"  ws [{lo},{hi}): bias={b:+.3f}, n={counts_per_month[month][(lo, hi)]}")

# Apply per-month to valid Q1 2026
v_p_cal = val["pred_mw"].values.copy().astype(float)
for month, bias in bias_per_month.items():
    mask_v = val["dt"].dt.month == month
    for (lo, hi), b in bias.items():
        bin_mask = mask_v & (val["wind_speed_120m"] >= lo) & (val["wind_speed_120m"] < hi)
        v_p_cal[bin_mask] += b
v_p_cal = np.clip(v_p_cal, 0.0, val["n_avail"].values * TURBINE_RATED_MW)
print(f"\nValid base: mean={val['pred_mw'].mean():.2f}")
print(f"Valid monthly cal: mean={v_p_cal.mean():.2f} (shift {v_p_cal.mean()-val['pred_mw'].mean():+.3f})")

# OOF metrics
oof_cal = oof["oof_pred"].values.copy().astype(float)
for month, bias in bias_per_month.items():
    mask_o = (oof["dt"].dt.year == 2025) & (oof["dt"].dt.month == month)
    for (lo, hi), b in bias.items():
        bin_mask = mask_o & (oof["wind_speed_120m"] >= lo) & (oof["wind_speed_120m"] < hi)
        oof_cal[bin_mask] += b

is_q1_2025 = ((oof["dt"].dt.year == 2025) & (oof["dt"].dt.month.isin([1, 2, 3]))).values
base_q1 = nmae(oof.loc[is_q1_2025, "y_true"].values, oof.loc[is_q1_2025, "oof_pred"].values)
cal_q1 = nmae(oof.loc[is_q1_2025, "y_true"].values, oof_cal[is_q1_2025])
print(f"\nQ1 2025 OOF: base={base_q1:.4f}, monthly cal={cal_q1:.4f} ({base_q1-cal_q1:+.4f})")

# Out-of-fit: Q4 2024 (use Jan 2025 cal as proxy)
is_q4_2024 = ((oof["dt"].dt.year == 2024) & (oof["dt"].dt.month.isin([10, 11, 12]))).values
oof_cal_q4 = oof["oof_pred"].values.copy().astype(float)
for (lo, hi), b in bias_per_month[1].items():  # use Jan bias
    bin_mask = is_q4_2024 & (oof["wind_speed_120m"] >= lo) & (oof["wind_speed_120m"] < hi)
    oof_cal_q4[bin_mask] += b
base_q4 = nmae(oof.loc[is_q4_2024, "y_true"].values, oof.loc[is_q4_2024, "oof_pred"].values)
cal_q4 = nmae(oof.loc[is_q4_2024, "y_true"].values, oof_cal_q4[is_q4_2024])
print(f"Q4 2024 OOF (Jan cal as proxy): base={base_q4:.4f}, cal={cal_q4:.4f} ({base_q4-cal_q4:+.4f})")

# Save valid
pd.DataFrame({"dt": val["dt"].values, "pred_mw": v_p_cal,
              "pred_mw_uncal": val["pred_mw"].values,
              "n_avail": val["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")

# Save oof
pd.DataFrame({"dt": oof["dt"].values, "y_true": oof["y_true"].values,
              "oof_pred": oof_cal}).to_parquet(EXP_DIR / "oof.parquet")

summary = {
    "exp_id": "EXP-045", "name": "monthly_conformal_jan_feb_mar_2025",
    "edges": EDGES_6,
    "bias_per_month": {str(m): {f"{lo}-{hi}": round(b, 4) for (lo, hi), b in v.items()} for m, v in bias_per_month.items()},
    "counts_per_month": {str(m): {f"{lo}-{hi}": c for (lo, hi), c in v.items()} for m, v in counts_per_month.items()},
    "base_q1_2025_nmae": round(float(base_q1), 4),
    "cal_q1_2025_nmae": round(float(cal_q1), 4),
    "q4_2024_holdout_base": round(float(base_q4), 4),
    "q4_2024_holdout_cal": round(float(cal_q4), 4),
    "cv_mean_nmae": 8.272,
    "cv_q1_only_mean_nmae": round(float(cal_q1), 4),
}
(EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print(f"\nSummary: {EXP_DIR / 'summary.json'}")
