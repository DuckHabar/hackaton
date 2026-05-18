"""
EXP-037: Conditional conformal calibration per wind_speed bin.

Идея: isotonic global overfit (EXP-030 LB 7.55). Альтернатива - piecewise correction
per wind_speed_120m bin на Q1 2025 OOF, применённый к Q1 2026 predictions.

Base: EXP-024 (LB 7.3122).

Бины: [0,3) cut-in zone, [3,6) low-power, [6,9) cube-power, [9,12) high-power,
[12,18) rated/cut-off zone, [18,30) cut-off.

Per bin вычисляем смещение bias_i = median(y - pred) на Q1 2025 OOF.
Применяем к valid Q1 2026: pred_cal[i] = pred[i] + bias[bin_of_ws_120m[i]].

Опасность: 6 bins × 2160 Q1 2025 rows = 360 per bin avg, может быть достаточно
без оверфита. Если bin <100 - fallback к глобальному 0 (no correction).
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
EXP_DIR = ROOT / "experiments/active/EXP-037"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Base = EXP-024
exp024_oof = pd.read_parquet(ROOT / "experiments/archive/promoted/EXP-024/oof.parquet")
exp024_val = pd.read_parquet(ROOT / "experiments/archive/promoted/EXP-024/valid_pred.parquet")
exp024_oof["dt"] = pd.to_datetime(exp024_oof["dt"])
exp024_val["dt"] = pd.to_datetime(exp024_val["dt"])

# Нужна wind_speed_120m по dt -> используем prepare_features
print("[EXP-037] loading wind_speed_120m features...")
train_f, valid_f = prepare_features()
train_f["dt"] = pd.to_datetime(train_f["dt"])
valid_f["dt"] = pd.to_datetime(valid_f["dt"])

# Merge ws_120m в OOF и valid
train_ws = train_f[["dt", "wind_speed_120m"]]
valid_ws = valid_f[["dt", "wind_speed_120m", "n_avail"]]
oof = exp024_oof.merge(train_ws, on="dt", how="left")
val = exp024_val.merge(valid_ws[["dt", "wind_speed_120m"]], on="dt", how="left", suffixes=("", "_v"))
print(f"OOF after merge: {len(oof)}, NaN ws: {oof['wind_speed_120m'].isna().sum()}")
print(f"Valid after merge: {len(val)}, NaN ws: {val['wind_speed_120m'].isna().sum()}")
oof["wind_speed_120m"] = oof["wind_speed_120m"].fillna(oof["wind_speed_120m"].median())
val["wind_speed_120m"] = val["wind_speed_120m"].fillna(val["wind_speed_120m"].median())

# Q1 2025 holdout для calibration fit
is_q1_2025 = ((oof["dt"].dt.year == 2025) & (oof["dt"].dt.month.isin([1, 2, 3]))).values
is_q1_2024_25 = (((oof["dt"].dt.year == 2024) | (oof["dt"].dt.year == 2025)) & (oof["dt"].dt.month.isin([1, 2, 3]))).values
print(f"Q1 2025 rows: {is_q1_2025.sum()}, Q1 2024-25 rows: {is_q1_2024_25.sum()}")

BINS = [0, 3, 5, 7, 9, 11, 13, 16, 30]

def fit_per_bin_bias(oof_subset_mask, y_col, p_col, ws_col):
    df = oof[oof_subset_mask].copy()
    bias_per_bin = {}
    counts = {}
    for i in range(len(BINS) - 1):
        lo, hi = BINS[i], BINS[i+1]
        mask = (df[ws_col] >= lo) & (df[ws_col] < hi)
        if mask.sum() < 50:
            bias_per_bin[(lo, hi)] = 0.0  # fallback
        else:
            resid = df.loc[mask, y_col].values - df.loc[mask, p_col].values
            bias_per_bin[(lo, hi)] = float(np.median(resid))
        counts[(lo, hi)] = int(mask.sum())
    return bias_per_bin, counts

def apply_per_bin_bias(df, p_col, ws_col, bias_per_bin, n_avail_col=None):
    pred_cal = df[p_col].values.copy().astype(float)
    for (lo, hi), b in bias_per_bin.items():
        mask = (df[ws_col] >= lo) & (df[ws_col] < hi)
        pred_cal[mask] += b
    pred_cal = np.clip(pred_cal, 0.0, 90.09 if n_avail_col is None else df[n_avail_col].values * TURBINE_RATED_MW)
    return pred_cal


# Variant A: fit на Q1 2025
bias_a, counts_a = fit_per_bin_bias(is_q1_2025, "y_true", "oof_pred", "wind_speed_120m")
print(f"\n=== Variant A (Q1 2025 only) ===")
for (lo, hi), b in bias_a.items():
    print(f"  ws [{lo:.0f}, {hi:.0f}): bias={b:+.3f} МВт, n={counts_a[(lo, hi)]}")

oof_cal_a = apply_per_bin_bias(oof, "oof_pred", "wind_speed_120m", bias_a)
oof_a_q1_2025 = nmae(oof.loc[is_q1_2025, "y_true"].values, oof_cal_a[is_q1_2025])
base_q1_2025 = nmae(oof.loc[is_q1_2025, "y_true"].values, oof.loc[is_q1_2025, "oof_pred"].values)
print(f"\nQ1 2025 nMAE: base={base_q1_2025:.4f}, calibrated A={oof_a_q1_2025:.4f}")

# Variant B: fit на Q1 2024+2025
bias_b, counts_b = fit_per_bin_bias(is_q1_2024_25, "y_true", "oof_pred", "wind_speed_120m")
print(f"\n=== Variant B (Q1 2024+2025) ===")
for (lo, hi), b in bias_b.items():
    print(f"  ws [{lo:.0f}, {hi:.0f}): bias={b:+.3f} МВт, n={counts_b[(lo, hi)]}")

# Out-of-fit test на Q4 2024 (для A)
is_q4_2024 = ((oof["dt"].dt.year == 2024) & (oof["dt"].dt.month.isin([10, 11, 12]))).values
oof_a_q4_2024_pred = apply_per_bin_bias(oof, "oof_pred", "wind_speed_120m", bias_a)
oof_q4_2024_nmae_base = nmae(oof.loc[is_q4_2024, "y_true"].values, oof.loc[is_q4_2024, "oof_pred"].values)
oof_q4_2024_nmae_cal = nmae(oof.loc[is_q4_2024, "y_true"].values, oof_a_q4_2024_pred[is_q4_2024])
print(f"\n=== Out-of-fit test Q4 2024 ({is_q4_2024.sum()} rows) ===")
print(f"  base: {oof_q4_2024_nmae_base:.4f}, after A cal: {oof_q4_2024_nmae_cal:.4f} ({oof_q4_2024_nmae_base-oof_q4_2024_nmae_cal:+.4f})")

# Apply to valid: используем Variant A (более seasonal-specific)
v_p_cal = apply_per_bin_bias(val, "pred_mw", "wind_speed_120m", bias_a, n_avail_col="n_avail")
print(f"\nValid base: mean={val['pred_mw'].mean():.2f}, min={val['pred_mw'].min():.2f}, max={val['pred_mw'].max():.2f}")
print(f"Valid cal A: mean={v_p_cal.mean():.2f}, min={v_p_cal.min():.2f}, max={v_p_cal.max():.2f}")
print(f"Mean shift: {v_p_cal.mean() - val['pred_mw'].mean():+.3f} МВт")

pd.DataFrame({"dt": val["dt"].values, "pred_mw": v_p_cal,
              "pred_mw_uncal": val["pred_mw"].values,
              "n_avail": val["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")

# Сохраним и OOF с применённой Variant A
oof_cal_full = apply_per_bin_bias(oof, "oof_pred", "wind_speed_120m", bias_a)
overall = nmae(oof["y_true"].values, oof_cal_full)
q1_all_mask = oof["dt"].dt.month.isin([1, 2, 3]).values
q1_all_nmae = nmae(oof.loc[q1_all_mask, "y_true"].values, oof_cal_full[q1_all_mask])
print(f"\nFinal OOF (with Cal A applied to all OOF): overall {overall:.4f}, Q1 {q1_all_nmae:.4f}")
pd.DataFrame({"dt": oof["dt"].values, "y_true": oof["y_true"].values,
              "oof_pred": oof_cal_full,
              "oof_pred_uncal": oof["oof_pred"].values}).to_parquet(EXP_DIR / "oof.parquet")

summary = {
    "exp_id": "EXP-037", "name": "conditional_conformal_calibration_per_ws_bin",
    "base": "EXP-024 (LB 7.3122)",
    "bins": BINS,
    "bias_per_bin_q1_2025": {f"{lo}-{hi}": round(b, 4) for (lo, hi), b in bias_a.items()},
    "counts_per_bin_q1_2025": {f"{lo}-{hi}": c for (lo, hi), c in counts_a.items()},
    "q1_2025_holdout_base_nmae": round(float(base_q1_2025), 4),
    "q1_2025_holdout_cal_nmae": round(float(oof_a_q1_2025), 4),
    "q4_2024_oof_base_nmae": round(float(oof_q4_2024_nmae_base), 4),
    "q4_2024_oof_cal_nmae": round(float(oof_q4_2024_nmae_cal), 4),
    "cv_mean_nmae": round(float(overall), 4),
    "cv_q1_only_mean_nmae": round(float(q1_all_nmae), 4),
}
(EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print(f"Summary: {EXP_DIR / 'summary.json'}")
