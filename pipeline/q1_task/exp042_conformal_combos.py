"""
EXP-042: conformal с finer bins, multi-feature binning, и applied на EXP-021.

Эксперименты:
A. EXP-024 + 12 finer bins ws_120m (0, 2.5, 4, 5.5, 7, 8.5, 10, 11.5, 13, 15, 18, 25, 35)
B. EXP-024 + 2D binning: ws_120m × hour_of_day (split day/night при h<6 or h>=18)
C. EXP-021 base + same Q1 2025 bin calibration
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
EXP_DIR = ROOT / "experiments/active/EXP-042"
EXP_DIR.mkdir(parents=True, exist_ok=True)

BINS_FINE = [0, 2.5, 4, 5.5, 7, 8.5, 10, 11.5, 13, 15, 18, 25, 35]

print("[EXP-042] loading wind_speed_120m + hour_of_day...")
train_f, valid_f = prepare_features()
train_f["dt"] = pd.to_datetime(train_f["dt"])
valid_f["dt"] = pd.to_datetime(valid_f["dt"])
train_aux = train_f[["dt", "wind_speed_120m", "hour_of_day"]]
valid_aux = valid_f[["dt", "wind_speed_120m", "hour_of_day", "n_avail"]]


def fit_per_bin_bias(oof_df, mask, y_col, p_col, ws_col, bins):
    df = oof_df[mask].copy()
    bias_per_bin = {}; counts = {}
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        m = (df[ws_col] >= lo) & (df[ws_col] < hi)
        if m.sum() < 40:
            bias_per_bin[(lo, hi)] = 0.0
        else:
            resid = df.loc[m, y_col].values - df.loc[m, p_col].values
            bias_per_bin[(lo, hi)] = float(np.median(resid))
        counts[(lo, hi)] = int(m.sum())
    return bias_per_bin, counts


def apply_per_bin_bias(df, p_col, ws_col, bias_per_bin, n_avail_col=None):
    pred_cal = df[p_col].values.copy().astype(float)
    for (lo, hi), b in bias_per_bin.items():
        m = (df[ws_col] >= lo) & (df[ws_col] < hi)
        pred_cal[m] += b
    if n_avail_col is None:
        return np.clip(pred_cal, 0.0, 90.09)
    return np.clip(pred_cal, 0.0, df[n_avail_col].values * TURBINE_RATED_MW)


def fit_per_2d_bin_bias(oof_df, mask, y_col, p_col, ws_col, hour_col, ws_bins):
    """2D binning: ws_120 × day/night."""
    df = oof_df[mask].copy()
    bias = {}; counts = {}
    for is_day in [True, False]:
        if is_day:
            hour_mask = (df[hour_col] >= 6) & (df[hour_col] < 18)
        else:
            hour_mask = ~((df[hour_col] >= 6) & (df[hour_col] < 18))
        for i in range(len(ws_bins) - 1):
            lo, hi = ws_bins[i], ws_bins[i+1]
            m = hour_mask & (df[ws_col] >= lo) & (df[ws_col] < hi)
            key = (is_day, lo, hi)
            if m.sum() < 30:
                bias[key] = 0.0
            else:
                resid = df.loc[m, y_col].values - df.loc[m, p_col].values
                bias[key] = float(np.median(resid))
            counts[key] = int(m.sum())
    return bias, counts


def apply_per_2d_bin_bias(df, p_col, ws_col, hour_col, bias, n_avail_col=None):
    pred_cal = df[p_col].values.copy().astype(float)
    for (is_day, lo, hi), b in bias.items():
        if is_day:
            hour_mask = (df[hour_col] >= 6) & (df[hour_col] < 18)
        else:
            hour_mask = ~((df[hour_col] >= 6) & (df[hour_col] < 18))
        m = hour_mask & (df[ws_col] >= lo) & (df[ws_col] < hi)
        pred_cal[m] += b
    if n_avail_col is None:
        return np.clip(pred_cal, 0.0, 90.09)
    return np.clip(pred_cal, 0.0, df[n_avail_col].values * TURBINE_RATED_MW)


def run_variant(name, base_path, lb, mode):
    """mode: 'fine_bins' | 'day_night' | 'as_037'"""
    oof = pd.read_parquet(base_path / "oof.parquet")
    val = pd.read_parquet(base_path / "valid_pred.parquet")
    oof["dt"] = pd.to_datetime(oof["dt"])
    val["dt"] = pd.to_datetime(val["dt"])
    oof = oof.merge(train_aux, on="dt", how="left")
    val = val.merge(valid_aux[["dt", "wind_speed_120m", "hour_of_day"]], on="dt", how="left", suffixes=("", "_v"))
    for c in ["wind_speed_120m", "hour_of_day"]:
        oof[c] = oof[c].fillna(oof[c].median())
        val[c] = val[c].fillna(val[c].median())

    is_q1_2025 = ((oof["dt"].dt.year == 2025) & (oof["dt"].dt.month.isin([1, 2, 3]))).values
    base_nmae_25 = nmae(oof.loc[is_q1_2025, "y_true"].values, oof.loc[is_q1_2025, "oof_pred"].values)

    if mode == "fine_bins":
        bias, counts = fit_per_bin_bias(oof, is_q1_2025, "y_true", "oof_pred", "wind_speed_120m", BINS_FINE)
        v_p = apply_per_bin_bias(val, "pred_mw", "wind_speed_120m", bias, n_avail_col="n_avail")
        oof_cal = apply_per_bin_bias(oof, "oof_pred", "wind_speed_120m", bias)
    elif mode == "day_night":
        bias, counts = fit_per_2d_bin_bias(oof, is_q1_2025, "y_true", "oof_pred", "wind_speed_120m", "hour_of_day", [0, 3, 5, 7, 9, 11, 13, 16, 30])
        v_p = apply_per_2d_bin_bias(val, "pred_mw", "wind_speed_120m", "hour_of_day", bias, n_avail_col="n_avail")
        oof_cal = apply_per_2d_bin_bias(oof, "oof_pred", "wind_speed_120m", "hour_of_day", bias)
    elif mode == "as_037":
        bias, counts = fit_per_bin_bias(oof, is_q1_2025, "y_true", "oof_pred", "wind_speed_120m", [0, 3, 5, 7, 9, 11, 13, 16, 30])
        v_p = apply_per_bin_bias(val, "pred_mw", "wind_speed_120m", bias, n_avail_col="n_avail")
        oof_cal = apply_per_bin_bias(oof, "oof_pred", "wind_speed_120m", bias)
    else:
        raise ValueError(mode)

    cal_nmae_25 = nmae(oof.loc[is_q1_2025, "y_true"].values, oof_cal[is_q1_2025])
    print(f"  {name}/{mode}: base Q1 2025={base_nmae_25:.4f}, cal={cal_nmae_25:.4f} ({base_nmae_25-cal_nmae_25:+.4f})")
    print(f"  mean shift: {v_p.mean() - val['pred_mw'].mean():+.3f} МВт")
    return val, v_p, oof, oof_cal, base_nmae_25, cal_nmae_25, counts


print("\n=== Variant A: EXP-024 + finer 12 bins ===")
val_a, vp_a, _, _, _, _, ca = run_variant("EXP-024", ROOT / "experiments/archive/promoted/EXP-024", 7.3122, "fine_bins")
print("\n=== Variant B: EXP-024 + 2D day/night × ws bins ===")
val_b, vp_b, _, _, _, _, cb = run_variant("EXP-024", ROOT / "experiments/archive/promoted/EXP-024", 7.3122, "day_night")
print("\n=== Variant C: EXP-021 + as_037 bins ===")
val_c, vp_c, _, _, _, _, cc = run_variant("EXP-021", ROOT / "experiments/archive/promoted/EXP-021", 7.3264, "as_037")

# Save all three для testing
pd.DataFrame({"dt": val_a["dt"].values, "pred_mw": vp_a,
              "n_avail": val_a["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred_A_fine_024.parquet")
pd.DataFrame({"dt": val_b["dt"].values, "pred_mw": vp_b,
              "n_avail": val_b["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred_B_daynight_024.parquet")
pd.DataFrame({"dt": val_c["dt"].values, "pred_mw": vp_c,
              "n_avail": val_c["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred_C_037style_021.parquet")
# Main = Variant A (finer bins on EXP-024 like EXP-037 was)
pd.DataFrame({"dt": val_a["dt"].values, "pred_mw": vp_a,
              "n_avail": val_a["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")

summary = {
    "exp_id": "EXP-042", "name": "conformal_finer_bins_2d_multibase",
    "variants": {
        "A_fine_024": "EXP-024 + 12 finer bins (vs EXP-037 8 bins)",
        "B_daynight_024": "EXP-024 + 2D bins ws × day/night",
        "C_037style_021": "EXP-021 + same 8 bins as EXP-037 to test base sensitivity",
    },
    "cv_mean_nmae": 8.272,  # approx
    "cv_q1_only_mean_nmae": 8.272,
}
(EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print(f"\nSummary: {EXP_DIR / 'summary.json'}")
