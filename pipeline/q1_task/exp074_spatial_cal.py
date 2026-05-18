"""
EXP-074: EXP-069 spatial single base + 3yr conformal cal на Q1 2023+2024+2025.

Это replicate EXP-037/051 paradigm but на новой base EXP-069 (spatial features).

Если spatial single + cal даёт LB < 7.30 -> spatial direction works.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import nmae

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-074"
EXP_DIR.mkdir(parents=True, exist_ok=True)

BINS = [0, 3, 5, 7, 9, 11, 13, 16, 30]


def fit_bins(cal_df, ws_col="ws_120", bins=BINS, min_n=30):
    out = {}
    cal_df = cal_df.copy()
    cal_df["bin_idx"] = np.digitize(cal_df[ws_col].values, bins=bins[1:-1])
    for b in cal_df["bin_idx"].unique():
        sub = cal_df[cal_df["bin_idx"] == b]
        if len(sub) >= min_n:
            out[int(b)] = float((sub["y_true"] - sub["pred"]).median())
        else:
            out[int(b)] = 0.0
    return out


def apply_bins(df, bias_map, ws_col="ws_120", bins=BINS):
    bin_idx = np.digitize(df[ws_col].values, bins=bins[1:-1])
    bias = np.array([bias_map.get(int(b), 0.0) for b in bin_idx])
    return df["pred"].values + bias


def main():
    # Load OOF + valid + ws_120
    train = pd.read_parquet(ROOT / "data/processed/train_with_physics.parquet")
    train["dt"] = pd.to_datetime(train["METEOFORECASTHOUR_OPENM_Datetime"])
    valid = pd.read_parquet(ROOT / "data/processed/valid_with_physics.parquet")
    valid["dt"] = pd.to_datetime(valid["METEOFORECASTHOUR_OPENM_Datetime"])

    oof = pd.read_parquet(ROOT / "experiments/active/EXP-069/oof.parquet").rename(columns={"oof_pred": "pred"})
    val = pd.read_parquet(ROOT / "experiments/active/EXP-069/valid_pred.parquet").rename(columns={"pred_mw": "pred"})
    oof["dt"] = pd.to_datetime(oof["dt"])
    val["dt"] = pd.to_datetime(val["dt"])
    ws_train = train.set_index("dt")["wind_speed_120m"]
    ws_valid = valid.set_index("dt")["wind_speed_120m"]
    oof["ws_120"] = oof["dt"].map(ws_train).values
    val["ws_120"] = val["dt"].map(ws_valid).values

    # 3yr cal Q1 2023+2024+2025
    cal_mask = oof["dt"].dt.month.isin([1, 2, 3]) & oof["dt"].dt.year.isin([2023, 2024, 2025])
    cal_df = oof[cal_mask].copy()
    print(f"Cal rows: {len(cal_df)}")
    bias_map = fit_bins(cal_df)
    print(f"Bias map: {bias_map}")

    # Apply
    oof["pred_cal"] = apply_bins(oof, bias_map)
    val["pred_cal"] = apply_bins(val, bias_map)

    # Score Q1 2025 OOF for sanity
    q1_2025_mask = (oof["dt"].dt.year == 2025) & oof["dt"].dt.month.isin([1, 2, 3])
    q1_2025 = oof[q1_2025_mask]
    nm_raw = nmae(q1_2025["y_true"].values, np.clip(q1_2025["pred"].values, 0, 90.09))
    nm_cal = nmae(q1_2025["y_true"].values, np.clip(q1_2025["pred_cal"].values, 0, 90.09))
    print(f"EXP-069 Q1 2025 raw OOF: {nm_raw:.4f}%")
    print(f"EXP-069 Q1 2025 cal OOF: {nm_cal:.4f}%")

    # Save sub
    val_clipped = np.clip(val["pred_cal"].values, -1, 94.6)
    val_arr_reversed = val_clipped[::-1]
    pd.DataFrame({"prediction": val_arr_reversed}).to_csv(EXP_DIR / "sub_spatial_cal.csv", index=False)
    val_raw_clipped = np.clip(val["pred"].values, -1, 94.6)[::-1]
    pd.DataFrame({"prediction": val_raw_clipped}).to_csv(EXP_DIR / "sub_spatial_raw.csv", index=False)

    summary = {
        "exp_id": "EXP-074", "name": "spatial_single_3yr_cal",
        "base": "EXP-069", "bins": BINS, "cal_period": "Q1 2023+2024+2025",
        "q1_2025_raw_nmae": float(nm_raw),
        "q1_2025_cal_nmae": float(nm_cal),
        "cv_mean_nmae": float(nm_cal),
        "bias_map": bias_map,
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\nSaved sub_spatial_cal.csv (mean={val_clipped.mean():.2f}), sub_spatial_raw.csv (mean={val_raw_clipped.mean():.2f})")


if __name__ == "__main__":
    main()
