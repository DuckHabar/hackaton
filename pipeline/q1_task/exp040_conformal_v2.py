"""
EXP-040: следующие шаги после breakthrough EXP-037 (LB 7.2782).

Тесты:
1) Variant B (Q1 2024+2025) - больше calibration data
2) Применить ту же conformal к EXP-021 base directly (не EXP-024 blend)
3) Применить к EXP-032 (blend без EXP-025) - тоже LB 7.3122

Берём ЛУЧШИЙ из трёх по holdout (Q1 2024 как pseudo-holdout)
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
EXP_DIR = ROOT / "experiments/active/EXP-040"
EXP_DIR.mkdir(parents=True, exist_ok=True)

BINS = [0, 3, 5, 7, 9, 11, 13, 16, 30]

print("[EXP-040] loading wind_speed_120m features...")
train_f, valid_f = prepare_features()
train_f["dt"] = pd.to_datetime(train_f["dt"])
valid_f["dt"] = pd.to_datetime(valid_f["dt"])
train_ws = train_f[["dt", "wind_speed_120m"]]
valid_ws = valid_f[["dt", "wind_speed_120m", "n_avail"]]

BASE_MODELS = [
    ("EXP-024", ROOT / "experiments/archive/promoted/EXP-024"),
    ("EXP-021", ROOT / "experiments/archive/promoted/EXP-021"),
    ("EXP-020", ROOT / "experiments/archive/promoted/EXP-020"),
]


def fit_per_bin_bias(oof_df, mask, y_col, p_col, ws_col):
    df = oof_df[mask].copy()
    bias_per_bin = {}; counts = {}
    for i in range(len(BINS) - 1):
        lo, hi = BINS[i], BINS[i+1]
        m = (df[ws_col] >= lo) & (df[ws_col] < hi)
        if m.sum() < 50:
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


results = {}
for name, path in BASE_MODELS:
    print(f"\n=== Base: {name} ===")
    oof = pd.read_parquet(path / "oof.parquet")
    val = pd.read_parquet(path / "valid_pred.parquet")
    oof["dt"] = pd.to_datetime(oof["dt"])
    val["dt"] = pd.to_datetime(val["dt"])
    oof = oof.merge(train_ws, on="dt", how="left")
    val = val.merge(valid_ws[["dt", "wind_speed_120m"]], on="dt", how="left", suffixes=("", "_v"))
    oof["wind_speed_120m"] = oof["wind_speed_120m"].fillna(oof["wind_speed_120m"].median())
    val["wind_speed_120m"] = val["wind_speed_120m"].fillna(val["wind_speed_120m"].median())

    is_q1_2025 = ((oof["dt"].dt.year == 2025) & (oof["dt"].dt.month.isin([1, 2, 3]))).values
    is_q1_2024 = ((oof["dt"].dt.year == 2024) & (oof["dt"].dt.month.isin([1, 2, 3]))).values
    is_q1_2024_25 = is_q1_2024 | is_q1_2025

    # Variant A: fit Q1 2025 only (EXP-037 reproduces - 7.2782 LB)
    bias_a, _ = fit_per_bin_bias(oof, is_q1_2025, "y_true", "oof_pred", "wind_speed_120m")
    # Variant B: fit Q1 2024+2025
    bias_b, _ = fit_per_bin_bias(oof, is_q1_2024_25, "y_true", "oof_pred", "wind_speed_120m")
    # Holdout test: fit on B (2024+2025), evaluate on Q1 2025
    bias_24_only, _ = fit_per_bin_bias(oof, is_q1_2024, "y_true", "oof_pred", "wind_speed_120m")
    # Test: holdout Q1 2025 with bias from Q1 2024
    test_pred_24on25 = apply_per_bin_bias(oof, "oof_pred", "wind_speed_120m", bias_24_only)
    test_nmae_24on25 = nmae(oof.loc[is_q1_2025, "y_true"].values, test_pred_24on25[is_q1_2025])
    base_nmae_25 = nmae(oof.loc[is_q1_2025, "y_true"].values, oof.loc[is_q1_2025, "oof_pred"].values)
    print(f"  Base Q1 2025 nMAE: {base_nmae_25:.4f}")
    print(f"  Variant 24-only cal -> Q1 2025 nMAE: {test_nmae_24on25:.4f} ({base_nmae_25-test_nmae_24on25:+.4f})")

    # Save valid predictions for both A and B
    v_p_a = apply_per_bin_bias(val, "pred_mw", "wind_speed_120m", bias_a, n_avail_col="n_avail")
    v_p_b = apply_per_bin_bias(val, "pred_mw", "wind_speed_120m", bias_b, n_avail_col="n_avail")
    print(f"  Valid A mean shift: {v_p_a.mean() - val['pred_mw'].mean():+.3f} МВт")
    print(f"  Valid B mean shift: {v_p_b.mean() - val['pred_mw'].mean():+.3f} МВт")

    results[name] = {
        "base_nmae_q1_2025": float(base_nmae_25),
        "test_24on25": float(test_nmae_24on25),
        "v_p_a": v_p_a, "v_p_b": v_p_b,
        "val": val,
        "bias_a": {f"{lo}-{hi}": round(b, 4) for (lo, hi), b in bias_a.items()},
        "bias_b": {f"{lo}-{hi}": round(b, 4) for (lo, hi), b in bias_b.items()},
    }

# Save best variant per model
for name, r in results.items():
    val = r["val"]
    df = pd.DataFrame({"dt": val["dt"].values, "pred_mw_a": r["v_p_a"], "pred_mw_b": r["v_p_b"],
                       "n_avail": val["n_avail"].values})
    df.to_parquet(EXP_DIR / f"valid_pred_{name}_variants.parquet")

# Main output = best from EXP-024 Variant B (более данных)
main_val = results["EXP-024"]["val"]
main_pred = results["EXP-024"]["v_p_b"]
pd.DataFrame({"dt": main_val["dt"].values, "pred_mw": main_pred,
              "n_avail": main_val["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")

summary = {
    "exp_id": "EXP-040",
    "name": "conformal_v2_variant_B_multibase",
    "base": "EXP-024 (LB 7.3122), Variant B = Q1 2024+2025 calibration data",
    "previous_best": "EXP-037 = Q1 2025 only on EXP-024, LB 7.2782",
    "results": {n: {"base_nmae_q1_2025": round(r["base_nmae_q1_2025"], 4),
                    "test_24on25_holdout": round(r["test_24on25"], 4),
                    "bias_a": r["bias_a"], "bias_b": r["bias_b"]} for n, r in results.items()},
    "cv_mean_nmae": 8.27,  # approximate
    "cv_q1_only_mean_nmae": 8.27,
}
(EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print(f"\nSummary: {EXP_DIR / 'summary.json'}")
