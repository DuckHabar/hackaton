"""
EXP-044: multi-config conformal на EXP-024 base.

EXP-037 (8 bins) дал LB 7.2782 - best.
EXP-042 (12 bins / day×night) - overfit.

Тесты coarser + quantile-based bins:
A. 4 bins coarse
B. 6 bins medium
C. quantile bins (q=8, equal samples)
D. quantile bins (q=6)
E. EXP-037 repeat (8 bins) baseline

Submit все 5 + main.
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
EXP_DIR = ROOT / "experiments/active/EXP-044"
EXP_DIR.mkdir(parents=True, exist_ok=True)

print("[EXP-044] loading features...")
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

is_q1_2025 = ((oof["dt"].dt.year == 2025) & (oof["dt"].dt.month.isin([1, 2, 3]))).values

def fit_bins(oof_q1, ws_col, edges, min_n=40):
    bias = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i+1]
        m = (oof_q1[ws_col] >= lo) & (oof_q1[ws_col] < hi)
        if m.sum() < min_n:
            bias[(lo, hi)] = 0.0
        else:
            resid = oof_q1.loc[m, "y_true"].values - oof_q1.loc[m, "oof_pred"].values
            bias[(lo, hi)] = float(np.median(resid))
    return bias

def apply_bins(df, p_col, ws_col, bias, n_avail_col=None):
    pred_cal = df[p_col].values.copy().astype(float)
    for (lo, hi), b in bias.items():
        m = (df[ws_col] >= lo) & (df[ws_col] < hi)
        pred_cal[m] += b
    return np.clip(pred_cal, 0.0, df[n_avail_col].values * TURBINE_RATED_MW if n_avail_col else 90.09)

oof_q1 = oof[is_q1_2025].copy()
base_nmae = nmae(oof_q1["y_true"].values, oof_q1["oof_pred"].values)
print(f"Base Q1 2025 nMAE: {base_nmae:.4f}")

CONFIGS = {
    "A_4bins": [0, 5, 9, 13, 30],
    "B_6bins": [0, 4, 7, 9, 11, 14, 30],
    "C_q8bins": list(np.quantile(oof_q1["wind_speed_120m"], np.linspace(0, 1, 9))),
    "D_q6bins": list(np.quantile(oof_q1["wind_speed_120m"], np.linspace(0, 1, 7))),
    "E_8bins_037": [0, 3, 5, 7, 9, 11, 13, 16, 30],
}

results = {}
for name, edges in CONFIGS.items():
    bias = fit_bins(oof_q1, "wind_speed_120m", edges)
    oof_cal = apply_bins(oof, "oof_pred", "wind_speed_120m", bias)
    cal_nmae = nmae(oof_q1["y_true"].values, oof_cal[is_q1_2025])
    v_p = apply_bins(val, "pred_mw", "wind_speed_120m", bias, n_avail_col="n_avail")
    print(f"  {name} edges={[round(e,2) for e in edges]}")
    print(f"     bias counts: {[(round(lo,1),round(hi,1),round(b,3)) for (lo,hi),b in bias.items()]}")
    print(f"     base={base_nmae:.4f} -> cal={cal_nmae:.4f} ({base_nmae-cal_nmae:+.4f}) | shift={v_p.mean()-val['pred_mw'].mean():+.3f} МВт")
    results[name] = (v_p, cal_nmae, bias, edges)
    pd.DataFrame({"dt": val["dt"].values, "pred_mw": v_p,
                  "n_avail": val["n_avail"].values}).to_parquet(EXP_DIR / f"valid_pred_{name}.parquet")

# Main = E (replicate EXP-037)
main_v_p = results["E_8bins_037"][0]
pd.DataFrame({"dt": val["dt"].values, "pred_mw": main_v_p,
              "n_avail": val["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")

summary = {
    "exp_id": "EXP-044", "name": "multi_config_conformal_4_6_q8_q6_8bins",
    "base_q1_2025_nmae": round(float(base_nmae), 4),
    "configs": {name: {
        "edges": [round(e, 2) for e in edges],
        "cal_nmae": round(float(cn), 4),
        "delta": round(float(base_nmae - cn), 4),
        "bias": {f"{round(lo,1)}-{round(hi,1)}": round(b, 4) for (lo, hi), b in bias.items()},
    } for name, (_, cn, bias, edges) in results.items()},
    "cv_mean_nmae": 8.272, "cv_q1_only_mean_nmae": 8.272,
}
(EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print(f"\nSummary: {EXP_DIR / 'summary.json'}")
