"""EXP-094: 5-base blend EXP-019/020/021/024/093b + per-ws-bin conformal cal Q1 2023+2024+2025."""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-094"
EXP_DIR.mkdir(parents=True, exist_ok=True)
N_INST = 90.09
N_TURB = 26
P_RATED_PER_TURB = 3.465


def nmae(y, p):
    return float(np.mean(np.abs(y - p))) / N_INST * 100.0


bases = {
    "EXP-019": ROOT / "experiments/active/EXP-019",
    "EXP-020": ROOT / "experiments/active/EXP-020",
    "EXP-021": ROOT / "experiments/active/EXP-021",
    "EXP-024": ROOT / "experiments/active/EXP-024",
    "EXP-093b": ROOT / "experiments/active/EXP-093b",
}

oofs = {}
valids = {}
for name, path in bases.items():
    oof = pd.read_parquet(path / "oof.parquet")
    valid = pd.read_parquet(path / "valid_pred.parquet")
    if "pred_mw" in oof.columns:
        oof = oof.rename(columns={"pred_mw": "pred"})
    elif "oof_pred" in oof.columns:
        oof = oof.rename(columns={"oof_pred": "pred"})
    if "y_true" in oof.columns:
        oof = oof.rename(columns={"y_true": "y"})
    oof["dt"] = pd.to_datetime(oof["dt"])
    valid["dt"] = pd.to_datetime(valid["dt"])
    oof = oof.sort_values("dt").drop_duplicates("dt", keep="first")
    valid = valid.sort_values("dt").drop_duplicates("dt", keep="first")
    oofs[name] = oof
    valids[name] = valid

common_oof = None
for n, df in oofs.items():
    common_oof = df["dt"] if common_oof is None else common_oof[common_oof.isin(df["dt"])]
common_oof = sorted(common_oof.tolist())

common_valid = None
for n, df in valids.items():
    common_valid = df["dt"] if common_valid is None else common_valid[common_valid.isin(df["dt"])]
common_valid = sorted(common_valid.tolist())

print(f"Common OOF rows: {len(common_oof)}, valid: {len(common_valid)}")

# Build aligned OOF arrays
y_arr = oofs["EXP-019"].set_index("dt").loc[common_oof, "y"].values
oof_preds = {n: oofs[n].set_index("dt").loc[common_oof, "pred"].values for n in oofs}

dt_oof = pd.DatetimeIndex(common_oof)
year = dt_oof.year.values
month = dt_oof.month.values
q1 = np.isin(month, [1, 2, 3])
q1_2023 = q1 & (year == 2023)
q1_2024 = q1 & (year == 2024)
q1_2025 = q1 & (year == 2025)

# Need ws for bin: load train parquet for ws
train_pq = ROOT / "data/processed/train_with_physics.parquet"
train_full = pd.read_parquet(train_pq)
train_full["dt"] = pd.to_datetime(train_full.get("METEOFORECASTHOUR_OPENM_Datetime", train_full.get("dt")))
ws_map = train_full[["dt", "wind_speed_120m"]].drop_duplicates("dt").set_index("dt")["wind_speed_120m"]
ws_oof = pd.Series(common_oof).map(ws_map).values

# valid ws
valid_pq = ROOT / "data/processed/valid_with_physics.parquet"
valid_full = pd.read_parquet(valid_pq)
valid_full["dt"] = pd.to_datetime(valid_full.get("METEOFORECASTHOUR_OPENM_Datetime", valid_full.get("dt")))
ws_v_map = valid_full[["dt", "wind_speed_120m"]].drop_duplicates("dt").set_index("dt")["wind_speed_120m"]
ws_valid = pd.Series(common_valid).map(ws_v_map).values
print(f"WS oof NaN: {pd.isna(ws_oof).sum()}, ws valid NaN: {pd.isna(ws_valid).sum()}")

bins = [0, 4, 6, 8, 10, 12, 14, 16, 60]
ws_oof_bin = np.digitize(np.nan_to_num(ws_oof, nan=0.0), bins, right=False)
ws_v_bin = np.digitize(np.nan_to_num(ws_valid, nan=0.0), bins, right=False)


def conformal_3yr(preds, cal_mask, all_q1):
    """Per-bin median bias on 3yr Q1 cal, return calibrated preds."""
    bias_by_bin = {}
    for b in np.unique(ws_oof_bin):
        m = cal_mask & (ws_oof_bin == b)
        if m.sum() >= 10:
            bias_by_bin[b] = float(np.median(y_arr[m] - preds[m]))
        else:
            bias_by_bin[b] = 0.0
    cal_add = np.array([bias_by_bin.get(b, 0.0) for b in ws_oof_bin])
    return np.clip(preds + cal_add, 0.0, N_INST), bias_by_bin


# Per-base Q1 2025 nMAE
print("\nPer-base OOF Q1 2025 nMAE:")
for n in oof_preds:
    print(f"  {n}: {nmae(y_arr[q1_2025], oof_preds[n][q1_2025]):.4f}%")

# Optuna 5-base weight tuning
names = list(oof_preds.keys())


def objective(trial):
    w = np.array([trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in names])
    s = w.sum()
    if s < 1e-6:
        return 100.0
    w = w / s
    blended = sum(w[i] * oof_preds[names[i]] for i in range(len(names)))
    cal_mask = q1_2023 | q1_2024 | q1_2025
    cal, _ = conformal_3yr(blended, cal_mask, q1)
    return nmae(y_arr[q1_2025], cal[q1_2025])


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=600, show_progress_bar=False)

bp = study.best_params
total = sum(bp.values())
w_dict = {n: bp[f"w_{n}"] / total for n in names}
print(f"\n[EXP-094] Best 5-base 3yr cal Q1 2025: {study.best_value:.4f}%")
print("Weights:")
for n in names:
    print(f"  {n}: {w_dict[n]:.4f}")

# Compute final cal bias_by_bin on Q1 2023+2024+2025 with optimal weights
blended_oof = sum(w_dict[n] * oof_preds[n] for n in names)
cal_mask = q1_2023 | q1_2024 | q1_2025
_, bias_by_bin = conformal_3yr(blended_oof, cal_mask, q1)
print(f"Bias by bin: {bias_by_bin}")

# Apply to valid
valid_preds = {n: valids[n].set_index("dt").loc[common_valid, "pred_mw"].values for n in names}
valid_n_avail = valids["EXP-019"].set_index("dt").loc[common_valid, "n_avail"].values
blended_valid = sum(w_dict[n] * valid_preds[n] for n in names)
cal_add_v = np.array([bias_by_bin.get(b, 0.0) for b in ws_v_bin])
final_valid = np.clip(blended_valid + cal_add_v, 0.0, valid_n_avail * P_RATED_PER_TURB)

print(f"\nValid prediction summary:")
print(f"  rows: {len(final_valid)}")
print(f"  min: {final_valid.min():.3f}, max: {final_valid.max():.3f}, mean: {final_valid.mean():.3f}")

# Save valid_pred parquet for use with predict.py
out_df = pd.DataFrame({"dt": common_valid, "pred_mw": final_valid,
                       "p_phys": np.nan, "n_avail": valid_n_avail})
out_df.to_parquet(EXP_DIR / "valid_pred.parquet")

# Save oof for downstream
oof_out = pd.DataFrame({"dt": common_oof, "y_true": y_arr, "oof_pred": blended_oof,
                        "oof_pred_cal": np.clip(blended_oof + np.array([bias_by_bin.get(b, 0.0) for b in ws_oof_bin]), 0, N_INST)})
oof_out.to_parquet(EXP_DIR / "oof.parquet")

summary = {
    "exp_id": "EXP-094",
    "weights": w_dict,
    "q1_2025_oof_cal_nmae": round(study.best_value, 4),
    "vs_exp054_cal": round(7.2782 - study.best_value, 4),
    "bias_by_bin": bias_by_bin,
}
(EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print(f"\nSaved to {EXP_DIR}/valid_pred.parquet and oof.parquet")
