"""
EXP-090 cal+blend: применить flat conformal cal к EXP-090 OOF + Optuna blend
с базами EXP-019/020/024 (replace EXP-021).

Использует EXP-054 paradigm: 3yr conformal Q1 2023+2024+2025, BINS=[0,3,5,7,9,11,13,16,30].
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
from common.physics_features import nmae

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-090"
BINS = [0, 3, 5, 7, 9, 11, 13, 16, 30]
BASES = ["EXP-019", "EXP-020", "EXP-024", "EXP-090"]  # replace 021 with 090


def load_base(exp_id):
    base = ROOT / "experiments/active" / exp_id
    oof = pd.read_parquet(base / "oof.parquet")
    val = pd.read_parquet(base / "valid_pred.parquet")
    oof["dt"] = pd.to_datetime(oof["dt"])
    val["dt"] = pd.to_datetime(val["dt"])
    return oof, val


def load_ws():
    train = pd.read_parquet(ROOT / "data/processed/train_with_physics.parquet")
    train["dt"] = pd.to_datetime(train["METEOFORECASTHOUR_OPENM_Datetime"])
    valid = pd.read_parquet(ROOT / "data/processed/valid_with_physics.parquet")
    valid["dt"] = pd.to_datetime(valid["METEOFORECASTHOUR_OPENM_Datetime"])
    ws_train = train.set_index("dt")["wind_speed_120m"]
    ws_valid = valid.set_index("dt")["wind_speed_120m"]
    return ws_train, ws_valid


def fit_bins_bias(cal_df, ws_col="ws", bins=BINS, min_n=30):
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


def apply_bins_bias(df, bias_map, ws_col="ws", bins=BINS):
    bin_idx = np.digitize(df[ws_col].values, bins=bins[1:-1])
    bias = np.array([bias_map.get(int(b), 0.0) for b in bin_idx])
    return df["pred"].values + bias


def main():
    t0 = time.time()
    print("="*70)
    print("EXP-090 cal+blend")
    print("="*70)
    ws_train, ws_valid = load_ws()
    bundle = {}
    for b in BASES:
        oof, val = load_base(b)
        oof = oof.rename(columns={"oof_pred": "pred"})
        val = val.rename(columns={"pred_mw": "pred"})
        oof["ws"] = oof["dt"].map(ws_train).values
        val["ws"] = val["dt"].map(ws_valid).values
        # Fit cal на Q1 2023+2024+2025
        cal_mask = oof["dt"].dt.month.isin([1,2,3]) & oof["dt"].dt.year.isin([2023,2024,2025])
        bias_map = fit_bins_bias(oof[cal_mask])
        oof["pred_cal"] = np.clip(apply_bins_bias(oof, bias_map), 0, 90.09)
        val["pred_cal"] = np.clip(apply_bins_bias(val, bias_map), 0, 90.09)
        q25_mask = (oof["dt"].dt.year == 2025) & oof["dt"].dt.month.isin([1,2,3])
        q1_2025_nmae = nmae(oof.loc[q25_mask, "y_true"].values, oof.loc[q25_mask, "pred_cal"].values)
        oof_uncal_nmae = nmae(oof.loc[q25_mask, "y_true"].values, oof.loc[q25_mask, "pred"].values)
        bundle[b] = {"oof": oof, "val": val, "bias_map": bias_map}
        print(f"  {b:8s}: Q1 2025 OOF raw nMAE = {oof_uncal_nmae:.4f}, cal = {q1_2025_nmae:.4f}")

    # Optuna blend
    print("\n--- Optuna blend (400 trials) ---")
    common_dt = None
    for b in BASES:
        q25 = bundle[b]["oof"][(bundle[b]["oof"]["dt"].dt.year == 2025) & bundle[b]["oof"]["dt"].dt.month.isin([1,2,3])].sort_values("dt")
        bundle[b]["oof_q25_cal"] = q25
        if common_dt is None:
            common_dt = set(q25["dt"].values)
        else:
            common_dt &= set(q25["dt"].values)
    common_dt = pd.to_datetime(sorted(list(common_dt)))
    cols = []; y = None
    for b in BASES:
        q25 = bundle[b]["oof_q25_cal"]
        q25 = q25[q25["dt"].isin(common_dt)].sort_values("dt").reset_index(drop=True)
        cols.append(q25["pred_cal"].values)
        if y is None: y = q25["y_true"].values
    X = np.stack(cols, axis=1)
    def obj(trial):
        w = np.array([trial.suggest_float(b, 0, 1) for b in BASES])
        s = w.sum()
        if s < 1e-6: return 100
        return nmae(y, X @ (w/s))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=400, show_progress_bar=False)
    best_w = np.array([study.best_params[b] for b in BASES])
    best_w = best_w / best_w.sum()
    print(f"  best Q1 2025 OOF cal nMAE = {study.best_value:.4f} (vs EXP-054 7.2782)")
    print(f"  weights:")
    for b, w in zip(BASES, best_w):
        print(f"    {b:10s}: {w:.3f}")

    # Apply weights to valid
    val_stack = np.stack([bundle[b]["val"]["pred_cal"].values for b in BASES], axis=1)
    valid_blend = np.clip(val_stack @ best_w, 0, 90.09)
    print(f"\n  valid blend: mean={valid_blend.mean():.3f}, sum={valid_blend.sum():.1f}")

    # Save submission (df.iloc[::-1] reverse order)
    val = bundle[BASES[0]]["val"].copy().sort_values("dt").reset_index(drop=True)
    val["prediction"] = valid_blend
    sub = val[["prediction"]].iloc[::-1].reset_index(drop=True)
    sub_path = EXP_DIR / "sub_phys_blend.csv"
    sub.to_csv(sub_path, index=False)
    print(f"  saved: {sub_path}")

    summary = {
        "exp_id": "EXP-090",
        "blend": "EXP-019/020/024/090",
        "weights": dict(zip(BASES, best_w.tolist())),
        "q1_2025_oof_cal_nmae": float(study.best_value),
        "vs_EXP_054": 7.2782 - float(study.best_value),
        "valid_mean_mw": float(valid_blend.mean()),
        "valid_sum_mwh": float(valid_blend.sum()),
        "time_sec": time.time() - t0,
    }
    (EXP_DIR / "blend_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
