"""
EXP-073: Optuna ensemble {EXP-019, EXP-020, EXP-021, EXP-024, EXP-069}
с 3yr conformal cal per base (same as EXP-054 paradigm).

EXP-069 (spatial NWP features) - новая база. CV 8.2086 - best single base.
EXP-054 winner - Optuna weights над 4 cal'd bases (019/020/021/024). 0.74 weight на EXP-021.

Hypothesis: adding spatial diversity -> better blend. Expected LB win если EXP-069 OOF orthogonal to others.

Process:
1. Load 5 OOFs.
2. Apply 3yr conformal cal (BINS=[0,3,5,7,9,11,13,16,30], cal_period Q1 2023-2025) per base.
3. Optuna 300 trials on cal'd OOFs Q1 2025 (since cal fitted on Q1 2023+2024).
4. Apply weights to valid predictions.
5. Save sub_spatial_ens.csv.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import nmae

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-073"
EXP_DIR.mkdir(parents=True, exist_ok=True)

BINS = [0, 3, 5, 7, 9, 11, 13, 16, 30]


def load_oof_valid(exp_id):
    base = ROOT / "experiments/archive/promoted" / exp_id
    if not base.exists():
        base = ROOT / "experiments/active" / exp_id
    oof = pd.read_parquet(base / "oof.parquet")
    valid = pd.read_parquet(base / "valid_pred.parquet")
    return oof, valid


def fit_bins_bias(cal_df, ws_col="ws_120", bins=BINS, min_n=30):
    """Returns dict {bin_idx: bias}, where bias = median(y_true - pred) per bin."""
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


def apply_bins_bias(df, bias_map, ws_col="ws_120", bins=BINS):
    bin_idx = np.digitize(df[ws_col].values, bins=bins[1:-1])
    bias = np.array([bias_map.get(int(b), 0.0) for b in bin_idx])
    return df["pred"].values + bias


def apply_conformal_cal(oof, valid, ws_train, ws_valid):
    """Apply 3yr cal Q1 2023+2024+2025 on oof, valid.
    Returns cal'd oof q1_2025 (for Optuna), cal'd full oof, cal'd valid."""
    oof = oof.copy().rename(columns={"oof_pred": "pred"})
    valid = valid.copy().rename(columns={"pred_mw": "pred"})
    oof["dt"] = pd.to_datetime(oof["dt"])
    valid["dt"] = pd.to_datetime(valid["dt"])
    oof["ws_120"] = ws_train
    valid["ws_120"] = ws_valid

    # Cal period: Q1 2023+2024+2025
    cal_mask = oof["dt"].dt.month.isin([1, 2, 3]) & oof["dt"].dt.year.isin([2023, 2024, 2025])
    cal_df = oof[cal_mask].copy()
    bias_map = fit_bins_bias(cal_df)

    # Apply cal to ALL oof + valid
    oof_cal_pred = apply_bins_bias(oof, bias_map)
    valid_cal_pred = apply_bins_bias(valid, bias_map)

    return oof_cal_pred, valid_cal_pred


def main():
    t0 = time.time()
    # Load ws_120 base (used as bin variable for conformal cal)
    train = pd.read_parquet(ROOT / "data/processed/train_with_physics.parquet")
    train["dt"] = pd.to_datetime(train["METEOFORECASTHOUR_OPENM_Datetime"])
    valid = pd.read_parquet(ROOT / "data/processed/valid_with_physics.parquet")
    valid["dt"] = pd.to_datetime(valid["METEOFORECASTHOUR_OPENM_Datetime"])
    ws_train_full = train.set_index("dt")["wind_speed_120m"]
    ws_valid = valid.set_index("dt")["wind_speed_120m"]

    bases = ["EXP-019", "EXP-020", "EXP-021", "EXP-024", "EXP-069"]
    oofs_cal = {}
    valids_cal = {}
    for b in bases:
        print(f"Loading {b} ...")
        oof, val = load_oof_valid(b)
        oof["dt"] = pd.to_datetime(oof["dt"])
        val["dt"] = pd.to_datetime(val["dt"])
        # Align ws_120 (some bases may have train_filter applied differently)
        ws_oof = oof["dt"].map(ws_train_full).values
        ws_val = val["dt"].map(ws_valid).values
        oof_pred_cal, val_pred_cal = apply_conformal_cal(oof, val, ws_oof, ws_val)
        oof_clipped = np.clip(oof_pred_cal, 0, 90.09)
        val_clipped = np.clip(val_pred_cal, 0, 90.09)
        # Q1 2025 OOF for Optuna optimization
        q1_2025_mask = (oof["dt"].dt.year == 2025) & (oof["dt"].dt.month.isin([1, 2, 3]))
        q1_2025_oof_cal = oof_clipped[q1_2025_mask.values]
        q1_2025_y = oof.loc[q1_2025_mask, "y_true"].values
        oofs_cal[b] = (q1_2025_oof_cal, q1_2025_y, oof, oof_clipped)
        valids_cal[b] = (val, val_clipped)
        print(f"  {b}: cal'd Q1 2025 OOF nMAE = {nmae(q1_2025_y, q1_2025_oof_cal):.4f}%, oof rows={len(oof)}")

    # Find common q1_2025 dts across bases (use pd.Timestamp for set type consistency)
    common_dt = set(pd.to_datetime(oofs_cal[bases[0]][2]["dt"].values).to_list())
    for b in bases[1:]:
        common_dt &= set(pd.to_datetime(oofs_cal[b][2]["dt"].values).to_list())
    print(f"\nCommon dts across {len(bases)} bases: {len(common_dt)}")

    # Build aligned Q1 2025 matrix
    ref_oof = oofs_cal["EXP-021"][2]
    ref_oof_idx = ref_oof.set_index("dt")
    q1_2025_dt_all = ref_oof_idx[(ref_oof_idx.index.year == 2025) & (ref_oof_idx.index.month.isin([1, 2, 3]))].index
    q1_2025_dt = [d for d in q1_2025_dt_all if d in common_dt]
    print(f"Aligned Q1 2025 rows: {len(q1_2025_dt)}")

    y_true = ref_oof.set_index("dt").loc[q1_2025_dt]["y_true"].values
    cal_preds_matrix = []
    for b in bases:
        # Map cal'd full oof to dt
        oof, oof_clipped = oofs_cal[b][2], oofs_cal[b][3]
        cal_map = pd.Series(oof_clipped, index=oof["dt"].values)
        aligned = cal_map.loc[q1_2025_dt].values
        cal_preds_matrix.append(aligned)
    cal_preds_matrix = np.array(cal_preds_matrix)  # (n_bases, n_q1_2025)
    print(f"Shape: {cal_preds_matrix.shape}")
    for i, b in enumerate(bases):
        print(f"  {b} individual cal'd Q1 2025: {nmae(y_true, cal_preds_matrix[i]):.4f}%")
    # Equal-avg baseline
    eq = cal_preds_matrix.mean(axis=0)
    print(f"\nEqual 5-base cal'd avg Q1 2025: {nmae(y_true, eq):.4f}%")

    # Optuna 300 trials
    def objective(trial):
        w = np.array([trial.suggest_float(f"w_{b}", 0.0, 1.0) for b in bases])
        if w.sum() == 0:
            return 1e10
        w = w / w.sum()
        e = (cal_preds_matrix * w[:, None]).sum(axis=0)
        return float(nmae(y_true, e))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=400, show_progress_bar=False)
    print(f"\nOptuna best Q1 2025 OOF: {study.best_value:.4f}%")
    w_opt = np.array([study.best_params[f"w_{b}"] for b in bases])
    w_opt /= w_opt.sum()
    print(f"Optimal weights:")
    for b, w in zip(bases, w_opt):
        print(f"  {b}: {w:.4f}")

    # Build valid prediction
    valid_aligned_dt = valids_cal["EXP-021"][0]["dt"].values
    val_matrix = []
    for b in bases:
        v_df, v_clipped = valids_cal[b]
        val_map = pd.Series(v_clipped, index=v_df["dt"].values)
        aligned = val_map.loc[valid_aligned_dt].values
        val_matrix.append(aligned)
    val_matrix = np.array(val_matrix)
    val_opt = (val_matrix * w_opt[:, None]).sum(axis=0)
    val_eq = val_matrix.mean(axis=0)

    # Reverse + clip + save CSV (per predict.py pattern)
    val_opt_clipped = np.clip(val_opt, -1, 94.6)
    val_eq_clipped = np.clip(val_eq, -1, 94.6)
    # NOTE: predict.py делает iloc[::-1] на parquet с dt asc -> reverse to platform order
    # Save valid_pred parquet first
    pd.DataFrame({"dt": valid_aligned_dt, "pred_mw": val_opt}).to_parquet(EXP_DIR / "valid_pred_optuna.parquet")
    pd.DataFrame({"dt": valid_aligned_dt, "pred_mw": val_eq}).to_parquet(EXP_DIR / "valid_pred_equal.parquet")

    # Build sub CSV - reverse (newest first as platform expects)
    arr_opt = val_opt_clipped[::-1]
    arr_eq = val_eq_clipped[::-1]
    pd.DataFrame({"prediction": arr_opt}).to_csv(EXP_DIR / "sub_opt.csv", index=False)
    pd.DataFrame({"prediction": arr_eq}).to_csv(EXP_DIR / "sub_equal.csv", index=False)
    print(f"\nSaved: {EXP_DIR}/sub_opt.csv (mean={val_opt.mean():.2f}), sub_equal.csv (mean={val_eq.mean():.2f})")

    summary = {
        "exp_id": "EXP-073",
        "name": "spatial_in_5base_ensemble",
        "bases": bases,
        "bins": BINS,
        "cal_period": "Q1 2023+2024+2025",
        "optuna_q1_2025_oof": float(study.best_value),
        "optuna_weights": dict(zip(bases, w_opt.tolist())),
        "equal_5base_q1_2025_oof": float(nmae(y_true, eq)),
        "individual_q1_2025": {b: float(nmae(y_true, cal_preds_matrix[i])) for i, b in enumerate(bases)},
        "best_lb_reference": 7.2977,
        "exp_054_optuna_q1_2025": 7.2782,
        "elapsed_sec": time.time() - t0,
    }
    (EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\nDone: {EXP_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
