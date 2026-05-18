"""
EXP-091 P1: Обогащённый conformal cal с 4D-бинами.
Гипотеза: shifts из LB-mining могут быть воспроизведены через структурированную
калибровку bias по (ws_bin × is_day_night × is_eve × n_rep3) без retrain модели.

Шаги:
1. Load 4 base OOFs (EXP-019, 020, 021, 024) и valid predictions.
2. Для каждой base: fit 4D bias map на cal Q1 2023+2024+2025 с Bayesian shrinkage.
3. Forward-walking validation: hold-out каждого Q1 года, фит на остальных, eval.
4. Optuna blend cal'd OOFs Q1 2025 -> optimal weights.
5. Apply на valid -> sub_phys_v1.csv. Сравнить с v16 mining.
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
EXP_DIR = ROOT / "experiments/active/EXP-091"
EXP_DIR.mkdir(parents=True, exist_ok=True)

BIN_EDGES = [0, 3, 5, 7, 9, 11, 13, 16, 30]

# -------- Загрузка / выравнивание --------------------------------------------

def load_base(exp_id):
    base = ROOT / "experiments/active" / exp_id
    oof = pd.read_parquet(base / "oof.parquet")
    val = pd.read_parquet(base / "valid_pred.parquet")
    oof["dt"] = pd.to_datetime(oof["dt"])
    val["dt"] = pd.to_datetime(val["dt"])
    return oof, val

def load_context():
    """Загрузить train+valid для извлечения ws_120m, hour, n_repair."""
    train = pd.read_parquet(ROOT / "data/processed/train_with_physics.parquet")
    train["dt"] = pd.to_datetime(train["METEOFORECASTHOUR_OPENM_Datetime"])
    valid = pd.read_parquet(ROOT / "data/processed/valid_with_physics.parquet")
    valid["dt"] = pd.to_datetime(valid["METEOFORECASTHOUR_OPENM_Datetime"])
    cols = ["dt", "wind_speed_120m", "Кол-во_ВЭУ_в_ремонте"]
    return train[cols], valid[cols]

def add_strata(df, train_ctx, valid_ctx):
    """Расширить df колонками: ws_120, hour, n_rep, bin_ws, is_day_strict, is_eve, n_rep3."""
    ctx = pd.concat([train_ctx, valid_ctx], ignore_index=True)
    df = df.merge(ctx, on="dt", how="left", suffixes=("", "_ctx"))
    df["ws_120"] = df["wind_speed_120m"]
    df["n_rep"] = df["Кол-во_ВЭУ_в_ремонте"].fillna(0).astype(int)
    df["hour"] = df["dt"].dt.hour
    df["bin_ws"] = np.digitize(df["ws_120"].values, BIN_EDGES[1:-1])
    df["is_day_strict"] = ((df["hour"] >= 6) & (df["hour"] < 16)).astype(int)
    df["is_eve"] = ((df["hour"] >= 16) & (df["hour"] < 20)).astype(int)
    df["is_night_strict"] = ((df["hour"] < 6) | (df["hour"] >= 20)).astype(int)
    df["n_rep3"] = (df["n_rep"] == 3).astype(int)
    return df

# -------- Калибратор 4D bins -------------------------------------------------

def fit_4d_bias(cal_df, ws_col_pred="pred", y_col="y_true", min_n=20, shrink_lambda=15.0):
    """Fit bias per (bin_ws, day_or_night, is_eve, n_rep3).
    Bayesian shrinkage: bias_cell = (n*median + λ*global_bin_median) / (n + λ)."""
    cal_df = cal_df.copy()
    cal_df["resid"] = cal_df[y_col] - cal_df[ws_col_pred]
    # Global per ws_bin (shrinkage target)
    global_bin = {}
    for b in cal_df["bin_ws"].unique():
        global_bin[int(b)] = float(cal_df.loc[cal_df["bin_ws"] == b, "resid"].median())
    # Per-cell
    cell_bias = {}
    cell_n = {}
    for (b, d, e, r), sub in cal_df.groupby(["bin_ws", "is_day_strict", "is_eve", "n_rep3"]):
        cell_med = float(sub["resid"].median())
        gm = global_bin.get(int(b), 0.0)
        n = len(sub)
        shrunk = (n * cell_med + shrink_lambda * gm) / (n + shrink_lambda)
        cell_bias[(int(b), int(d), int(e), int(r))] = shrunk
        cell_n[(int(b), int(d), int(e), int(r))] = n
    return cell_bias, global_bin, cell_n

def apply_4d_bias(df, cell_bias, global_bin, ws_col_pred="pred"):
    pred = df[ws_col_pred].values.copy()
    bias_arr = np.zeros(len(df))
    for i, (b, d, e, r) in enumerate(zip(df["bin_ws"], df["is_day_strict"], df["is_eve"], df["n_rep3"])):
        key = (int(b), int(d), int(e), int(r))
        if key in cell_bias:
            bias_arr[i] = cell_bias[key]
        else:
            bias_arr[i] = global_bin.get(int(b), 0.0)
    return pred + bias_arr

# -------- Forward-walking validation -----------------------------------------

def forward_walking_eval(oof_df, hold_year, base_name):
    """Hold-out hold_year Q1, fit cal на остальных 3 Q1, eval."""
    cal_mask = oof_df["dt"].dt.month.isin([1,2,3]) & oof_df["dt"].dt.year.isin([2022,2023,2024,2025])
    cal_mask &= oof_df["dt"].dt.year != hold_year
    hold_mask = oof_df["dt"].dt.month.isin([1,2,3]) & (oof_df["dt"].dt.year == hold_year)
    cal_df = oof_df[cal_mask]
    hold_df = oof_df[hold_mask].copy()
    cell_bias, global_bin, cell_n = fit_4d_bias(cal_df)
    hold_df["pred_cal"] = apply_4d_bias(hold_df, cell_bias, global_bin)
    hold_df["pred_cal"] = np.clip(hold_df["pred_cal"], 0, 90.09)
    # Flat conformal baseline (per ws_bin only)
    hold_df["pred_flat"] = hold_df["pred"].values
    flat_bias = {b: float((cal_df[cal_df["bin_ws"]==b]["y_true"] - cal_df[cal_df["bin_ws"]==b]["pred"]).median()) for b in cal_df["bin_ws"].unique()}
    hold_df["pred_flat"] = hold_df["pred"].values + hold_df["bin_ws"].map(flat_bias).fillna(0).values
    hold_df["pred_flat"] = np.clip(hold_df["pred_flat"], 0, 90.09)
    nmae_raw = nmae(hold_df["y_true"].values, hold_df["pred"].values)
    nmae_flat = nmae(hold_df["y_true"].values, hold_df["pred_flat"].values)
    nmae_4d = nmae(hold_df["y_true"].values, hold_df["pred_cal"].values)
    return {"year": hold_year, "n": len(hold_df), "raw": nmae_raw, "flat": nmae_flat, "4d": nmae_4d}

# -------- Main pipeline -------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 70)
    print("EXP-091 P1: 4D conformal cal")
    print("=" * 70)
    train_ctx, valid_ctx = load_context()
    bases = ["EXP-019", "EXP-020", "EXP-021", "EXP-024"]

    # Store all OOF + valid in dict for blend
    bundle = {}
    for b in bases:
        oof, val = load_base(b)
        oof = oof.rename(columns={"oof_pred": "pred"})
        val = val.rename(columns={"pred_mw": "pred"})
        oof = add_strata(oof, train_ctx, valid_ctx)
        val = add_strata(val, train_ctx, valid_ctx)
        bundle[b] = {"oof": oof, "val": val}
        # Forward-walking
        print(f"\n--- {b} forward-walking (Q1 hold-out) ---")
        print(f"  year | n     | raw   | flat  | 4d    | 4d-flat")
        results = []
        for yr in [2022, 2023, 2024, 2025]:
            r = forward_walking_eval(oof, yr, b)
            print(f"  {r['year']} | {r['n']:5d} | {r['raw']:.4f} | {r['flat']:.4f} | {r['4d']:.4f} | {r['4d']-r['flat']:+.4f}")
            results.append(r)
        bundle[b]["results"] = results
        mean_4d_minus_flat = np.mean([r["4d"] - r["flat"] for r in results])
        print(f"  MEAN 4d-flat: {mean_4d_minus_flat:+.4f} п.п.")

    # === Fit 4D bias on Q1 2023+2024+2025 (как EXP-054), apply to valid ===
    print("\n--- Fit 4D bias на Q1 2023+2024+2025, apply на valid ---")
    valid_cal = {}
    for b in bases:
        oof = bundle[b]["oof"]
        val = bundle[b]["val"]
        cal_mask = oof["dt"].dt.month.isin([1,2,3]) & oof["dt"].dt.year.isin([2023,2024,2025])
        cell_bias, global_bin, _ = fit_4d_bias(oof[cal_mask])
        # Apply to oof + val
        oof["pred_cal"] = np.clip(apply_4d_bias(oof, cell_bias, global_bin), 0, 90.09)
        val["pred_cal"] = np.clip(apply_4d_bias(val, cell_bias, global_bin), 0, 90.09)
        # Q1 2025 OOF nMAE
        q25_mask = (oof["dt"].dt.year == 2025) & oof["dt"].dt.month.isin([1,2,3])
        q1_2025_nmae = nmae(oof.loc[q25_mask, "y_true"].values, oof.loc[q25_mask, "pred_cal"].values)
        print(f"  {b}: Q1 2025 OOF cal nMAE = {q1_2025_nmae:.4f}")
        bundle[b]["oof_cal"] = oof
        bundle[b]["val_cal"] = val

    # === Optuna blend на Q1 2025 OOF cal ===
    print("\n--- Optuna blend (400 trials) на Q1 2025 cal'd OOFs ---")
    common_dt = None
    for b in bases:
        oof = bundle[b]["oof_cal"]
        q25 = oof[(oof["dt"].dt.year == 2025) & oof["dt"].dt.month.isin([1,2,3])].sort_values("dt")
        bundle[b]["oof_q25_cal"] = q25
        if common_dt is None:
            common_dt = set(q25["dt"].values)
        else:
            common_dt &= set(q25["dt"].values)
    common_dt = pd.to_datetime(sorted(list(common_dt)))
    # Stack predictions
    cols = []
    y = None
    for b in bases:
        q25 = bundle[b]["oof_q25_cal"]
        q25 = q25[q25["dt"].isin(common_dt)].sort_values("dt").reset_index(drop=True)
        cols.append(q25["pred_cal"].values)
        if y is None:
            y = q25["y_true"].values
    X = np.stack(cols, axis=1)
    def obj(trial):
        w = np.array([trial.suggest_float(b, 0, 1) for b in bases])
        s = w.sum()
        if s < 1e-6:
            return 100
        w = w / s
        p = X @ w
        return nmae(y, p)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=400, show_progress_bar=False)
    best_w = np.array([study.best_params[b] for b in bases])
    best_w = best_w / best_w.sum()
    print(f"  best Q1 2025 OOF cal nMAE = {study.best_value:.4f}")
    print(f"  weights: {dict(zip(bases, [round(x,3) for x in best_w]))}")

    # Apply weights to valid
    val_stack = []
    for b in bases:
        val_stack.append(bundle[b]["val_cal"]["pred_cal"].values)
    val_stack = np.stack(val_stack, axis=1)
    valid_blend = np.clip(val_stack @ best_w, 0, 90.09)
    print(f"\n  valid blend: mean={valid_blend.mean():.3f}, sum={valid_blend.sum():.1f}")

    # Save submission (df.iloc[::-1] reverse order)
    val = bundle[bases[0]]["val"].copy().sort_values("dt").reset_index(drop=True)
    val["prediction"] = valid_blend
    sub = val[["prediction"]].iloc[::-1].reset_index(drop=True)
    sub_path = EXP_DIR / "sub_phys_v1.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\n  saved: {sub_path}")

    # Summary
    summary = {
        "exp_id": "EXP-091",
        "name": "phys_4d_conformal",
        "bases": bases,
        "weights": dict(zip(bases, best_w.tolist())),
        "q1_2025_oof_cal_nmae": float(study.best_value),
        "valid_mean_mw": float(valid_blend.mean()),
        "valid_sum_mwh": float(valid_blend.sum()),
        "forward_walking_per_base": {b: bundle[b]["results"] for b in bases},
        "time_sec": time.time() - t0
    }
    with open(EXP_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"OOF nMAE Q1 2025 (4D cal + blend): {study.best_value:.4f} vs EXP-054 7.2782")

if __name__ == "__main__":
    main()
