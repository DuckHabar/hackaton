"""
EXP-091 v2 - additive structure cal вместо 4D bins.

Идея: вместо 64-cell cube - additive shifts:
  pred_cal = pred_base + bias_ws_bin[b] + α_eve * is_eve + α_n_rep3 * (n_rep3 & b>=7)
где параметры α - fit через least squares на OOF residuals.

Это сильнее всего похоже на LB mining (additive overlays), но learned from OOF.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import Ridge
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
from common.physics_features import nmae

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-091v2"
EXP_DIR.mkdir(parents=True, exist_ok=True)

BIN_EDGES = [0, 3, 5, 7, 9, 11, 13, 16, 30]
BASES = ["EXP-019", "EXP-020", "EXP-021", "EXP-024"]


def load_base(exp_id):
    base = ROOT / "experiments/active" / exp_id
    oof = pd.read_parquet(base / "oof.parquet")
    val = pd.read_parquet(base / "valid_pred.parquet")
    oof["dt"] = pd.to_datetime(oof["dt"])
    val["dt"] = pd.to_datetime(val["dt"])
    return oof, val


def load_context():
    train = pd.read_parquet(ROOT / "data/processed/train_with_physics.parquet")
    train["dt"] = pd.to_datetime(train["METEOFORECASTHOUR_OPENM_Datetime"])
    valid = pd.read_parquet(ROOT / "data/processed/valid_with_physics.parquet")
    valid["dt"] = pd.to_datetime(valid["METEOFORECASTHOUR_OPENM_Datetime"])
    cols = ["dt", "wind_speed_120m", "Кол-во_ВЭУ_в_ремонте"]
    return train[cols], valid[cols]


def add_strata(df, train_ctx, valid_ctx):
    ctx = pd.concat([train_ctx, valid_ctx], ignore_index=True).drop_duplicates(subset=["dt"])
    df = df.merge(ctx, on="dt", how="left")
    df["ws_120"] = df["wind_speed_120m"]
    df["n_rep"] = df["Кол-во_ВЭУ_в_ремонте"].fillna(0).astype(int)
    df["hour"] = df["dt"].dt.hour
    df["bin_ws"] = np.digitize(df["ws_120"].values, BIN_EDGES[1:-1])
    df["is_day_strict"] = ((df["hour"] >= 6) & (df["hour"] < 16)).astype(int)
    df["is_eve"] = ((df["hour"] >= 16) & (df["hour"] < 20)).astype(int)
    df["is_night_strict"] = ((df["hour"] < 6) | (df["hour"] >= 20)).astype(int)
    df["n_rep3"] = (df["n_rep"] == 3).astype(int)
    df["bin7_n_rep3"] = ((df["bin_ws"] == 7) & (df["n_rep3"] == 1)).astype(int)
    df["bin2_night"] = ((df["bin_ws"] == 2) & (df["is_night_strict"] == 1)).astype(int)
    df["bin3_day"] = ((df["bin_ws"] == 3) & (df["is_day_strict"] == 1)).astype(int)
    return df


def build_design_matrix(df):
    """X: one-hot ws_bin (8) + 4 physics interactions + 1 eve overlay = 13 cols."""
    n = len(df)
    cols = []
    names = []
    for b in range(8):
        cols.append((df["bin_ws"] == b).astype(int).values)
        names.append(f"bin_{b}")
    # Physical interactions (F1, F4)
    cols.append(df["bin2_night"].values); names.append("bin2_night")
    cols.append(df["bin3_day"].values); names.append("bin3_day")
    cols.append(df["bin7_n_rep3"].values); names.append("bin7_n_rep3")
    # Evening overlay (F2)
    cols.append(df["is_eve"].values); names.append("is_eve")
    return np.stack(cols, axis=1).astype(float), names


def fit_additive(cal_df, alpha=10.0):
    X, names = build_design_matrix(cal_df)
    y = (cal_df["y_true"] - cal_df["pred"]).values
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(X, y)
    return reg, names


def apply_additive(df, reg):
    X, _ = build_design_matrix(df)
    bias = reg.predict(X)
    return df["pred"].values + bias


def forward_walking(oof_df, hold_year, base_name):
    cal_mask = oof_df["dt"].dt.month.isin([1,2,3]) & oof_df["dt"].dt.year.isin([2022,2023,2024,2025])
    cal_mask &= oof_df["dt"].dt.year != hold_year
    hold_mask = oof_df["dt"].dt.month.isin([1,2,3]) & (oof_df["dt"].dt.year == hold_year)
    cal_df = oof_df[cal_mask].copy()
    hold_df = oof_df[hold_mask].copy()
    reg, names = fit_additive(cal_df)
    hold_df["pred_add"] = np.clip(apply_additive(hold_df, reg), 0, 90.09)
    # Baseline: flat per-bin cal
    flat_bias = {b: float((cal_df[cal_df["bin_ws"]==b]["y_true"] - cal_df[cal_df["bin_ws"]==b]["pred"]).median()) for b in cal_df["bin_ws"].unique()}
    hold_df["pred_flat"] = np.clip(hold_df["pred"].values + hold_df["bin_ws"].map(flat_bias).fillna(0).values, 0, 90.09)
    return {
        "year": hold_year, "n": len(hold_df),
        "raw": nmae(hold_df["y_true"].values, hold_df["pred"].values),
        "flat": nmae(hold_df["y_true"].values, hold_df["pred_flat"].values),
        "add": nmae(hold_df["y_true"].values, hold_df["pred_add"].values),
        "coefs": dict(zip(names, reg.coef_.tolist()))
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXP-091v2 P1: additive structure cal")
    print("=" * 70)
    train_ctx, valid_ctx = load_context()
    bundle = {}
    for b in BASES:
        oof, val = load_base(b)
        oof = oof.rename(columns={"oof_pred": "pred"})
        val = val.rename(columns={"pred_mw": "pred"})
        oof = add_strata(oof, train_ctx, valid_ctx)
        val = add_strata(val, train_ctx, valid_ctx)
        bundle[b] = {"oof": oof, "val": val}
        print(f"\n--- {b} forward-walking ---")
        print(f"  year | n     | raw   | flat  | add   | add-flat")
        results = []
        for yr in [2022, 2023, 2024, 2025]:
            r = forward_walking(oof, yr, b)
            print(f"  {yr} | {r['n']:5d} | {r['raw']:.4f} | {r['flat']:.4f} | {r['add']:.4f} | {r['add']-r['flat']:+.4f}")
            results.append(r)
        bundle[b]["results"] = results
        mean_delta = np.mean([r["add"] - r["flat"] for r in results])
        print(f"  MEAN add-flat: {mean_delta:+.4f} п.п.")
        # Show coefs (averaged across folds)
        all_coefs = {n: [] for n in results[0]["coefs"].keys()}
        for r in results:
            for n, v in r["coefs"].items():
                all_coefs[n].append(v)
        print(f"  coefs (mean across folds):")
        for n, vs in all_coefs.items():
            print(f"    {n:18s}: {np.mean(vs):+.3f}")

    # Fit additive cal на Q1 2023+2024+2025 (как EXP-054), apply на valid
    print("\n--- Fit additive cal на Q1 2023+2024+2025, apply на valid ---")
    for b in BASES:
        oof = bundle[b]["oof"]
        val = bundle[b]["val"]
        cal_mask = oof["dt"].dt.month.isin([1,2,3]) & oof["dt"].dt.year.isin([2023,2024,2025])
        reg, _ = fit_additive(oof[cal_mask])
        oof["pred_cal"] = np.clip(apply_additive(oof, reg), 0, 90.09)
        val["pred_cal"] = np.clip(apply_additive(val, reg), 0, 90.09)
        q25_mask = (oof["dt"].dt.year == 2025) & oof["dt"].dt.month.isin([1,2,3])
        q1_2025_nmae = nmae(oof.loc[q25_mask, "y_true"].values, oof.loc[q25_mask, "pred_cal"].values)
        print(f"  {b}: Q1 2025 OOF cal nMAE = {q1_2025_nmae:.4f}")

    # Optuna blend Q1 2025 OOF cal
    print("\n--- Optuna blend (400 trials) ---")
    common_dt = None
    for b in BASES:
        oof = bundle[b]["oof"]
        q25 = oof[(oof["dt"].dt.year == 2025) & oof["dt"].dt.month.isin([1,2,3])].sort_values("dt")
        bundle[b]["oof_q25_cal"] = q25
        if common_dt is None:
            common_dt = set(q25["dt"].values)
        else:
            common_dt &= set(q25["dt"].values)
    common_dt = pd.to_datetime(sorted(list(common_dt)))
    cols = []
    y = None
    for b in BASES:
        q25 = bundle[b]["oof_q25_cal"]
        q25 = q25[q25["dt"].isin(common_dt)].sort_values("dt").reset_index(drop=True)
        cols.append(q25["pred_cal"].values)
        if y is None:
            y = q25["y_true"].values
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
    print(f"  best Q1 2025 OOF cal+add nMAE = {study.best_value:.4f}")
    print(f"  weights: {dict(zip(BASES, [round(x,3) for x in best_w]))}")

    val_stack = np.stack([bundle[b]["val"]["pred_cal"].values for b in BASES], axis=1)
    valid_blend = np.clip(val_stack @ best_w, 0, 90.09)
    print(f"\n  valid blend: mean={valid_blend.mean():.3f}, sum={valid_blend.sum():.1f}")

    val = bundle[BASES[0]]["val"].copy().sort_values("dt").reset_index(drop=True)
    val["prediction"] = valid_blend
    sub = val[["prediction"]].iloc[::-1].reset_index(drop=True)
    sub_path = EXP_DIR / "sub_phys_v2.csv"
    sub.to_csv(sub_path, index=False)
    print(f"  saved: {sub_path}")
    print(f"\nTotal time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
