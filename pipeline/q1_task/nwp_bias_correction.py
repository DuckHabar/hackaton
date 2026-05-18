"""
EXP-003: NWP bias correction.

Метод:
1. На train: для строк с активной частью power curve (ws ∈ [4, 11]) решаем inverse
   power curve: для каждой строки находим ws_real такое что P_curve(ws_real) ≈ p_per_turbine.
2. bias = ws_real - ws_120_NWP. Группируем по (sector_8, month) -> mean/median/std.
3. Применяем как глобальный bias_table к train + valid -> новая фича ws_corrected_120.
4. Запускаем lgbm_q50 c расширенным feature set.

Глобальная (не fold-aware) калибровка - компромисс. Fold-aware вариант даст ту же
оценку с меньшей дисперсией.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

sys.path.insert(0, str(Path("~/wind_hackathon/pipeline").expanduser()))
from common.physics_features import (
    TARGET_COL, REPAIR_COL, DT_COL,
    TURBINE_RATED_MW, P_INST_MW,
    smooth_power_curve_ti,
    predict_p_physical_total,
    nmae,
)
# Также импортируем feature engineering из lgbm_q50 (для reuse)
sys.path.insert(0, str(Path("~/wind_hackathon/pipeline/q1_task").expanduser()))
from lgbm_q50 import (
    prepare_features,
    get_feature_cols,
    fit_cv,
    compute_segments,
    predict_valid,
    HPS, N_BOOST, EARLY_STOP, SECTOR_NAMES,
)

ROOT = Path("~/wind_hackathon").expanduser()
EXP_DIR = ROOT / "experiments/active/EXP-003"
PRED_OOF = EXP_DIR / "oof.parquet"
PRED_VALID = EXP_DIR / "valid_pred.parquet"
SUMMARY = EXP_DIR / "summary.json"
BIAS_TABLE = EXP_DIR / "bias_table.json"


def build_inverse_curve(pc_smooth):
    """
    Из smoothed power curve строим обратную функцию ws_real(p_per_turbine_MW)
    в monotonic active region [cut_in_eff, rated_eff].
    Возвращает (ws_grid, p_grid_mw) на активном участке.
    """
    ws = pc_smooth["wind_speed"].values
    p_mw = pc_smooth["value"].values / 1e6  # МВт per turbine

    # Активная зона: строго возрастающая часть
    rising_mask = np.concatenate([[True], np.diff(p_mw) > 1e-6])
    plateau_start = np.argmax(p_mw >= p_mw.max() * 0.99)  # ≥99% rated -> plateau
    cut_in_idx = np.argmax(p_mw > 0.01)
    active = slice(cut_in_idx, plateau_start + 1)
    return ws[active], p_mw[active]


def inverse_power_curve(p_per_turb_mw, ws_active, p_active):
    """Инвертируем: дано p (МВт per turbine), найти ws на активной части кривой.
    np.interp требует x_sorted: y_active монотонно возрастает в active region."""
    p_clipped = np.clip(p_per_turb_mw, p_active.min(), p_active.max())
    return np.interp(p_clipped, p_active, ws_active)


def estimate_bias(train, pc_smooth, ws_lo=4.0, ws_hi=11.0):
    """
    Возвращает bias_table[(sector_int, month)] = {'mean': ..., 'median': ..., 'std': ..., 'n': ...}
    Использует только строки с ws_120 ∈ [ws_lo, ws_hi] (cubic region) и p_per_turbine > 0.1 МВт.
    """
    ws_active, p_active = build_inverse_curve(pc_smooth)
    print(f"Active curve range: ws [{ws_active.min():.2f}, {ws_active.max():.2f}], "
          f"p [{p_active.min():.3f}, {p_active.max():.3f}] МВт/turbine")

    df = train.copy()
    df = df[df["n_avail"] > 0]
    df["p_per_turb"] = df[TARGET_COL] / df["n_avail"]
    df["p_per_turb"] = df["p_per_turb"].clip(lower=0.0, upper=TURBINE_RATED_MW * 1.05)

    # Только in-active-region строки
    mask = (
        (df["wind_speed_120m"] >= ws_lo) & (df["wind_speed_120m"] <= ws_hi)
        & (df["p_per_turb"] >= p_active.min())
        & (df["p_per_turb"] <= p_active.max() * 0.98)  # не на plateau
    )
    df = df[mask].copy()
    df["ws_real_120"] = inverse_power_curve(df["p_per_turb"].values, ws_active, p_active)
    df["bias_120"] = df["ws_real_120"] - df["wind_speed_120m"]
    print(f"Rows for bias estim: {len(df)} (из {len(train)} train)")
    print(f"Overall bias: mean={df['bias_120'].mean():+.3f}, median={df['bias_120'].median():+.3f}, "
          f"std={df['bias_120'].std():.3f}")

    # Per (sector_8, month) group
    bias_table = {}
    for (s, m), g in df.groupby(["sector_8", "month"], observed=True):
        if len(g) < 30:
            continue
        bias_table[f"{int(s)}_{int(m)}"] = {
            "mean": float(g["bias_120"].mean()),
            "median": float(g["bias_120"].median()),
            "std": float(g["bias_120"].std()),
            "n": int(len(g)),
        }
    # Глобальные fallbacks (без sector / без month)
    bias_table["_global"] = {
        "mean": float(df["bias_120"].mean()),
        "median": float(df["bias_120"].median()),
        "std": float(df["bias_120"].std()),
        "n": int(len(df)),
    }
    return bias_table, df["bias_120"]


def apply_bias(df, bias_table, mode="mean"):
    """Применить bias correction к df. Добавляет 'ws_corrected_120' и 'bias_applied'."""
    global_b = bias_table["_global"][mode]
    biases = np.full(len(df), global_b, dtype=np.float64)
    keys = (df["sector_8"].astype(str) + "_" + df["month"].astype(str)).values
    for i, k in enumerate(keys):
        if k in bias_table:
            biases[i] = bias_table[k][mode]
    df["bias_applied"] = biases
    df["ws_corrected_120"] = df["wind_speed_120m"] + biases
    # Также density correction на corrected ws
    df["ws_corrected_120_corr"] = df["ws_corrected_120"] * (df["rho_air"] / 1.225) ** (1.0 / 3.0)
    return df


def add_p_phys_corrected(df, pc_smooth):
    """Пересчитать p_phys с использованием ws_corrected_120 (density corrected)."""
    p_per_turb_kw = np.interp(
        df["ws_corrected_120_corr"].values,
        pc_smooth["wind_speed"].values,
        pc_smooth["value"].values,
        left=0.0, right=0.0,
    ) / 1000.0
    p_per_turb_mw = p_per_turb_kw / 1000.0
    p_total_mw = p_per_turb_mw * df["n_avail"].values
    cap = df["n_avail"].values * TURBINE_RATED_MW
    df["p_phys_corrected"] = np.minimum(p_total_mw, cap)
    return df


def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("[1/4] Загрузка через prepare_features() (включает full feature pipeline)...")
    train_f, valid_f = prepare_features()
    pc_smooth = smooth_power_curve_ti(ti=0.20, rated_speed=10.0)

    print("[2/4] Bias estimation на train...")
    bias_table, all_bias = estimate_bias(train_f, pc_smooth)
    BIAS_TABLE.write_text(json.dumps(bias_table, ensure_ascii=False, indent=2))
    print(f"Bias table saved: {BIAS_TABLE}  ({len(bias_table)} keys incl. _global)")

    print("[3/4] Apply bias correction (mean mode) + recompute p_phys_corrected...")
    train_f = apply_bias(train_f, bias_table, mode="mean")
    train_f = add_p_phys_corrected(train_f, pc_smooth)
    valid_f = apply_bias(valid_f, bias_table, mode="mean")
    valid_f = add_p_phys_corrected(valid_f, pc_smooth)

    # Sanity: nMAE physics baseline только с corrected ws + cap
    phys_nmae = nmae(train_f[TARGET_COL].values, train_f["p_phys_corrected"].values)
    print(f"\nPhysics baseline с corrected ws: {phys_nmae:.3f}% (для сравнения EXP-001b = 12.092%)")

    feature_cols = get_feature_cols(train_f)
    print(f"\nFeature count: {len(feature_cols)} (включая ws_corrected_120, bias_applied, p_phys_corrected)")

    # Fit cv direct mode (best из EXP-002)
    print("\n[4/4] LightGBM CV (direct mode, расширенные фичи)...")
    train_clean = train_f.dropna(subset=[TARGET_COL]).copy()
    for c in feature_cols:
        if train_clean[c].isna().any():
            med = train_clean[c].median()
            train_clean[c] = train_clean[c].fillna(med)
            valid_f[c] = valid_f[c].fillna(med)

    result = fit_cv(train_clean, feature_cols, mode="direct", n_splits=12)
    cv_mean = float(np.mean(result["fold_nmae"]))
    cv_std = float(np.std(result["fold_nmae"]))
    cv_q1 = float(np.mean(result["fold_nmae_q1_only"])) if result["fold_nmae_q1_only"] else None
    cv_q1_std = float(np.std(result["fold_nmae_q1_only"])) if result["fold_nmae_q1_only"] else None

    seg = compute_segments(train_clean[TARGET_COL].values, result["oof"], train_clean)
    imp = sorted(zip(feature_cols, result["feat_importance"]), key=lambda kv: -kv[1])
    top_imp = [{"feat": f, "gain": round(float(g), 1)} for f, g in imp[:15]]

    summary = {
        "exp_id": "EXP-003",
        "name": "lgbm_q50_with_nwp_bias_corr",
        "n_features": len(feature_cols),
        "physics_baseline_with_corrected_ws_nmae": round(phys_nmae, 4),
        "cv_mean_nmae": round(cv_mean, 4),
        "cv_std_nmae": round(cv_std, 4),
        "cv_q1_only_mean_nmae": round(cv_q1, 4) if cv_q1 is not None else None,
        "cv_q1_only_std_nmae": round(cv_q1_std, 4) if cv_q1_std is not None else None,
        "fold_nmae": [round(x, 3) for x in result["fold_nmae"]],
        "fold_nmae_q1_only": [round(x, 3) for x in result["fold_nmae_q1_only"]],
        "fold_best_iter": result["fold_best_iter"],
        "segments": seg,
        "top15_importance": top_imp,
        "bias_stats": {
            "global_mean_bias": float(all_bias.mean()),
            "global_median_bias": float(all_bias.median()),
            "global_std_bias": float(all_bias.std()),
            "n_keys_per_sector_month": len(bias_table) - 1,
        },
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    # Valid pred
    valid_pred = predict_valid(valid_f, result["models"], feature_cols, "direct")
    valid_out = pd.DataFrame({
        "dt": valid_f["dt"].values,
        "pred_mw": valid_pred,
        "p_phys": valid_f["p_phys"].values,
        "p_phys_corrected": valid_f["p_phys_corrected"].values,
        "n_avail": valid_f["n_avail"].values,
    })
    valid_out.to_parquet(PRED_VALID, index=False)

    oof_out = pd.DataFrame({
        "dt": train_clean["dt"].values,
        "y_true": train_clean[TARGET_COL].values,
        "oof_pred": result["oof"],
        "p_phys_corrected": train_clean["p_phys_corrected"].values,
    })
    oof_out.to_parquet(PRED_OOF, index=False)

    print()
    print(f"=== EXP-003 DONE  CV nMAE = {cv_mean:.3f}% ± {cv_std:.3f}% ===")
    print(f"Q1-only: {cv_q1:.3f}% ± {cv_q1_std:.3f}%")
    print(f"Improvement vs EXP-002 (9.311): {9.311 - cv_mean:+.3f} п.п.")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
