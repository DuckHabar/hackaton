"""
EXP-005: lgbm_q50 с best Optuna params (из EXP-004 study).
Использует bias correction (EXP-003 setup) + tuned HPs.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import (
    TARGET_COL, TURBINE_RATED_MW, nmae, smooth_power_curve_ti,
)
from lgbm_q50 import prepare_features, get_feature_cols, fit_cv, compute_segments, predict_valid
from nwp_bias_correction import apply_bias, add_p_phys_corrected, estimate_bias

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-005"
EXP_DIR.mkdir(parents=True, exist_ok=True)

study = optuna.load_study(
    study_name="exp004_lgbm_q1",
    storage="sqlite:////home/duck/wind_hackathon/experiments/active/EXP-004/optuna.db",
)
best_params = study.best_params.copy()
print(f"Loaded best params (score={study.best_value:.4f}):")
print(json.dumps(best_params, indent=2))

# Apply бid correction (EXP-003 setup)
print("\n[1/3] Prepare features + bias correction...")
train_f, valid_f = prepare_features()
pc_smooth = smooth_power_curve_ti(ti=0.20, rated_speed=10.0)
bias_table, _ = estimate_bias(train_f, pc_smooth)
train_f = apply_bias(train_f, bias_table, mode="mean")
train_f = add_p_phys_corrected(train_f, pc_smooth)
valid_f = apply_bias(valid_f, bias_table, mode="mean")
valid_f = add_p_phys_corrected(valid_f, pc_smooth)

feature_cols = get_feature_cols(train_f)
train_clean = train_f.dropna(subset=[TARGET_COL]).copy()
for c in feature_cols:
    if train_clean[c].isna().any():
        med = train_clean[c].median()
        train_clean[c] = train_clean[c].fillna(med)
        valid_f[c] = valid_f[c].fillna(med)

# Patch HPS в модуле lgbm_q50, чтобы fit_cv использовал tuned HPs
import lgbm_q50 as lgmod
sw_q1_factor = best_params.pop("sw_q1_factor", 2.0)
n_boost = best_params.pop("n_boost", 2000)
lgmod.HPS.update({k: v for k, v in best_params.items() if k != "sw_q1_factor"})
lgmod.N_BOOST = n_boost
# Override sample_weight внутри fit_cv через monkey-patch
import numpy as _np
orig_fit_cv = lgmod.fit_cv


def fit_cv_with_factor(train_clean, feature_cols, mode="direct", n_splits=12):
    # Re-use original code but with sw_q1_factor.
    # Trick: temporarily replace sample_weight constant in original fit_cv body.
    # Simpler: rewrite fit_cv body here with sw_q1_factor parametric.
    from sklearn.model_selection import GroupKFold
    import lightgbm as lgb
    from common.physics_features import nmae as _nmae

    X = train_clean[feature_cols].values
    n_avail = train_clean["n_avail"].values
    p_phys = train_clean["p_phys"].values
    y_raw = train_clean[TARGET_COL].values
    y = y_raw  # mode='direct'
    sw = _np.where(train_clean["dt"].dt.month.isin([1, 2, 3]).values, sw_q1_factor, 1.0)

    groups = train_clean["month_key"].values
    gkf = GroupKFold(n_splits=n_splits)
    oof = _np.zeros(len(train_clean), dtype=_np.float64)
    fold_nmae = []
    fold_q1 = []
    feat_imp = _np.zeros(len(feature_cols))
    models = []
    cat_idx = [feature_cols.index("sector_8")] if "sector_8" in feature_cols else "auto"
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        ds_tr = lgb.Dataset(X[tr], label=y[tr], weight=sw[tr], categorical_feature=cat_idx)
        ds_te = lgb.Dataset(X[te], label=y[te], weight=sw[te], reference=ds_tr, categorical_feature=cat_idx)
        m = lgb.train(lgmod.HPS, ds_tr, num_boost_round=lgmod.N_BOOST,
                      valid_sets=[ds_te],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred = m.predict(X[te], num_iteration=m.best_iteration)
        pred_mw = _np.clip(pred, 0.0, n_avail[te] * TURBINE_RATED_MW)
        oof[te] = pred_mw
        fold_nmae.append(_nmae(y_raw[te], pred_mw))
        is_q1 = train_clean["dt"].iloc[te].dt.month.isin([1, 2, 3]).values
        if is_q1.sum() > 50:
            fold_q1.append(_nmae(y_raw[te][is_q1], pred_mw[is_q1]))
        feat_imp += m.feature_importance("gain")
        models.append(m)
        print(f"  fold {fold:02d}: nMAE={fold_nmae[-1]:.3f}%  iter={m.best_iteration}")
    return {
        "oof": oof, "fold_nmae": fold_nmae, "fold_nmae_q1_only": fold_q1,
        "feat_importance": feat_imp / n_splits, "models": models,
        "fold_best_iter": [m.best_iteration for m in models],
    }


print("\n[2/3] CV with tuned HPs...")
t0 = time.time()
result = fit_cv_with_factor(train_clean, feature_cols, mode="direct", n_splits=12)
cv_mean = float(_np.mean(result["fold_nmae"]))
cv_std = float(_np.std(result["fold_nmae"]))
cv_q1 = float(_np.mean(result["fold_nmae_q1_only"]))
cv_q1_std = float(_np.std(result["fold_nmae_q1_only"]))
seg = compute_segments(train_clean[TARGET_COL].values, result["oof"], train_clean)

print(f"\n=== EXP-005 CV mean = {cv_mean:.4f}% ± {cv_std:.4f}%  Q1-only = {cv_q1:.4f}% ===")
print(f"vs EXP-003 (9.2828 / 8.7644): {9.2828 - cv_mean:+.4f} / {8.7644 - cv_q1:+.4f}")

print("\n[3/3] Valid predict + save artifacts...")
valid_pred = predict_valid(valid_f, result["models"], feature_cols, "direct")
valid_out = pd.DataFrame({
    "dt": valid_f["dt"].values,
    "pred_mw": valid_pred,
    "p_phys": valid_f["p_phys"].values,
    "p_phys_corrected": valid_f["p_phys_corrected"].values,
    "n_avail": valid_f["n_avail"].values,
})
valid_out.to_parquet(EXP_DIR / "valid_pred.parquet", index=False)

oof_out = pd.DataFrame({
    "dt": train_clean["dt"].values,
    "y_true": train_clean[TARGET_COL].values,
    "oof_pred": result["oof"],
})
oof_out.to_parquet(EXP_DIR / "oof.parquet", index=False)

imp = sorted(zip(feature_cols, result["feat_importance"]), key=lambda kv: -kv[1])[:15]

summary = {
    "exp_id": "EXP-005",
    "name": "lgbm_q50_optuna_tuned",
    "based_on_optuna_study": "exp004_lgbm_q1",
    "best_optuna_score": study.best_value,
    "tuned_hps": lgmod.HPS,
    "sw_q1_factor": sw_q1_factor,
    "n_boost": lgmod.N_BOOST,
    "cv_mean_nmae": round(cv_mean, 4),
    "cv_std_nmae": round(cv_std, 4),
    "cv_q1_only_mean_nmae": round(cv_q1, 4),
    "cv_q1_only_std_nmae": round(cv_q1_std, 4),
    "fold_nmae": [round(x, 3) for x in result["fold_nmae"]],
    "fold_nmae_q1_only": [round(x, 3) for x in result["fold_nmae_q1_only"]],
    "fold_best_iter": result["fold_best_iter"],
    "segments": seg,
    "top15_importance": [{"feat": f, "gain": round(float(g), 1)} for f, g in imp],
    "elapsed_sec": round(time.time() - t0, 1),
}
(EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print(f"Summary: {EXP_DIR / 'summary.json'}")
print(f"Time: {summary['elapsed_sec']:.1f}s")
