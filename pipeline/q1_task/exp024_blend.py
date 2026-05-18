"""
EXP-024: Optuna blend top 4 LB (EXP-017/019/020/021) с Q1-only objective.

LBs:
- EXP-017: 7.6914
- EXP-019: 7.4827
- EXP-020: 7.3365
- EXP-021: 7.3264 (current best)

Все relaxed filter, same OOF size. Inner-join даст полные 30790 rows.

Гипотеза: blend забирает diversity, может дать +0.01-0.03 п.п. над EXP-021.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-024"
EXP_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ("EXP-017", "/home/duck/wind_hackathon/experiments/archive/promoted/EXP-017", 7.6914),
    ("EXP-019", "/home/duck/wind_hackathon/experiments/archive/promoted/EXP-019", 7.4827),
    ("EXP-020", "/home/duck/wind_hackathon/experiments/archive/promoted/EXP-020", 7.3365),
    ("EXP-021", "/home/duck/wind_hackathon/experiments/archive/promoted/EXP-021", 7.3264),
]

oofs = {}; valids = {}
for name, path, lb in SOURCES:
    p = Path(path)
    o = pd.read_parquet(p / "oof.parquet")
    v = pd.read_parquet(p / "valid_pred.parquet")
    o["dt"] = pd.to_datetime(o["dt"])
    v["dt"] = pd.to_datetime(v["dt"])
    oofs[name] = o; valids[name] = v
    print(f"{name}: oof={len(o)}, valid={len(v)}, LB={lb}")

base_dt = oofs["EXP-021"][["dt", "y_true"]].copy()
for name in [s[0] for s in SOURCES]:
    o = oofs[name][["dt", "oof_pred"]].rename(columns={"oof_pred": f"oof_{name}"})
    base_dt = base_dt.merge(o, on="dt", how="inner")
print(f"\nCommon OOF: {len(base_dt)} rows")

y_oof = base_dt["y_true"].values
dt_oof = base_dt["dt"]
oof_cols = [f"oof_{s[0]}" for s in SOURCES]
preds_oof = base_dt[oof_cols].values

n = len(base_dt)
tune_mask = np.arange(n) < int(n * 0.7)
hold_mask = ~tune_mask
is_q1_all = dt_oof.dt.month.isin([1, 2, 3]).values
print(f"Tune n={tune_mask.sum()}, holdout n={hold_mask.sum()}, Q1 holdout n={(is_q1_all & hold_mask).sum()}")


def loss_q1(weights, mask):
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0, None)
    if w.sum() == 0: return 100.0
    w = w / w.sum()
    p = preds_oof @ w
    q1_mask = is_q1_all & mask
    if q1_mask.sum() < 50:
        return nmae(y_oof[mask], p[mask])
    return nmae(y_oof[q1_mask], p[q1_mask])


n_models = len(SOURCES)
print("\nSolo Q1-only loss (holdout):")
for i, (name, _, lb) in enumerate(SOURCES):
    w = np.zeros(n_models); w[i] = 1
    print(f"  {name}: {loss_q1(w, hold_mask):.4f} (LB {lb})")


def objective(trial):
    w = [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(n_models)]
    return loss_q1(w, tune_mask)


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=800, show_progress_bar=False)
best_w = np.array([study.best_params[f"w_{i}"] for i in range(n_models)])
best_w_norm = best_w / best_w.sum()
hold_loss = loss_q1(best_w, hold_mask)
print(f"\nTuned weights (holdout {hold_loss:.4f}): {dict(zip([s[0] for s in SOURCES], best_w_norm.round(3)))}")


def obj_full(trial):
    w = [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(n_models)]
    return loss_q1(w, np.ones(n, dtype=bool))


study_full = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study_full.optimize(obj_full, n_trials=800, show_progress_bar=False)
full_w = np.array([study_full.best_params[f"w_{i}"] for i in range(n_models)])
full_w_norm = full_w / full_w.sum()
print(f"Full-data weights: {dict(zip([s[0] for s in SOURCES], full_w_norm.round(3)))}")
print(f"Full Q1 loss: {loss_q1(full_w, np.ones(n, dtype=bool)):.4f}")

# Apply to valid
v_base = valids["EXP-021"][["dt", "n_avail"]].copy()
for name in [s[0] for s in SOURCES]:
    vp = valids[name][["dt", "pred_mw"]].rename(columns={"pred_mw": f"pred_{name}"})
    v_base = v_base.merge(vp, on="dt", how="inner")
print(f"\nValid join: {len(v_base)} rows")

pred_cols = [f"pred_{s[0]}" for s in SOURCES]
v_preds = v_base[pred_cols].values
v_blend = v_preds @ full_w_norm
v_blend = np.clip(v_blend, 0.0, v_base["n_avail"].values * TURBINE_RATED_MW)

out = pd.DataFrame({"dt": v_base["dt"].values, "pred_mw": v_blend, "n_avail": v_base["n_avail"].values})
out.to_parquet(EXP_DIR / "valid_pred.parquet")

oof_blend = preds_oof @ full_w_norm
overall_oof = nmae(y_oof, oof_blend)
q1_oof = nmae(y_oof[is_q1_all], oof_blend[is_q1_all])
pd.DataFrame({"dt": base_dt["dt"].values, "y_true": y_oof, "oof_pred": oof_blend}).to_parquet(EXP_DIR / "oof.parquet")

print(f"\nFinal OOF: overall {overall_oof:.4f}, Q1 {q1_oof:.4f}")

summary = {
    "exp_id": "EXP-024", "name": "blend_top4_q1_only",
    "sources": [{"name": s[0], "lb": s[2]} for s in SOURCES],
    "weights_full": dict(zip([s[0] for s in SOURCES], full_w_norm.round(4).tolist())),
    "weights_tune": dict(zip([s[0] for s in SOURCES], best_w_norm.round(4).tolist())),
    "holdout_q1_loss": round(float(hold_loss), 4),
    "cv_mean_nmae": round(float(overall_oof), 4),
    "cv_q1_only_mean_nmae": round(float(q1_oof), 4),
}
(EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print(f"\nSummary: {EXP_DIR / 'summary.json'}")
