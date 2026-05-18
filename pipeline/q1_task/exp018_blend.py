"""
EXP-018: blend EXP-012/015/017 OOF через Optuna с Q1-only objective.

EXP-014 проблема была: objective `0.4*overall + 0.6*Q1` дал 0 веса EXP-012 (best LB).
EXP-018 fix: чистое Q1-only loss (минимизируем только Q1 nMAE).

Sources:
- EXP-012 (LB 8.0061) - baseline long lags
- EXP-015 (LB 7.9982) - baseline + relaxed filter
- EXP-017 (LB 7.6914) - sister NWP stacking (CURRENT BEST)

EXP-017 уже доминирует, blend может дать +0.02-0.05 п.п. если найдёт diversity.
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
EXP_DIR = ROOT / "experiments/active/EXP-018"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Top 3 by LB. EXP-017 dominant, others для diversity.
SOURCES = [
    ("EXP-012", "/home/duck/wind_hackathon/experiments/archive/promoted/EXP-012", 8.0061),
    ("EXP-015", "/home/duck/wind_hackathon/experiments/archive/promoted/EXP-015", 7.9982),
    ("EXP-017", "/home/duck/wind_hackathon/experiments/archive/promoted/EXP-017", 7.6914),
]

oofs = {}
valids = {}
for name, path, lb in SOURCES:
    p = Path(path)
    o = pd.read_parquet(p / "oof.parquet")
    v = pd.read_parquet(p / "valid_pred.parquet")
    o["dt"] = pd.to_datetime(o["dt"])
    v["dt"] = pd.to_datetime(v["dt"])
    oofs[name] = o
    valids[name] = v
    print(f"{name}: oof={len(o)}, valid={len(v)}, LB={lb}")

# Inner join OOFs by dt
base_dt = oofs["EXP-017"][["dt", "y_true"]].copy()
for name in [s[0] for s in SOURCES]:
    o = oofs[name][["dt", "oof_pred"]].rename(columns={"oof_pred": f"oof_{name}"})
    base_dt = base_dt.merge(o, on="dt", how="inner")
print(f"\nCommon OOF rows: {len(base_dt)}")

y_oof = base_dt["y_true"].values
dt_oof = base_dt["dt"]
oof_cols = [f"oof_{s[0]}" for s in SOURCES]
preds_oof = base_dt[oof_cols].values

# 70% tune / 30% holdout
n = len(base_dt)
tune_mask = np.arange(n) < int(n * 0.7)
hold_mask = ~tune_mask

is_q1_all = dt_oof.dt.month.isin([1, 2, 3]).values
print(f"Tune n={tune_mask.sum()}, holdout n={hold_mask.sum()}, Q1 holdout n={(is_q1_all & hold_mask).sum()}")


def loss_q1_only(weights, mask):
    """Q1-only objective."""
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0, None)
    if w.sum() == 0:
        return 100.0
    w = w / w.sum()
    p = preds_oof @ w
    q1_mask = is_q1_all & mask
    if q1_mask.sum() < 50:
        return nmae(y_oof[mask], p[mask])
    return nmae(y_oof[q1_mask], p[q1_mask])


# Baselines
n_models = len(SOURCES)
print("\nBaselines (Q1-only loss on holdout):")
for i, (name, _, lb) in enumerate(SOURCES):
    w = np.zeros(n_models); w[i] = 1
    l = loss_q1_only(w, hold_mask)
    print(f"  solo {name}: {l:.4f} (LB {lb})")
equal_w = np.ones(n_models) / n_models
print(f"  equal weights: {loss_q1_only(equal_w, hold_mask):.4f}")


def objective(trial):
    w = [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(n_models)]
    return loss_q1_only(w, tune_mask)


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=500, show_progress_bar=False)

best_w = np.array([study.best_params[f"w_{i}"] for i in range(n_models)])
best_w_norm = best_w / best_w.sum()
tune_loss = loss_q1_only(best_w, tune_mask)
hold_loss = loss_q1_only(best_w, hold_mask)
print(f"\nOptuna Q1-only weights (tune): {dict(zip([s[0] for s in SOURCES], best_w_norm.round(3)))}")
print(f"  Tune Q1 loss: {tune_loss:.4f}, Holdout Q1 loss: {hold_loss:.4f}")


# Full-data optimization (для финального применения к valid)
def obj_full(trial):
    w = [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(n_models)]
    return loss_q1_only(w, np.ones(n, dtype=bool))


study_full = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study_full.optimize(obj_full, n_trials=500, show_progress_bar=False)
full_w = np.array([study_full.best_params[f"w_{i}"] for i in range(n_models)])
full_w_norm = full_w / full_w.sum()
full_loss = loss_q1_only(full_w, np.ones(n, dtype=bool))
print(f"\nFull-data weights: {dict(zip([s[0] for s in SOURCES], full_w_norm.round(3)))}")
print(f"  Full Q1 loss: {full_loss:.4f}")

# Apply weights to valid (inner-join all valids)
v_base = valids["EXP-017"][["dt", "n_avail"]].copy()
for name in [s[0] for s in SOURCES]:
    vp = valids[name][["dt", "pred_mw"]].rename(columns={"pred_mw": f"pred_{name}"})
    v_base = v_base.merge(vp, on="dt", how="inner")
print(f"\nValid join rows: {len(v_base)}")

pred_cols = [f"pred_{s[0]}" for s in SOURCES]
v_preds = v_base[pred_cols].values
v_blend = v_preds @ full_w_norm
v_blend = np.clip(v_blend, 0.0, v_base["n_avail"].values * TURBINE_RATED_MW)

out = pd.DataFrame({"dt": v_base["dt"].values, "pred_mw": v_blend, "n_avail": v_base["n_avail"].values})
for c in pred_cols:
    out[c] = v_base[c].values
out.to_parquet(EXP_DIR / "valid_pred.parquet")

# OOF blend (для записи)
oof_blend = preds_oof @ full_w_norm
overall_oof = nmae(y_oof, oof_blend)
q1_oof = nmae(y_oof[is_q1_all], oof_blend[is_q1_all])

oof_out = pd.DataFrame({"dt": base_dt["dt"].values, "y_true": y_oof, "oof_pred": oof_blend})
oof_out.to_parquet(EXP_DIR / "oof.parquet")

print(f"\nFinal OOF: overall {overall_oof:.4f}, Q1 {q1_oof:.4f}")
print("Solo OOF Q1 per source (inner-joined subset):")
for i, (name, _, lb) in enumerate(SOURCES):
    n_q1 = nmae(y_oof[is_q1_all], preds_oof[is_q1_all, i])
    print(f"  {name}: {n_q1:.4f} (LB {lb})")

summary = {
    "exp_id": "EXP-018",
    "name": "blend_top3_q1_only_objective",
    "sources": [{"name": s[0], "lb": s[2]} for s in SOURCES],
    "weights_tune": dict(zip([s[0] for s in SOURCES], best_w_norm.round(4).tolist())),
    "weights_full": dict(zip([s[0] for s in SOURCES], full_w_norm.round(4).tolist())),
    "tune_q1_loss": round(float(tune_loss), 4),
    "holdout_q1_loss": round(float(hold_loss), 4),
    "full_q1_loss": round(float(full_loss), 4),
    "cv_mean_nmae": round(float(overall_oof), 4),
    "cv_q1_only_mean_nmae": round(float(q1_oof), 4),
}
(EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print(f"\nSummary: {EXP_DIR / 'summary.json'}")
