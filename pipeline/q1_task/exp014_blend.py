"""
EXP-014: ensemble blend EXP-007 / EXP-009 / EXP-010 / EXP-012 / EXP-011 (XGB) через Optuna weights.

Strategy: каждый EXP даёт valid_pred.parquet и oof.parquet. Соединяем все OOF (с inner join
по dt), нормализуем веса (sum=1), оптимизируем под минимум CV на OUTER fold (отделяем последние
6 месяцев из OOF train для unbiased weight estimate - это псевдо-holdout). Затем применяем
weights к valid_pred - это и есть submission EXP-014.

LBs:
- EXP-007: 8.0624
- EXP-009: 8.0494
- EXP-010: 8.0307
- EXP-011: 8.2130 (хуже но diversity)
- EXP-012: 8.0061
- Average если всё одинаково: ~8.07
- Blend with optimal weights: ожидаем 7.95-8.00 (gain 0.03-0.06 п.п. от EXP-012 best)
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
EXP_DIR = ROOT / "experiments/active/EXP-014"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Source experiments (use promoted EXP-007 from archive)
SOURCES = [
    ("EXP-007", "/home/duck/wind_hackathon/experiments/archive/promoted/EXP-007", 8.0624),
    ("EXP-009", "/home/duck/wind_hackathon/experiments/active/EXP-009", 8.0494),
    ("EXP-010", "/home/duck/wind_hackathon/experiments/active/EXP-010", 8.0307),
    ("EXP-011", "/home/duck/wind_hackathon/experiments/active/EXP-011", 8.2130),
    ("EXP-012", "/home/duck/wind_hackathon/experiments/active/EXP-012", 8.0061),
]

# Load all
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

# Inner join all OOFs by dt - work with common subset
base_dt = oofs["EXP-012"][["dt", "y_true"]].copy()
for name in [s[0] for s in SOURCES]:
    o = oofs[name][["dt", "oof_pred"]].rename(columns={"oof_pred": f"oof_{name}"})
    base_dt = base_dt.merge(o, on="dt", how="inner")
print(f"\nCommon OOF rows after inner join: {len(base_dt)}")

# Q1 mask + outer holdout (last 6 months from common OOF for unbiased weight tuning)
y_oof = base_dt["y_true"].values
dt_oof = base_dt["dt"]
oof_cols = [f"oof_{s[0]}" for s in SOURCES]
preds_oof = base_dt[oof_cols].values  # (n, k)

# Split: first 70% by time = tuning, last 30% = holdout for weight validation
n = len(base_dt)
tune_mask = np.arange(n) < int(n * 0.7)
hold_mask = ~tune_mask

is_q1_all = dt_oof.dt.month.isin([1, 2, 3]).values
print(f"Tune n={tune_mask.sum()}, holdout n={hold_mask.sum()}, Q1 in holdout: {is_q1_all[hold_mask].sum()}")


def loss_fn(weights, mask):
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0, None)
    if w.sum() == 0:
        return 100
    w = w / w.sum()
    p = preds_oof @ w
    overall = nmae(y_oof[mask], p[mask])
    q1_mask = is_q1_all & mask
    q1 = nmae(y_oof[q1_mask], p[q1_mask]) if q1_mask.sum() > 50 else overall
    return 0.4 * overall + 0.6 * q1


# Baseline: equal weights
n_models = len(SOURCES)
equal_w = np.ones(n_models) / n_models
base_loss = loss_fn(equal_w, hold_mask)
print(f"Equal-weight blend loss (holdout): {base_loss:.4f}")

# Single best
for i, (name, _, lb) in enumerate(SOURCES):
    w_solo = np.zeros(n_models); w_solo[i] = 1
    l = loss_fn(w_solo, hold_mask)
    print(f"  solo {name}: holdout loss {l:.4f}")


def objective(trial):
    w = [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(n_models)]
    return loss_fn(w, tune_mask)


study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=500, show_progress_bar=False)

best_w = np.array([study.best_params[f"w_{i}"] for i in range(n_models)])
best_w_norm = best_w / best_w.sum()
tune_loss = loss_fn(best_w, tune_mask)
hold_loss = loss_fn(best_w, hold_mask)
print(f"\nOptuna best weights (normalized): {dict(zip([s[0] for s in SOURCES], best_w_norm.round(3)))}")
print(f"Tune loss: {tune_loss:.4f}, Holdout loss: {hold_loss:.4f}")

# All data fit (final weights for valid prediction)
def obj_full(trial):
    w = [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(n_models)]
    return loss_fn(w, np.ones(n, dtype=bool))


study_full = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
study_full.optimize(obj_full, n_trials=500, show_progress_bar=False)
full_w = np.array([study_full.best_params[f"w_{i}"] for i in range(n_models)])
full_w_norm = full_w / full_w.sum()
full_loss = loss_fn(full_w, np.ones(n, dtype=bool))
print(f"\nFull-data weights: {dict(zip([s[0] for s in SOURCES], full_w_norm.round(3)))}")
print(f"Full-data loss: {full_loss:.4f}")

# Apply to valid (use full_w - more data -> better generalization)
v_base = valids["EXP-012"][["dt", "n_avail"]].copy()
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

# OOF blend
oof_blend = preds_oof @ full_w_norm
overall_oof = nmae(y_oof, oof_blend)
q1_oof = nmae(y_oof[is_q1_all], oof_blend[is_q1_all])

oof_out = pd.DataFrame({"dt": base_dt["dt"].values, "y_true": y_oof, "oof_pred": oof_blend})
oof_out.to_parquet(EXP_DIR / "oof.parquet")

# Solo CV per source for comparison
print("\nSolo OOF nMAE per source (inner-joined subset):")
for i, (name, _, lb) in enumerate(SOURCES):
    n_ = nmae(y_oof, preds_oof[:, i])
    n_q1 = nmae(y_oof[is_q1_all], preds_oof[is_q1_all, i])
    print(f"  {name}: {n_:.4f} (Q1 {n_q1:.4f}, LB {lb})")
print(f"  BLEND: {overall_oof:.4f} (Q1 {q1_oof:.4f})")

summary = {
    "exp_id": "EXP-014", "name": "blend_5_models_optuna_weights",
    "sources": [{"name": s[0], "lb": s[2]} for s in SOURCES],
    "weights_tune": dict(zip([s[0] for s in SOURCES], best_w_norm.round(4).tolist())),
    "weights_full": dict(zip([s[0] for s in SOURCES], full_w_norm.round(4).tolist())),
    "tune_loss": round(float(tune_loss), 4),
    "holdout_loss": round(float(hold_loss), 4),
    "full_loss": round(float(full_loss), 4),
    "cv_mean_nmae": round(float(overall_oof), 4),
    "cv_q1_only_mean_nmae": round(float(q1_oof), 4),
    "elapsed_sec": round(time.time() - 0.0, 1),
}
(EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print(f"\nSummary: {EXP_DIR / 'summary.json'}")
