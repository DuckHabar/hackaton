"""
EXP-032: blend EXP-019 + EXP-020 + EXP-021 + EXP-024 (БЕЗ EXP-025 noise) Optuna Q1-only.

EXP-024 уже blend top4 (EXP-017/019/020/021) с {021=0.571, 020=0.338, 019=0.091, 017=0}.
Здесь даём Optuna 4D пространство [019, 020, 021, 024] чтобы he мог докрутить.
Если EXP-024=1.0 - значит ничего нового. Если иначе - есть room для tuning.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import optuna

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-032"
EXP_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ("EXP-019", ROOT / "experiments/archive/promoted/EXP-019", 7.4827),
    ("EXP-020", ROOT / "experiments/archive/promoted/EXP-020", 7.3365),
    ("EXP-021", ROOT / "experiments/archive/promoted/EXP-021", 7.3264),
    ("EXP-024", ROOT / "experiments/archive/promoted/EXP-024", 7.3122),
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

base_dt = oofs["EXP-024"][["dt", "y_true"]].copy()
for name in [s[0] for s in SOURCES]:
    o = oofs[name][["dt", "oof_pred"]].rename(columns={"oof_pred": f"oof_{name}"})
    base_dt = base_dt.merge(o, on="dt", how="inner")
print(f"\nCommon OOF: {len(base_dt)}")

y_oof = base_dt["y_true"].values
preds_oof = base_dt[[f"oof_{s[0]}" for s in SOURCES]].values
is_q1_all = base_dt["dt"].dt.month.isin([1, 2, 3]).values
print(f"Q1 OOF rows: {is_q1_all.sum()}")


def objective(trial):
    raw_w = np.array([trial.suggest_float(f"w_{s[0]}", 0.0, 1.0) for s in SOURCES])
    s = raw_w.sum()
    if s < 1e-6:
        return 100.0
    w = raw_w / s
    p = preds_oof @ w
    return nmae(y_oof[is_q1_all], p[is_q1_all])


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=500, show_progress_bar=False)

best = study.best_params
raw_w = np.array([best[f"w_{s[0]}"] for s in SOURCES])
w = raw_w / raw_w.sum()
print(f"\nBest weights (Q1):")
for (name, _, _), wi in zip(SOURCES, w):
    print(f"  {name}: {wi:.4f}")
print(f"Best Q1 loss: {study.best_value:.4f}")

oof_blend = preds_oof @ w
overall_oof = nmae(y_oof, oof_blend)
q1_oof = nmae(y_oof[is_q1_all], oof_blend[is_q1_all])
print(f"Final OOF: overall {overall_oof:.4f}, Q1 {q1_oof:.4f}")

# Apply to valid
v_base = valids["EXP-024"][["dt", "n_avail"]].copy()
for name in [s[0] for s in SOURCES]:
    vp = valids[name][["dt", "pred_mw"]].rename(columns={"pred_mw": f"pred_{name}"})
    v_base = v_base.merge(vp, on="dt", how="inner")
print(f"Common valid: {len(v_base)}")

pred_cols = [f"pred_{s[0]}" for s in SOURCES]
v_blend = v_base[pred_cols].values @ w
v_blend = np.clip(v_blend, 0.0, v_base["n_avail"].values * TURBINE_RATED_MW)

pd.DataFrame({"dt": v_base["dt"].values, "pred_mw": v_blend,
              "n_avail": v_base["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")
pd.DataFrame({"dt": base_dt["dt"].values, "y_true": y_oof, "oof_pred": oof_blend}).to_parquet(EXP_DIR / "oof.parquet")

summary = {
    "exp_id": "EXP-032", "name": "blend_no025_q1only_4src",
    "sources": [{"name": s[0], "lb": s[2]} for s in SOURCES],
    "weights": {s[0]: float(wi) for (s, wi) in zip(SOURCES, w)},
    "best_trial_value_q1": round(float(study.best_value), 4),
    "cv_mean_nmae": round(float(overall_oof), 4),
    "cv_q1_only_mean_nmae": round(float(q1_oof), 4),
}
(EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print(f"Summary: {EXP_DIR / 'summary.json'}")
