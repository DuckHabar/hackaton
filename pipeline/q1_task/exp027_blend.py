"""
EXP-027: blend EXP-021 + EXP-024 (БЕЗ EXP-025 чтобы избежать его LB регрессии).

EXP-021 (solo 5src+dis, LB 7.3264, holdout Q1 7.326)
EXP-024 (blend top4, LB 7.3122 current best, holdout Q1 7.314)

Two diverse strategies, both проверены на LB. Меньше LB risk чем blend с EXP-025.
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
EXP_DIR = ROOT / "experiments/active/EXP-027"
EXP_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ("EXP-021", "/home/duck/wind_hackathon/experiments/archive/promoted/EXP-021", 7.3264),
    ("EXP-024", "/home/duck/wind_hackathon/experiments/archive/promoted/EXP-024", 7.3122),
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

base_dt = oofs["EXP-024"][["dt", "y_true"]].copy()
for name in [s[0] for s in SOURCES]:
    o = oofs[name][["dt", "oof_pred"]].rename(columns={"oof_pred": f"oof_{name}"})
    base_dt = base_dt.merge(o, on="dt", how="inner")
print(f"\nCommon OOF: {len(base_dt)}")

y_oof = base_dt["y_true"].values
preds_oof = base_dt[[f"oof_{s[0]}" for s in SOURCES]].values
n = len(base_dt)
is_q1_all = base_dt["dt"].dt.month.isin([1, 2, 3]).values

# Optuna search для 50/50, 40/60, 30/70 etc
best_loss = 100; best_w = None
for w0 in np.arange(0.0, 1.01, 0.05):
    w1 = 1 - w0
    p = preds_oof @ np.array([w0, w1])
    loss = nmae(y_oof[is_q1_all], p[is_q1_all])
    if loss < best_loss:
        best_loss = loss
        best_w = np.array([w0, w1])

print(f"\nBest weights (Q1 OOF): EXP-021={best_w[0]:.2f}, EXP-024={best_w[1]:.2f}, Q1 loss {best_loss:.4f}")

# Apply
v_base = valids["EXP-024"][["dt", "n_avail"]].copy()
for name in [s[0] for s in SOURCES]:
    vp = valids[name][["dt", "pred_mw"]].rename(columns={"pred_mw": f"pred_{name}"})
    v_base = v_base.merge(vp, on="dt", how="inner")
pred_cols = [f"pred_{s[0]}" for s in SOURCES]
v_blend = v_base[pred_cols].values @ best_w
v_blend = np.clip(v_blend, 0.0, v_base["n_avail"].values * TURBINE_RATED_MW)

pd.DataFrame({"dt": v_base["dt"].values, "pred_mw": v_blend,
              "n_avail": v_base["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")

oof_blend = preds_oof @ best_w
overall_oof = nmae(y_oof, oof_blend)
q1_oof = nmae(y_oof[is_q1_all], oof_blend[is_q1_all])
pd.DataFrame({"dt": base_dt["dt"].values, "y_true": y_oof, "oof_pred": oof_blend}).to_parquet(EXP_DIR / "oof.parquet")

summary = {
    "exp_id": "EXP-027", "name": "blend_exp021_exp024_simple_grid",
    "sources": [{"name": s[0], "lb": s[2]} for s in SOURCES],
    "weights": {SOURCES[0][0]: float(best_w[0]), SOURCES[1][0]: float(best_w[1])},
    "cv_mean_nmae": round(float(overall_oof), 4),
    "cv_q1_only_mean_nmae": round(float(q1_oof), 4),
}
(EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print(f"Summary: {EXP_DIR / 'summary.json'}")
