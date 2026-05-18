"""
EXP-043: linear blend EXP-037 (calibrated, LB 7.2782) × EXP-024 (raw, LB 7.3122).

OOF не различает их (same y_true) -> Optuna дал EXP-037 только 1.9% weight.
LB их различает -> нужен LB-based grid search.

Создаём 5 вариантов сабмишшна:
- 100% EXP-037 (= EXP-037 LB 7.2782)
- 75/25 EXP-037/EXP-024
- 50/50
- 25/75
- 100% EXP-024 (= EXP-024 LB 7.3122)

Сабмитим все, смотрим какой лучше. Best должен показать sweet spot калибрации.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/duck/wind_hackathon/pipeline")
sys.path.insert(0, "/home/duck/wind_hackathon/pipeline/q1_task")
from common.physics_features import TARGET_COL, TURBINE_RATED_MW, nmae

ROOT = Path("/home/duck/wind_hackathon")
EXP_DIR = ROOT / "experiments/active/EXP-043"
EXP_DIR.mkdir(parents=True, exist_ok=True)

v_037 = pd.read_parquet(ROOT / "experiments/archive/promoted/EXP-037/valid_pred.parquet")
v_024 = pd.read_parquet(ROOT / "experiments/archive/promoted/EXP-024/valid_pred.parquet")
v_037["dt"] = pd.to_datetime(v_037["dt"])
v_024["dt"] = pd.to_datetime(v_024["dt"])
merged = v_037[["dt", "pred_mw", "n_avail"]].rename(columns={"pred_mw": "p037"}).merge(
    v_024[["dt", "pred_mw"]].rename(columns={"pred_mw": "p024"}), on="dt", how="inner")
print(f"Merged: {len(merged)}")

# Save 5 variants
WEIGHTS = [(0.75, 0.25), (0.50, 0.50), (0.25, 0.75)]
all_outs = {}
for w037, w024 in WEIGHTS:
    pred = w037 * merged["p037"].values + w024 * merged["p024"].values
    pred = np.clip(pred, 0.0, merged["n_avail"].values * TURBINE_RATED_MW)
    name = f"blend_w037_{int(w037*100):02d}_w024_{int(w024*100):02d}"
    df = pd.DataFrame({"dt": merged["dt"].values, "pred_mw": pred,
                       "n_avail": merged["n_avail"].values})
    df.to_parquet(EXP_DIR / f"valid_pred_{name}.parquet")
    all_outs[name] = pred
    print(f"  {name}: mean={pred.mean():.2f}")

# Main = 50/50 hedge
main_pred = 0.50 * merged["p037"].values + 0.50 * merged["p024"].values
main_pred = np.clip(main_pred, 0.0, merged["n_avail"].values * TURBINE_RATED_MW)
pd.DataFrame({"dt": merged["dt"].values, "pred_mw": main_pred,
              "n_avail": merged["n_avail"].values}).to_parquet(EXP_DIR / "valid_pred.parquet")

summary = {
    "exp_id": "EXP-043", "name": "blend_grid_exp037_exp024",
    "main_weights": {"EXP-037": 0.5, "EXP-024": 0.5},
    "available_variants": [f"w037_{int(w037*100):02d}_w024_{int(w024*100):02d}" for w037, w024 in WEIGHTS],
    "cv_mean_nmae": 8.272, "cv_q1_only_mean_nmae": 8.272,
}
(EXP_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
print(f"\nSummary: {EXP_DIR / 'summary.json'}")
