"""Submit EXP-044 variants + run+submit EXP-045."""
import sys, subprocess
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/duck/wind_hackathon")

# EXP-044 has 5 variants
SUBMITS = [
    ("EXP-044", "valid_pred_A_4bins.parquet", "sub_conf_4bins.csv"),
    ("EXP-044", "valid_pred_B_6bins.parquet", "sub_conf_6bins.csv"),
    ("EXP-044", "valid_pred_C_q8bins.parquet", "sub_conf_q8bins.csv"),
    ("EXP-044", "valid_pred_D_q6bins.parquet", "sub_conf_q6bins.csv"),
    ("EXP-044", "valid_pred_E_8bins_037.parquet", "sub_conf_8bins_repl.csv"),
]

for exp, parquet_name, csv_name in SUBMITS:
    p = ROOT / "experiments/active" / exp / parquet_name
    if not p.exists():
        print(f"!! missing {p}")
        continue
    df = pd.read_parquet(p)
    pred = np.clip(df["pred_mw"].values, 0.0, 90.09)
    rev = pd.DataFrame({"prediction": pred[::-1]}).reset_index(drop=True)
    out_csv = ROOT / "experiments/active" / exp / csv_name
    rev.to_csv(out_csv, index=False)
    print(f"OK {csv_name}: rows={len(rev)}, mean={rev['prediction'].mean():.2f}")
    cmd = ["bash", "-c", f"cd {ROOT} && EXPERIMENT_HYGIENE_SUBMIT=1 .venv/bin/python scripts/submit.py --exp {exp} --task q1 --file {csv_name} --force --force-tempo --no-poll 2>&1 | tail -4"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
print("DONE")
