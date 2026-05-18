"""convert variant parquets в CSV и submit на платформу."""
import sys, subprocess
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/duck/wind_hackathon")

SUBMITS = [
    # EXP-042 variants (3)
    ("EXP-042", "valid_pred_A_fine_024.parquet", "sub_finerbins_024.csv"),
    ("EXP-042", "valid_pred_B_daynight_024.parquet", "sub_daynight_024.csv"),
    ("EXP-042", "valid_pred_C_037style_021.parquet", "sub_conformal_021.csv"),
    # EXP-043 blend grid (3)
    ("EXP-043", "valid_pred_blend_w037_75_w024_25.parquet", "sub_blend_75_037.csv"),
    ("EXP-043", "valid_pred_blend_w037_50_w024_50.parquet", "sub_blend_50_037.csv"),
    ("EXP-043", "valid_pred_blend_w037_25_w024_75.parquet", "sub_blend_25_037.csv"),
]

for exp, parquet_name, csv_name in SUBMITS:
    p = ROOT / "experiments/active" / exp / parquet_name
    if not p.exists():
        print(f"!! missing {p}")
        continue
    df = pd.read_parquet(p)
    pred = np.clip(df["pred_mw"].values, 0.0, 90.09)
    # Reverse + header 'prediction'
    rev = pd.DataFrame({"prediction": pred[::-1]}).reset_index(drop=True)
    out_csv = ROOT / "experiments/active" / exp / csv_name
    rev.to_csv(out_csv, index=False)
    print(f"OK {csv_name}: rows={len(rev)}, mean={rev['prediction'].mean():.2f}, min={rev['prediction'].min():.2f}, max={rev['prediction'].max():.2f}")
    # Submit
    cmd = ["bash", "-c", f"cd {ROOT} && EXPERIMENT_HYGIENE_SUBMIT=1 .venv/bin/python scripts/submit.py --exp {exp} --task q1 --file {csv_name} --force --force-tempo --no-poll 2>&1 | tail -5"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
print("DONE")
