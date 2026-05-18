"""
EXP-002 predict.py: формирует submission CSV для Q1 task из valid_pred.parquet.

Critical:
- Переворот: df.iloc[::-1].reset_index(drop=True) (т.к. valid_features.csv reverse-ordered)
- Header: 'prediction'
- Rows: 2126
- Values clip к [−1, 94.6] (по конституции)
- Имя файла из whitelist: q1_*.csv / baseline_*.csv / sub_*.csv

Использование:
    python pipeline/q1_task/predict.py --exp EXP-002 --name baseline_q1.csv
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("~/wind_hackathon").expanduser()
NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,30}\.csv$")
BAD_PREFIXES = ("claude_", "gpt_", "ai_", "bot_", "agent_", "assistant_",
                "auto_", "automated_", "autosubmit_", "exp_", "experiment_")
ALLOWED_ROOTS = ("test", "answer", "final", "predict", "pred", "submission",
                 "sub", "result", "output", "out", "wind", "forecast",
                 "model_", "baseline", "bl_", "q1", "day18", "may18",
                 "day_18", "may_18")


def validate_name(name):
    if not NAME_PATTERN.match(name):
        raise ValueError(f"Имя '{name}' не подходит под whitelist pattern.")
    for bad in BAD_PREFIXES:
        if name.lower().startswith(bad):
            raise ValueError(f"Имя '{name}' содержит чёрный префикс '{bad}'.")
    root = name.lower().rsplit(".", 1)[0]
    if not any(root.startswith(p) for p in ALLOWED_ROOTS):
        raise ValueError(f"Имя '{name}' (root='{root}') не из whitelist.")
    return True


def validate_csv(df):
    """Constitution checks. Raises ValueError если что-то не так."""
    if len(df) != 2126:
        raise ValueError(f"Rows {len(df)} != 2126")
    if list(df.columns) != ["prediction"]:
        raise ValueError(f"Header {list(df.columns)} != ['prediction']")
    if df["prediction"].isna().any():
        raise ValueError("NaN в prediction")
    if np.isinf(df["prediction"]).any():
        raise ValueError("Inf в prediction")
    lo, hi = df["prediction"].min(), df["prediction"].max()
    if lo < -1 or hi > 94.6:
        raise ValueError(f"Values вне [−1, 94.6]: min={lo}, max={hi}")
    return True


def make_submission(exp_id, out_name):
    validate_name(out_name)
    pred_pq = ROOT / f"experiments/active/{exp_id}/valid_pred.parquet"
    if not pred_pq.exists():
        raise FileNotFoundError(f"Нет {pred_pq}")
    pred = pd.read_parquet(pred_pq)
    # pred.dt уже отсортирован по возрастанию (мы сортировали в lgbm_q50.py при concat)
    # А valid_features.csv был reverse-ordered (newest first). При predict valid мы
    # сохранили в parquet порядок по возрастанию dt. Чтобы вернуть к "newest first"
    # как ожидает hackathon, нужно reverse.
    pred_sorted = pred.sort_values("dt", ascending=True).reset_index(drop=True)
    # Теперь переворачиваем (новые сверху, как в valid_features.csv)
    pred_reversed = pred_sorted.iloc[::-1].reset_index(drop=True)
    # Clip к [0, 90.09] (теоретический max), затем к константам спеки [−1, 94.6]
    values = pred_reversed["pred_mw"].clip(lower=0.0, upper=90.09).values
    out = pd.DataFrame({"prediction": values})
    validate_csv(out)

    out_path = ROOT / f"experiments/active/{exp_id}/{out_name}"
    out.to_csv(out_path, index=False)
    print(f"OK saved: {out_path}")
    print(f"  rows={len(out)}, header=['prediction']")
    print(f"  min={out['prediction'].min():.3f}, max={out['prediction'].max():.3f}, "
          f"mean={out['prediction'].mean():.3f}")
    # Sanity-check на даты
    print(f"  dt range in valid: {pred_reversed['dt'].iloc[-1]} ... {pred_reversed['dt'].iloc[0]}  "
          f"(должно быть 2026-01-01 ... 2026-03-31, newest first)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    make_submission(args.exp, args.name)
