"""прогноз q1 на тестовых данных.

вход: model/q1/valid_pred.parquet (предсказания после обучения)
      model/q1/bias_map.json (поправка по бинам)
выход: submissions/q1/q1_final.csv (2126 строк, заголовок prediction)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model" / "q1"
OUT = ROOT / "submissions" / "q1" / "q1_final.csv"
BINS = [0, 3, 5, 7, 9, 11, 13, 16, 30]
RATED = 90.09


def apply_bias(pred, ws, bias_map):
    bins = np.digitize(ws, bins=BINS[1:-1])
    return pred + np.array([bias_map.get(str(int(b)), 0.0) for b in bins])


def main():
    valid_pred = MODEL_DIR / "valid_pred.parquet"
    bias_file = MODEL_DIR / "bias_map.json"
    if not valid_pred.is_file() or not bias_file.is_file():
        raise SystemExit(
            "не найдены артефакты модели. сначала запустите: python train_q1.py"
        )

    df = pd.read_parquet(valid_pred).sort_values("dt").reset_index(drop=True)
    bias_map = json.loads(bias_file.read_text())
    pred_cal = np.clip(
        apply_bias(df["pred_mw"].values, df["ws_120m"].values, bias_map), 0, RATED
    )
    out_df = pd.DataFrame({"prediction": pred_cal})
    out_df = out_df.iloc[::-1].reset_index(drop=True)

    rows = len(out_df)
    if rows != 2126:
        raise SystemExit(f"ошибка: ожидалось 2126 строк, получено {rows}")
    if out_df["prediction"].isna().any() or np.isinf(out_df["prediction"]).any():
        raise SystemExit("ошибка: есть nan или inf")
    lo, hi = float(out_df["prediction"].min()), float(out_df["prediction"].max())
    if lo < -1.0 or hi > 94.6:
        raise SystemExit(f"ошибка: значения вне диапазона: min {lo}, max {hi}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)
    print(f"сохранено: {OUT}")
    print(f"строк: {rows}, диапазон: [{lo:.3f}, {hi:.3f}], среднее: {out_df['prediction'].mean():.3f}")


if __name__ == "__main__":
    main()
