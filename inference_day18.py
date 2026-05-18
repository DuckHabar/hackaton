"""прогноз на 18 мая 2026 по уже обученной модели.

подгружает model/day18/model_final.pkl, собирает признаки на 24 часа целевого
дня (по варианту погоды - org по умолчанию, можно --variant om), предсказывает
и пишет csv. при отсутствии модели делегирует обучение train_day18.py.
"""

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.day18_task import data as day18_data
from pipeline.day18_task import features as day18_feat
from pipeline.day18_task import weather_variants as wv
from pipeline.day18_task.validate import TARGET, TS, _select_features

MODEL_DIR = ROOT / "model" / "day18"
OUT_CSV = ROOT / "submissions" / "day18" / "day18_final.csv"
CFG_PATH = ROOT / "pipeline" / "day18_task" / "config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)
TARGET_DATE = CFG["target_date"]
CLIP_MIN = CFG["clip_min"]
CLIP_MAX = CFG["clip_max"]
HEADER = CFG["output_header"]


def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _parse_args():
    p = argparse.ArgumentParser(description="прогноз выработки на 2026-05-18")
    p.add_argument("--variant", choices=("om", "org"), default="org",
                   help="вариант погоды (org - по умолчанию)")
    return p.parse_args()


def _load_model():
    path = MODEL_DIR / "model_final.pkl"
    if not path.is_file():
        _log("обученной модели нет, запускаю train_day18.py")
        res = subprocess.run([sys.executable, str(ROOT / "train_day18.py")], check=False)
        if res.returncode != 0:
            raise SystemExit("обучение не удалось")
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_target(variant):
    df_all = day18_data.load_all()
    if variant == "om":
        try:
            df_all = wv.apply_weather_om(df_all, target_dates=[TARGET_DATE])
            _log("погода: консенсус open-meteo на 18.05")
        except Exception as exc:
            _log(f"open-meteo недоступно ({exc}), беру вариант org")
    full = day18_feat.build_all_features(df_all)
    mask = pd.to_datetime(full[TS]).dt.normalize() == pd.Timestamp(TARGET_DATE)
    out = full[mask].sort_values(TS).reset_index(drop=True)
    if len(out) != 24:
        raise RuntimeError(f"ожидаем 24 строки на {TARGET_DATE}, получено {len(out)}")
    return out


def _save(preds):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame({HEADER: preds})
    df_out.to_csv(OUT_CSV, index=False)
    if df_out.shape != (24, 1) or df_out.columns.tolist() != [HEADER]:
        raise RuntimeError("неверный формат csv")
    if np.isnan(preds).any() or np.isinf(preds).any():
        raise RuntimeError("есть nan/inf")
    if (preds < CLIP_MIN).any() or (preds > CLIP_MAX).any():
        raise RuntimeError("значения вне диапазона")
    _log(f"csv записан: {OUT_CSV}")


def main():
    args = _parse_args()
    _log(f"старт прогноза, вариант = {args.variant}")
    model = _load_model()
    target = _build_target(args.variant)
    feat_list = _select_features(args.variant)
    preds = np.clip(model.predict(target[feat_list]), CLIP_MIN, CLIP_MAX)
    _save(preds)
    print(json.dumps({
        "variant": args.variant,
        "rows": int(len(preds)),
        "min": float(preds.min()),
        "max": float(preds.max()),
        "mean": float(preds.mean()),
        "sum_total": float(preds.sum()),
        "sum_00_08": float(preds[:8].sum()),
        "output_csv": str(OUT_CSV.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
