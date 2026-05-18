"""обучение модели на 18 мая 2026 (вариант org).

обёртка над pipeline.day18_task.predict.run_final - оригинальный код, который
читает train_dataset, строит признаки (включая лаги выработки с защитой от
подсматривания), обучает LightGBM (тау = 0.5) и сразу делает прогноз на 24 часа.

выход:
- model/day18/model_final.pkl - обученная модель
- submissions/day18/day18_final.csv - прогноз на 18.05
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.day18_task.predict import run_final

VARIANT = "org"
MODEL_DIR = ROOT / "model" / "day18"
EXP_DIR = ROOT / "experiments" / "active" / "EXP-day18-001"
OUT_CSV = ROOT / "submissions" / "day18" / "day18_final.csv"


def main() -> int:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"обучение модели на 18 мая (вариант {VARIANT})")
    meta = run_final(branch=VARIANT, output_path=str(OUT_CSV))

    src_model = EXP_DIR / f"model_final_{VARIANT}.pkl"
    dst_model = MODEL_DIR / "model_final.pkl"
    if src_model.is_file():
        shutil.copy(src_model, dst_model)

    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
