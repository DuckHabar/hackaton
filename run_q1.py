#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""обучение + прогноз q1.

вызывает train_q1.py (подбор весов смешивания через optuna)
и inference_q1.py (формирование итогового csv).
"""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent

    print("обучение модели q1...")
    res = subprocess.run([sys.executable, str(root / "train_q1.py")], check=False)
    if res.returncode != 0:
        print("ошибка: обучение q1 не удалось")
        return res.returncode

    print()
    print("формирование прогноза q1...")
    res = subprocess.run([sys.executable, str(root / "inference_q1.py")], check=False)
    if res.returncode != 0:
        print("ошибка: прогноз q1 не удался")
        return res.returncode

    out = root / "submissions" / "q1" / "q1_final.csv"
    print()
    print(f"готово: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
