#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""обучение + прогноз на 18 мая.

вызывает train_day18.py (обучение квантильного lightgbm на полной истории)
и inference_day18.py (формирование итогового csv).
по умолчанию вариант org - сырые данные от организатора.
для подмены погоды консенсусом open-meteo: python run_day18.py --variant om
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="запуск обучения и прогноза на 18 мая")
    parser.add_argument(
        "--variant",
        choices=("om", "org"),
        default="org",
        help="вариант погоды для инференса (по умолчанию org)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent

    print("обучение модели на 18 мая...")
    res = subprocess.run([sys.executable, str(root / "train_day18.py")], check=False)
    if res.returncode != 0:
        print("ошибка: обучение не удалось")
        return res.returncode

    print()
    print(f"формирование прогноза на 18 мая (вариант: {args.variant})...")
    res = subprocess.run(
        [sys.executable, str(root / "inference_day18.py"), "--variant", args.variant],
        check=False,
    )
    if res.returncode != 0:
        print("ошибка: прогноз не удался")
        return res.returncode

    out = root / "submissions" / "day18" / "day18_final.csv"
    print()
    print(f"готово: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
