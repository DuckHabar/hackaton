#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# главная точка входа: строит оба прогноза (Q1 и 18 мая).
# результаты появятся в каталоге submissions/.

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent

    print("запуск прогноза Q1...")
    res_q1 = subprocess.run(
        [sys.executable, str(root / "run_q1.py")], check=False
    )
    if res_q1.returncode != 0:
        print("ошибка: сборка прогноза Q1 завершилась неудачно")
        return res_q1.returncode

    print()
    print("запуск прогноза 18.05...")
    res_day18 = subprocess.run(
        [sys.executable, str(root / "run_day18.py")], check=False
    )
    if res_day18.returncode != 0:
        print("ошибка: сборка прогноза на 18.05 завершилась неудачно")
        return res_day18.returncode

    print()
    print("готово. результаты в submissions/")
    print(f"  - {root / 'submissions' / 'q1' / 'q1_final.csv'}")
    print(f"  - {root / 'submissions' / 'day18' / 'day18_final.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
