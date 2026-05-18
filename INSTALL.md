# Установка

Инструкция для запуска решения на чистой машине. Подходит для Linux, macOS, Windows. Нужен Python 3.10 или новее.

## Шаги

- проверить версию питона:

  ```bash
  python3 --version
  ```

- создать окружение в корне репозитория:

  ```bash
  python3 -m venv .venv
  ```

- активировать окружение:

  ```bash
  source .venv/bin/activate
  ```

  Для выхода - команда `deactivate`.

- обновить pip и поставить зависимости:

  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

  Загрузка около 400 МБ, занимает 2-5 минут.

- быстрая проверка:

  ```bash
  python -c "import lightgbm, pandas, numpy, sklearn, scipy, pyarrow; print('ok')"
  ```

## Запуск

```bash
python run.py
```

Скрипт построит оба прогноза (Q1 и 18 мая) и положит CSV в `submissions/`. На Windows вместо `python` иногда нужен `py` либо полный путь к интерпретатору.

Отдельные цели:

- только Q1: `python run_q1.py`
- только 18 мая: `python run_day18.py`
- альтернативный вариант для 18 мая: `python run_day18.py --variant om`
- тесты: `pytest tests/`

## Возможные проблемы

- если pip ругается на отсутствие `wheel` - `pip install wheel` и повторить;
- на macOS с Apple Silicon LightGBM иногда требует `libomp` из Homebrew: `brew install libomp`;
- если не находит `python3` - указать полный путь, например `/opt/homebrew/bin/python3`.
