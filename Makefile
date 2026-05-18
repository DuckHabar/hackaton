.PHONY: help install q1 day18 all test clean

help:
	@echo "доступные цели:"
	@echo "  make install   - создать venv и поставить зависимости"
	@echo "  make q1        - построить прогноз Q1"
	@echo "  make day18     - построить прогноз на 18.05"
	@echo "  make all       - оба прогноза"
	@echo "  make test      - прогнать автотесты"
	@echo "  make clean     - удалить кеши и байткод"

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

q1:
	python run_q1.py

day18:
	python run_day18.py

all:
	python run.py

test:
	.venv/bin/pytest tests/

clean:
	find . -name __pycache__ -type d -exec rm -rf {} +
	find . -name '*.pyc' -delete
