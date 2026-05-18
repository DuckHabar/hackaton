import json
import time
from pathlib import Path

import pytest

from monitors.gpu_watchdog import (
    parse_nvidia_smi_csv,
    decide_idle,
    write_sample,
)


def test_parse_nvidia_smi_single_gpu():
    raw = "85, 12345\n"
    samples = parse_nvidia_smi_csv(raw)
    assert samples == [{"util": 85, "mem_used_mib": 12345}]


def test_parse_nvidia_smi_multi_gpu():
    raw = "10, 1000\n80, 25000\n"
    samples = parse_nvidia_smi_csv(raw)
    assert samples == [
        {"util": 10, "mem_used_mib": 1000},
        {"util": 80, "mem_used_mib": 25000},
    ]


def test_parse_nvidia_smi_strips_whitespace():
    raw = "   42  ,   8192   \n"
    samples = parse_nvidia_smi_csv(raw)
    assert samples == [{"util": 42, "mem_used_mib": 8192}]


def test_parse_nvidia_smi_ignores_blank_lines():
    raw = "\n50, 100\n\n60, 200\n\n"
    samples = parse_nvidia_smi_csv(raw)
    assert len(samples) == 2


def test_decide_idle_true_when_all_under_threshold():
    # 10 минут, все util < 30
    history = [(time.time() - i * 60, 15) for i in range(11)]
    assert decide_idle(history, threshold_util=30, window_seconds=600) is True


def test_decide_idle_false_when_recent_busy():
    history = [(time.time() - i * 60, 10) for i in range(11)]
    history[0] = (time.time(), 90)  # последнее значение высокое
    assert decide_idle(history, threshold_util=30, window_seconds=600) is False


def test_decide_idle_false_when_history_too_short():
    history = [(time.time(), 5)]  # одна точка, недостаточно данных
    assert decide_idle(history, threshold_util=30, window_seconds=600) is False


def test_write_sample_appends_jsonl(tmp_path: Path):
    log = tmp_path / "gpu_util.jsonl"
    write_sample(log, {"ts": 1700000000.0, "util": 75, "mem_used_mib": 1000})
    write_sample(log, {"ts": 1700000060.0, "util": 80, "mem_used_mib": 1200})
    lines = log.read_text().strip().split("\n")
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["util"] == 75
