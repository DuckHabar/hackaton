import time
from pathlib import Path

import pytest
import yaml

from monitors.gpu_dispatcher import (
    list_pending_jobs,
    pick_next_job,
    parse_job_yaml,
    gpu_is_free,
)


def _write_job(dir_: Path, name: str, payload: dict) -> Path:
    p = dir_ / name
    p.write_text(yaml.safe_dump(payload))
    return p


def test_list_pending_jobs_sorted_by_priority_then_time(tmp_path: Path):
    q = tmp_path / ".gpu_queue"
    q.mkdir()
    older_low = _write_job(q, "low_1700000000_a.yaml", {"name": "a"})
    newer_low = _write_job(q, "low_1700000010_b.yaml", {"name": "b"})
    older_high = _write_job(q, "high_1700000005_c.yaml", {"name": "c"})

    found = list_pending_jobs(q)
    names = [p.name for p in found]
    assert names == ["high_1700000005_c.yaml", "low_1700000000_a.yaml", "low_1700000010_b.yaml"]


def test_pick_next_job_returns_none_if_empty(tmp_path: Path):
    q = tmp_path / ".gpu_queue"
    q.mkdir()
    assert pick_next_job(q) is None


def test_pick_next_job_returns_highest_priority(tmp_path: Path):
    q = tmp_path / ".gpu_queue"
    q.mkdir()
    _write_job(q, "low_1700000000_a.yaml", {"name": "a", "cmd": "echo a"})
    _write_job(q, "med_1700000000_b.yaml", {"name": "b", "cmd": "echo b"})
    job = pick_next_job(q)
    assert job is not None
    assert job["name"] == "b"


def test_parse_job_yaml_rejects_missing_cmd(tmp_path: Path):
    p = tmp_path / "low_1_x.yaml"
    p.write_text(yaml.safe_dump({"name": "x"}))
    with pytest.raises(ValueError):
        parse_job_yaml(p)


def test_parse_job_yaml_returns_dict(tmp_path: Path):
    p = tmp_path / "low_1_x.yaml"
    p.write_text(yaml.safe_dump({"name": "x", "cmd": "python -c 'pass'"}))
    job = parse_job_yaml(p)
    assert job["name"] == "x"
    assert job["cmd"] == "python -c 'pass'"


def test_gpu_is_free_with_mock(monkeypatch):
    monkeypatch.setattr(
        "monitors.gpu_dispatcher._query_util",
        lambda: [5, 10],
    )
    assert gpu_is_free(threshold=20) is True


def test_gpu_is_free_false_when_busy(monkeypatch):
    monkeypatch.setattr(
        "monitors.gpu_dispatcher._query_util",
        lambda: [85],
    )
    assert gpu_is_free(threshold=20) is False


import json


def test_log_started_appends_jsonl(tmp_path: Path):
    from monitors.gpu_dispatcher import log_started
    root = tmp_path
    job = {"name": "test_job", "cmd": "echo hi"}
    log_started(root, job, "gpu_job_test_123")
    log_file = root / "logs" / "gpu_jobs.jsonl"
    assert log_file.exists()
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "started"
    assert record["session"] == "gpu_job_test_123"
    assert record["name"] == "test_job"
    assert record["cmd"] == "echo hi"


def test_pick_next_job_skips_invalid_yaml(tmp_path: Path):
    """если первый файл валиден yaml но без cmd, должен попробовать следующий."""
    from monitors.gpu_dispatcher import pick_next_job
    q = tmp_path / ".gpu_queue"
    q.mkdir()
    # 1-й по приоритету, invalid
    (q / "high_1700000000_bad.yaml").write_text(yaml.safe_dump({"name": "bad"}))
    # 2-й валиден
    (q / "high_1700000001_good.yaml").write_text(yaml.safe_dump({"name": "good", "cmd": "echo ok"}))
    job = pick_next_job(q)
    assert job is not None
    assert job["name"] == "good"
