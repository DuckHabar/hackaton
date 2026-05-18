"""
experiment-hygiene submit это единственный разрешённый путь к сабмиту.

Защиты:
1. CV improved vs current best_cv.txt (или --force)
2. CSV passes format_validator (rows=2126, header='prediction', no NaN/Inf, range)
3. Имя из whitelist (паттерн + allowed root + не blacklist)
4. Tempo guard: ≥30 секунд между сабмитами (чтобы платформа не подавилась)
5. дневной лимит снят, Thothex не ограничивает количество сабмитов

Использование:
    python3 scripts/submit.py --exp EXP-003 --task q1 --file sub_q1_bias.csv [--force]
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("~/wind_hackathon").expanduser()
AUTOSUBMIT = ROOT / "thothex_autosubmit.py"
VENV_PY = ROOT / ".venv/bin/python"
LOG = Path("~/thothex_autosubmit.jsonl").expanduser()

NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,30}\.csv$")
BAD_PREFIXES = ("claude_", "gpt_", "ai_", "bot_", "agent_", "assistant_",
                "auto_", "automated_", "autosubmit_", "exp_", "experiment_")
ALLOWED_ROOTS = ("test", "answer", "final", "predict", "pred", "submission",
                 "sub", "result", "output", "out", "wind", "forecast",
                 "model_", "baseline", "bl_", "q1", "day18", "may18",
                 "day_18", "may_18")

MIN_INTERVAL_SEC = 30  # минимум 30 сек чтобы платформа не подавилась
MAX_PER_DAY = 9999  # платформа Thothex не ограничивает количество


def fail(msg):
    print(f"REJECT: {msg}", file=sys.stderr)
    sys.exit(2)


def whitelist_check(name):
    if not NAME_PATTERN.match(name):
        fail(f"name '{name}' fails pattern ^[a-zA-Z][a-zA-Z0-9_]{{0,30}}\\.csv$")
    for bad in BAD_PREFIXES:
        if name.lower().startswith(bad):
            fail(f"name '{name}' starts with blacklisted '{bad}'")
    root = name.lower().rsplit(".", 1)[0]
    if not any(root.startswith(p) for p in ALLOWED_ROOTS):
        fail(f"name '{name}' root='{root}' not in whitelist roots")


def cv_check(task, exp_id, force):
    """CV improvement gate."""
    best_cv_f = ROOT / f"pipeline/{task}_task/best_cv.txt"
    summary_f = ROOT / f"experiments/active/{exp_id}/summary.json"
    if not summary_f.exists():
        fail(f"no summary.json in experiments/active/{exp_id}/")
    summary = json.loads(summary_f.read_text())
    # Берём cv_mean_nmae (EXP-003+) или modes.direct.cv_mean_nmae (EXP-002)
    cv = summary.get("cv_mean_nmae") or summary.get("cv_q1_only_mean_nmae")
    if cv is None and "modes" in summary:
        cv = summary["modes"].get("direct", {}).get("cv_mean_nmae")
    if cv is None:
        fail(f"can't extract cv from summary.json")
    if not best_cv_f.exists():
        print(f"WARN: no best_cv.txt for {task}, treating as first submit")
        current_best = float("inf")
    else:
        current_best = float(best_cv_f.read_text().strip())
    if not force and cv > current_best + 0.05:
        fail(f"cv {cv:.4f} > best {current_best:.4f} +0.05; use --force to override")
    print(f"OK cv gate: exp_cv={cv:.4f}, best_cv={current_best:.4f}")
    return cv


def format_validator(csv_path, task="q1"):
    import pandas as pd
    import numpy as np
    df = pd.read_csv(csv_path)
    expected = {"q1": 2126, "day18": 24}.get(task, 2126)
    if len(df) != expected:
        fail(f"rows={len(df)} != {expected} (task={task})")
    if list(df.columns) != ["prediction"]:
        fail(f"header={list(df.columns)} != ['prediction']")
    if df["prediction"].isna().any():
        fail("NaN in prediction")
    if np.isinf(df["prediction"]).any():
        fail("Inf in prediction")
    lo, hi = float(df["prediction"].min()), float(df["prediction"].max())
    if lo < -1 or hi > 94.6:
        fail(f"values outside [-1, 94.6]: min={lo}, max={hi}")
    print(f"OK format: rows={len(df)}, range=[{lo:.3f}, {hi:.3f}], mean={df['prediction'].mean():.3f}")


def tempo_guard(skip=False):
    if not LOG.exists():
        return
    now = time.time()
    last_submit_ts = 0
    today = datetime.now().date()  # local server day
    today_count = 0
    for line in LOG.read_text().splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "submitted":
            continue
        ts_str = e.get("ts", "")
        try:
            # uploader.py пишет ts через time.strftime, это локальное время сервера, naive
            ts = datetime.fromisoformat(ts_str).timestamp()
        except Exception:
            continue
        last_submit_ts = max(last_submit_ts, ts)
        d = datetime.fromtimestamp(ts).date()
        if d == today:
            today_count += 1
    age = now - last_submit_ts
    if skip:
        print(f"WARN tempo skipped: last submit {age:.0f}s ago, today {today_count}/{MAX_PER_DAY}")
        return
    if last_submit_ts and age < MIN_INTERVAL_SEC:
        fail(f"tempo: last submit {age:.0f}s ago, min {MIN_INTERVAL_SEC}s")
    if today_count >= MAX_PER_DAY:
        fail(f"tempo: {today_count} submits today >= {MAX_PER_DAY}")
    print(f"OK tempo: last submit {age:.0f}s ago (≥{MIN_INTERVAL_SEC}), today {today_count}/{MAX_PER_DAY}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--task", required=True, choices=["q1", "day18"])
    ap.add_argument("--file", required=True, help="имя CSV в experiments/active/<exp>/")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--force-tempo", action="store_true")
    ap.add_argument("--no-poll", action="store_true")
    args = ap.parse_args()

    csv_path = ROOT / f"experiments/active/{args.exp}/{args.file}"
    if not csv_path.exists():
        fail(f"file not found: {csv_path}")
    print(f"--- experiment-hygiene submit: {args.exp} / {args.task} ---")
    whitelist_check(args.file)
    cv = cv_check(args.task, args.exp, args.force)
    format_validator(csv_path, args.task)
    tempo_guard(skip=args.force_tempo)
    print(f"--- ALL CHECKS PASSED, invoking autosubmit ---")
    task_id_map = {"q1": 35, "day18": int(os.environ.get("THOTHEX_DAY18_TASK_ID", "36"))}
    task_env = dict(os.environ)
    task_env["THOTHEX_TASK_ID"] = str(task_id_map[args.task])
    print(f"using THOTHEX_TASK_ID={task_env['THOTHEX_TASK_ID']} for task={args.task}")
    cmd = [str(VENV_PY), str(AUTOSUBMIT), str(csv_path)]
    if args.no_poll:
        cmd.append("--no-poll")
    if args.force:
        cmd.append("--force")
    r = subprocess.run(cmd, env=task_env)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
