#!/usr/bin/env python3
"""Lightweight status monitor for the four MRPD SFT ablation training jobs."""

from __future__ import annotations
import glob
import os
import re
import time
from datetime import datetime
from pathlib import Path

RUNS = [
    (
        'sft_grpo_only',
        '/data/codes/gui_grounding/data/logs/training/q4b-sft-grpo-only-e1-20260527-082554.rjob.log',
    ),
    (
        'sft_opsd_only',
        '/data/codes/gui_grounding/data/logs/training/q4b-sft-opsd-only-gauss-e1-20260527-082554.rjob.log',
    ),
    (
        'sft_grpo_opsd_zoom',
        '/data/codes/gui_grounding/data/logs/training/q4b-sftopsd-zoom-e1-20260527-082554.rjob.log',
    ),
    (
        'sft_grpo_opsd_jitter',
        '/data/codes/gui_grounding/data/logs/training/q4b-sftopsd-jitter-e1-20260527-082554.rjob.log',
    ),
    (
        'sft_grpo_opsd_no_hint',
        '/data/codes/gui_grounding/data/logs/training/q4b-sftopsd-nohint-e1-20260527-093400-aos2n.rjob.log',
    ),
    (
        'sft_grpo_opsd_soft_window',
        '/data/codes/gui_grounding/data/logs/training/q4b-sftopsd-soft-window-e1-20260527-095000.rjob.log',
    ),
]

STATUS_PATH = Path('/data/codes/gui_grounding/data/logs/training/mrpd_sft_ablation_monitor.status.log')
INTERVAL_SEC = int(os.environ.get('MRPD_MONITOR_INTERVAL_SEC', '300'))
STRICT_ERROR_RE = re.compile(
    r'Traceback \(most recent call last\)|RuntimeError:|CUDA out of memory|'
    r'OutOfMemoryError|NCCL.*error|\bERROR\b|Failed to|failed with',
    re.IGNORECASE,
)
STEP_RE = re.compile(r"global_step/max_steps': '([0-9]+/[0-9]+)'|global_step/max_steps=([0-9]+/[0-9]+)")
CKPT_RE = re.compile(r'(?:latest_ckpt=|last_model_checkpoint: |Saving model checkpoint to )'
                     r'(/mnt/jfs/copilot/[^\s,]+/checkpoint-[0-9]+)')


def latest_checkpoint_from_log(text: str) -> tuple[int, str]:
    checkpoints: list[tuple[int, str]] = []
    for path in CKPT_RE.findall(text):
        try:
            step = int(os.path.basename(path).split('-')[-1])
        except ValueError:
            step = -1
        checkpoints.append((step, path))
    return sorted(checkpoints)[-1] if checkpoints else (-1, 'none')


def summarize_once() -> str:
    lines = [f'==== {datetime.now():%Y-%m-%d %H:%M:%S} ====']
    for name, log_path in RUNS:
        path = Path(log_path)
        text = path.read_text(errors='ignore') if path.exists() else ''
        matches = STEP_RE.findall(text)
        step = 'none'
        if matches:
            step = matches[-1][0] or matches[-1][1]
        ckpt_step, ckpt_path = latest_checkpoint_from_log(text)
        lines.append(f'{name}: step={step} latest_step={ckpt_step} latest_ckpt={ckpt_path} '
                     f"completed_markers={text.count('completed')} strict_error={bool(STRICT_ERROR_RE.search(text))}")
    return '\n'.join(lines) + '\n'


def main() -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    while True:
        status = summarize_once()
        STATUS_PATH.write_text(status, encoding='utf-8')
        print(status, flush=True)
        time.sleep(INTERVAL_SEC)


if __name__ == '__main__':
    main()
