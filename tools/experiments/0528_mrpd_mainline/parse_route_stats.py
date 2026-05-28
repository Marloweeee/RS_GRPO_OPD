#!/usr/bin/env python3
"""Parse MRPD route metrics from training logs.

The trainer prints Python-dict style metric records. This script extracts the
records that contain route metrics, summarizes them globally and by progress
segments, and writes JSON/CSV artifacts for paper tables and plots.
"""

from __future__ import annotations
import argparse
import ast
import csv
import re
from pathlib import Path
from statistics import mean

import json

METRIC_KEYS = [
    'sdpo/route_good',
    'sdpo/route_failed',
    'sdpo/route_ambiguous',
    'sdpo/iou_mean',
    'sdpo/iou@0.5',
    'sdpo/loss',
    'kl',
    'reward',
    'reward_std',
]


def iter_metric_records(log_path: Path):
    pattern = re.compile(r"\{.*'global_step/max_steps':\s*'(\d+)/(\d+)'.*\}")
    with log_path.open(errors='ignore') as f:
        for line in f:
            if 'sdpo/route_' not in line or "'global_step/max_steps'" not in line:
                continue
            match = pattern.search(line)
            if not match:
                continue
            text = match.group(0)
            try:
                record = ast.literal_eval(text)
            except Exception:
                continue
            step, max_steps = match.groups()
            record['_step'] = int(step)
            record['_max_steps'] = int(max_steps)
            yield record


def summarize(records):
    if not records:
        return {}
    result = {'num_records': len(records), 'first_step': records[0]['_step'], 'last_step': records[-1]['_step']}
    for key in METRIC_KEYS:
        values = [float(r[key]) for r in records if key in r and r[key] is not None]
        if values:
            result[key] = {
                'mean': round(mean(values), 6),
                'first': round(values[0], 6),
                'last': round(values[-1], 6),
                'min': round(min(values), 6),
                'max': round(max(values), 6),
            }
    return result


def segment_records(records):
    if not records:
        return {}
    max_step = max(r['_max_steps'] for r in records)
    segments = {
        'early_0_20pct': [],
        'middle_20_80pct': [],
        'late_80_100pct': [],
    }
    for record in records:
        ratio = record['_step'] / max(1, max_step)
        if ratio <= 0.2:
            segments['early_0_20pct'].append(record)
        elif ratio <= 0.8:
            segments['middle_20_80pct'].append(record)
        else:
            segments['late_80_100pct'].append(record)
    return {name: summarize(rows) for name, rows in segments.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True, type=Path)
    parser.add_argument('--out_dir', required=True, type=Path)
    parser.add_argument('--name', default='')
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = list(iter_metric_records(args.log))
    summary = {
        'name': args.name or args.log.stem,
        'log': str(args.log),
        'global': summarize(records),
        'segments': segment_records(records),
    }

    summary_path = args.out_dir / 'route_stats.json'
    csv_path = args.out_dir / 'route_stats_per_step.csv'
    with summary_path.open('w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with csv_path.open('w', newline='') as f:
        fieldnames = ['step', 'max_steps'] + METRIC_KEYS
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {'step': record['_step'], 'max_steps': record['_max_steps']}
            for key in METRIC_KEYS:
                row[key] = record.get(key)
            writer.writerow(row)

    status = {
        'status': 'done',
        'summary_path': str(summary_path),
        'csv_path': str(csv_path),
        'num_records': len(records),
    }
    with (args.out_dir / 'status.json').open('w') as f:
        json.dump(status, f, indent=2)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == '__main__':
    main()
