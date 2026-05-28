#!/usr/bin/env python3
"""Aggregate greedy and pass@8 summaries into a gap-reduction table."""

from __future__ import annotations
import argparse
import csv
from pathlib import Path

import json


def load_json(path: Path):
    if not path or not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def round_metric(value):
    return None if value is None else round(float(value), 4)


def recover_pass8_metrics(summary_path: Path, summary: dict) -> dict:
    """Recover new pass@8 fields from old per-sample candidate dumps."""
    if summary.get('first_mIoU') is not None and summary.get('best@8_mIoU') is not None:
        return summary

    per_sample = summary_path.parent / 'per_sample.jsonl'
    if not per_sample.exists():
        return summary

    total = 0
    first_ious = []
    best_ious = []
    first_parse = 0
    first_valid = 0
    any_parse = 0
    any_valid = 0
    with per_sample.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            candidates = row.get('candidates') or []
            if not candidates:
                continue
            total += 1
            first = candidates[0]
            best = max(candidates, key=lambda c: float(c.get('iou') or 0.0))
            first_iou = float(first.get('iou') or 0.0)
            best_iou = float(best.get('iou') or 0.0)
            first_ious.append(first_iou)
            best_ious.append(best_iou)
            if first.get('parse_ok'):
                first_parse += 1
            if first.get('valid_box'):
                first_valid += 1
            if any(c.get('parse_ok') for c in candidates):
                any_parse += 1
            if any(c.get('valid_box') for c in candidates):
                any_valid += 1

    if not total:
        return summary

    recovered = dict(summary)
    recovered['first_mIoU'] = round_metric(sum(first_ious) / total)
    recovered['best@8_mIoU'] = round_metric(sum(best_ious) / total)
    recovered['first_parse_rate'] = round_metric(first_parse / total)
    recovered['first_valid_rate'] = round_metric(first_valid / total)
    recovered['any_parse_rate'] = round_metric(any_parse / total)
    recovered['any_valid_rate'] = round_metric(any_valid / total)
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        key = f'{threshold:.1f}'
        recovered[f'first_IoU@{key}'] = round_metric(sum(iou >= threshold for iou in first_ious) / total)
        value = round_metric(sum(iou >= threshold for iou in best_ious) / total)
        recovered[f'best@8_IoU@{key}'] = value
        recovered[f'pass@8@{key}'] = value
    return recovered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--out_dir', required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    rows = []
    for item in manifest['models']:
        greedy = load_json(Path(item['greedy_summary']))
        if greedy is None:
            continue
        greedy_miou = greedy.get('mIoU')
        for hint in item['pass8']:
            summary_path = Path(hint['summary'])
            summary = load_json(summary_path)
            if summary is None:
                rows.append({
                    'model_id': item['id'],
                    'hint_mode': hint['hint_mode'],
                    'status': 'missing',
                    'summary': hint['summary'],
                })
                continue
            summary = recover_pass8_metrics(summary_path, summary)
            first = summary.get('first_mIoU')
            best = summary.get('best@8_mIoU')
            gap = None if best is None or greedy_miou is None else round(best - greedy_miou, 6)
            rows.append({
                'model_id': item['id'],
                'model_name': item['name'],
                'hint_mode': hint['hint_mode'],
                'status': 'done',
                'greedy_mIoU': greedy_miou,
                'first_mIoU': first,
                'best@8_mIoU': best,
                'gap_best_minus_greedy': gap,
                'IoU@0.5_greedy': greedy.get('IoU@0.5'),
                'first_IoU@0.5': summary.get('first_IoU@0.5'),
                'best@8_IoU@0.5': summary.get('best@8_IoU@0.5'),
                'summary': hint['summary'],
            })

    by_hint_baseline = {}
    for row in rows:
        if row.get('status') == 'done' and row['model_id'] == manifest.get('gap_baseline_model', 'sft'):
            by_hint_baseline[row['hint_mode']] = row.get('gap_best_minus_greedy')
    for row in rows:
        base_gap = by_hint_baseline.get(row.get('hint_mode'))
        gap = row.get('gap_best_minus_greedy')
        if row.get('status') == 'done' and base_gap and gap is not None:
            row['gap_reduction_vs_baseline'] = round((base_gap - gap) / base_gap, 6)
        else:
            row['gap_reduction_vs_baseline'] = None

    json_path = args.out_dir / 'pass8_gap_summary.json'
    csv_path = args.out_dir / 'pass8_gap_summary.csv'
    json_path.write_text(json.dumps({'rows': rows}, indent=2, ensure_ascii=False))
    with csv_path.open('w', newline='') as f:
        fieldnames = [
            'model_id', 'model_name', 'hint_mode', 'status', 'greedy_mIoU', 'first_mIoU', 'best@8_mIoU',
            'gap_best_minus_greedy', 'gap_reduction_vs_baseline', 'IoU@0.5_greedy', 'first_IoU@0.5', 'best@8_IoU@0.5',
            'summary'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    print(json.dumps({'status': 'done', 'json': str(json_path), 'csv': str(csv_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
