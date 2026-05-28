#!/usr/bin/env python3
"""Compute route-wise teacher/student gap from pass@8 per-sample outputs."""

from __future__ import annotations
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import json


def load_jsonl(path: Path):
    rows = {}
    with path.open(errors='ignore') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            # Existing pass@8 tools are not fully uniform: no-hint outputs use
            # ref_id only, while hint outputs may add an offset sample_id. The
            # dataset identity is ref_id, so use it first for cross-run joins.
            key = row.get('ref_id')
            if key is None:
                key = row.get('sample_id')
            rows[str(key)] = row
    return rows


def route_candidate(candidate, group_mean, group_std, tau_good, tau_fail, delta):
    reward = float(candidate.get('reward') or 0.0)
    iou = float(candidate.get('iou') or 0.0)
    valid_box = float(candidate.get('valid_box') or 0.0) > 0
    adv = (reward - group_mean) / (group_std + 1e-6)
    if valid_box and iou >= tau_good and reward >= group_mean:
        return 'good', adv
    if (not valid_box) or iou < tau_fail or adv < -delta:
        return 'failed', adv
    return 'ambiguous', adv


def summarize(values):
    if not values:
        return {}
    return {
        'n': len(values),
        'mean_student_iou': round(mean(v['student_iou'] for v in values), 6),
        'mean_teacher_iou': round(mean(v['teacher_iou'] for v in values), 6),
        'mean_teacher_minus_student': round(mean(v['teacher_minus_student'] for v in values), 6),
        'teacher_win_rate': round(sum(v['teacher_minus_student'] > 0 for v in values) / len(values), 6),
        'student_win_rate': round(sum(v['teacher_minus_student'] < 0 for v in values) / len(values), 6),
        'tie_rate': round(sum(v['teacher_minus_student'] == 0 for v in values) / len(values), 6),
        'student_iou@0.5': round(sum(v['student_iou'] > 0.5 for v in values) / len(values), 6),
        'teacher_iou@0.5': round(sum(v['teacher_iou'] > 0.5 for v in values) / len(values), 6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--student_nohint_per_sample', required=True, type=Path)
    parser.add_argument('--teacher_hint_per_sample', required=True, type=Path)
    parser.add_argument('--out_dir', required=True, type=Path)
    parser.add_argument('--name', default='')
    parser.add_argument('--tau_good', type=float, default=0.5)
    parser.add_argument('--tau_fail', type=float, default=0.3)
    parser.add_argument('--delta', type=float, default=0.5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    student_rows = load_jsonl(args.student_nohint_per_sample)
    teacher_rows = load_jsonl(args.teacher_hint_per_sample)

    per_route = defaultdict(list)
    per_candidate_path = args.out_dir / 'route_teacher_gap_per_candidate.csv'
    with per_candidate_path.open('w', newline='') as f:
        fieldnames = [
            'sample_id',
            'route',
            'rollout_id',
            'student_iou',
            'teacher_iou',
            'teacher_minus_student',
            'student_reward',
            'student_advantage',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id, student in student_rows.items():
            teacher = teacher_rows.get(sample_id)
            if not teacher:
                continue
            teacher_iou = float(teacher.get('first_iou') or 0.0)
            candidates = student.get('candidates') or []
            if not candidates:
                continue
            rewards = [float(c.get('reward') or 0.0) for c in candidates]
            group_mean = sum(rewards) / max(1, len(rewards))
            group_std = math.sqrt(sum((r - group_mean)**2 for r in rewards) / max(1, len(rewards)))
            for candidate in candidates:
                route, adv = route_candidate(candidate, group_mean, group_std, args.tau_good, args.tau_fail, args.delta)
                student_iou = float(candidate.get('iou') or 0.0)
                item = {
                    'sample_id': sample_id,
                    'route': route,
                    'rollout_id': candidate.get('rollout_id'),
                    'student_iou': student_iou,
                    'teacher_iou': teacher_iou,
                    'teacher_minus_student': teacher_iou - student_iou,
                    'student_reward': float(candidate.get('reward') or 0.0),
                    'student_advantage': adv,
                }
                per_route[route].append(item)
                writer.writerow(item)

    total = sum(len(v) for v in per_route.values())
    summary = {
        'name': args.name,
        'student_nohint_per_sample': str(args.student_nohint_per_sample),
        'teacher_hint_per_sample': str(args.teacher_hint_per_sample),
        'tau_good': args.tau_good,
        'tau_fail': args.tau_fail,
        'delta': args.delta,
        'total_candidates': total,
        'routes': {},
    }
    for route in ('good', 'ambiguous', 'failed'):
        route_values = per_route.get(route, [])
        row = summarize(route_values)
        row['route_ratio'] = round(len(route_values) / max(1, total), 6)
        summary['routes'][route] = row
    all_values = [item for values in per_route.values() for item in values]
    summary['overall'] = summarize(all_values)

    summary_path = args.out_dir / 'route_teacher_gap.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    status = {
        'status': 'done',
        'summary_path': str(summary_path),
        'per_candidate_csv': str(per_candidate_path),
        'total_candidates': total,
    }
    (args.out_dir / 'status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status, ensure_ascii=False))


if __name__ == '__main__':
    main()
