#!/usr/bin/env python3
"""CPU-only analysis for MRPD paper experiments.

The script consumes existing eval `per_sample.jsonl` files and dataset jsonl
files. It does not run model inference. Outputs are written as JSON/CSV/HTML so
the results can be copied directly into paper tables or follow-up experiment
notes.
"""

from __future__ import annotations
import argparse
import csv
import html
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json

REPO_ROOT = Path('/data/codes/gui_grounding/GUI-SD-code-main')
DATA_ROOT = Path('/data/codes/gui_grounding/data')
EVAL_ROOT = DATA_ROOT / 'logs' / 'eval'

DEFAULT_METHODS = {
    'sft':
    EVAL_ROOT / 'qwen3-4b-rrsisd-sft-20260526-220915-checkpoint-96-test-greedy' / 'per_sample.jsonl',
    'opsd_only':
    EVAL_ROOT / 'q4b-sft-opsd-only-gaussian-20260527-085701-test' / 'per_sample.jsonl',
    'grpo_only':
    EVAL_ROOT / 'q4b-sft-grpo-only-20260527-082554-test' / 'per_sample.jsonl',
    'mrpd_metric':
    EVAL_ROOT / 'gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_metric_gaussian_e1-20260526-225857_test' / 'per_sample.jsonl',
    'mrpd_ema098':
    EVAL_ROOT / 'gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_ema098_gaussian_e1-20260527-113500-ema098_test'
    / 'per_sample.jsonl',
    'mrpd_no_refresh':
    EVAL_ROOT / 'gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_no_refresh_gaussian_e1-20260527-114200-norefresh_test'
    / 'per_sample.jsonl',
    'mrpd_step80':
    EVAL_ROOT / 'gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_fixed_s80_gaussian_e1-20260527-113500-s80_test'
    / 'per_sample.jsonl',
    'mrpd_zoom':
    EVAL_ROOT / 'q4b-sft-grpo-opsd-zoom-in-20260527-082554-test' / 'per_sample.jsonl',
    'mrpd_jitter':
    EVAL_ROOT / 'q4b-sft-grpo-opsd-jitter-box-20260527-082554-test' / 'per_sample.jsonl',
    'mrpd_soft_window':
    EVAL_ROOT / 'q4b-sft-grpo-opsd-soft-window-20260527-100136-test' / 'per_sample.jsonl',
    'mrpd_no_hint':
    EVAL_ROOT / 'q4b-sft-grpo-opsd-no-hint-20260527-093400-aos2n-test' / 'per_sample.jsonl',
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_key(row: Dict[str, Any]) -> str:
    for key in ('sample_id', 'ref_id'):
        if key in row:
            return str(row[key])
    additional = row.get('additional_paras')
    if isinstance(additional, str):
        try:
            additional = json.loads(additional)
        except Exception:
            additional = {}
    if isinstance(additional, dict):
        for key in ('sample_id', 'ref_id'):
            if key in additional:
                return str(additional[key])
    return str(row.get('images', '')) + '::' + str(row.get('text', ''))


def load_method_file(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = load_jsonl(path)
    return {sample_key(row): row for row in rows}


def iou(row: Dict[str, Any]) -> float:
    value = row.get('iou')
    if value is None:
        value = row.get('best_iou')
    return float(value or 0.0)


def metric_summary(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    values = [iou(row) for row in rows]
    n = len(values)
    if n == 0:
        return {'n': 0, 'mIoU': 0.0, 'IoU@0.5': 0.0, 'IoU@0.6': 0.0, 'IoU@0.7': 0.0, 'IoU@0.8': 0.0, 'IoU@0.9': 0.0}
    out = {'n': float(n), 'mIoU': sum(values) / n}
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        out[f'IoU@{threshold:.1f}'] = sum(v >= threshold for v in values) / n
    return out


def paired_bootstrap(
    base: Dict[str, Dict[str, Any]],
    other: Dict[str, Dict[str, Any]],
    iterations: int,
    seed: int,
) -> Dict[str, Any]:
    keys = sorted(set(base) & set(other))
    base_values = [iou(base[k]) for k in keys]
    other_values = [iou(other[k]) for k in keys]
    n = len(keys)
    diffs = [b - a for a, b in zip(base_values, other_values)]
    observed = sum(diffs) / max(1, n)
    wins = sum(d > 0 for d in diffs)
    ties = sum(abs(d) <= 1e-12 for d in diffs)

    rng = random.Random(seed)
    samples: List[float] = []
    if n > 0:
        for _ in range(iterations):
            total = 0.0
            for _ in range(n):
                total += diffs[rng.randrange(n)]
            samples.append(total / n)
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))] if samples else 0.0
    hi = samples[int(0.975 * (len(samples) - 1))] if samples else 0.0
    if samples:
        p_two_sided = 2.0 * min(
            sum(x <= 0 for x in samples) / len(samples),
            sum(x >= 0 for x in samples) / len(samples),
        )
        p_two_sided = min(1.0, p_two_sided)
    else:
        p_two_sided = 1.0
    return {
        'n': n,
        'base_mean': sum(base_values) / max(1, n),
        'other_mean': sum(other_values) / max(1, n),
        'mean_delta': observed,
        'ci95_low': lo,
        'ci95_high': hi,
        'p_two_sided': p_two_sided,
        'win_rate': wins / max(1, n),
        'tie_rate': ties / max(1, n),
    }


def gt_box(row: Dict[str, Any]) -> Optional[List[float]]:
    box = row.get('gt_norm')
    if isinstance(box, list) and len(box) == 4:
        return [float(x) for x in box]
    return None


def area_bucket(box: List[float]) -> str:
    x1, y1, x2, y2 = box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / 1_000_000.0
    if area < 0.005:
        return 'tiny(<0.5%)'
    if area < 0.02:
        return 'small(0.5-2%)'
    if area < 0.08:
        return 'medium(2-8%)'
    return 'large(>=8%)'


def aspect_bucket(box: List[float]) -> str:
    x1, y1, x2, y2 = box
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    ratio = max(w / h, h / w)
    if ratio >= 5:
        return 'very_elongated(>=5)'
    if ratio >= 3:
        return 'elongated(3-5)'
    return 'regular(<3)'


def edge_bucket(box: List[float]) -> str:
    x1, y1, x2, y2 = box
    return 'edge' if x1 <= 50 or y1 <= 50 or x2 >= 950 or y2 >= 950 else 'center'


def category(row: Dict[str, Any]) -> str:
    value = row.get('category')
    if value:
        return str(value)
    return 'unknown'


def grouped_metric_table(
    methods: Dict[str, Dict[str, Dict[str, Any]]],
    base_method: str,
    group_fn,
    min_count: int = 1,
) -> List[Dict[str, Any]]:
    base_rows = methods[base_method]
    groups: Dict[str, List[str]] = defaultdict(list)
    for key, row in base_rows.items():
        group = group_fn(row)
        if group is not None:
            groups[str(group)].append(key)

    table: List[Dict[str, Any]] = []
    for group, keys in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(keys) < min_count:
            continue
        record: Dict[str, Any] = {'group': group, 'n': len(keys)}
        for method, rows in methods.items():
            values = [rows[k] for k in keys if k in rows]
            record[f'{method}_mIoU'] = metric_summary(values)['mIoU'] if values else None
            record[f'{method}_IoU@0.5'] = metric_summary(values)['IoU@0.5'] if values else None
        if record.get('sft_mIoU') is not None:
            for method in methods:
                if method != 'sft' and record.get(f'{method}_mIoU') is not None:
                    record[f'{method}_minus_sft'] = record[f'{method}_mIoU'] - record['sft_mIoU']
        if record.get('grpo_only_mIoU') is not None:
            for method in methods:
                if method != 'grpo_only' and record.get(f'{method}_mIoU') is not None:
                    record[f'{method}_minus_grpo'] = record[f'{method}_mIoU'] - record['grpo_only_mIoU']
        table.append(record)
    return table


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def split_image_id(row: Dict[str, Any]) -> str:
    additional = row.get('additional_paras')
    if isinstance(additional, str):
        try:
            additional = json.loads(additional)
        except Exception:
            additional = {}
    if isinstance(additional, dict) and additional.get('image_id'):
        return str(additional['image_id'])
    image = str(row.get('images', ''))
    return Path(image).stem


def split_leakage_report(paths: Dict[str, Path]) -> Dict[str, Any]:
    splits: Dict[str, Dict[str, Any]] = {}
    for split, path in paths.items():
        rows = load_jsonl(path)
        image_ids = [split_image_id(row) for row in rows]
        ref_ids = [sample_key(row) for row in rows]
        categories = Counter(category_from_dataset(row) for row in rows)
        splits[split] = {
            'path': str(path),
            'num_refs': len(rows),
            'num_images': len(set(image_ids)),
            'duplicate_ref_ids': len(ref_ids) - len(set(ref_ids)),
            'category_counts': dict(categories.most_common()),
            'image_ids': set(image_ids),
            'ref_ids': set(ref_ids),
        }

    overlap: Dict[str, Any] = {}
    names = sorted(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            image_overlap = sorted(splits[a]['image_ids'] & splits[b]['image_ids'])
            ref_overlap = sorted(splits[a]['ref_ids'] & splits[b]['ref_ids'])
            overlap[f'{a}_vs_{b}'] = {
                'num_image_overlap': len(image_overlap),
                'num_ref_overlap': len(ref_overlap),
                'image_overlap_examples': image_overlap[:20],
                'ref_overlap_examples': ref_overlap[:20],
            }

    public_splits = {}
    for split, info in splits.items():
        public_splits[split] = {k: v for k, v in info.items() if k not in {'image_ids', 'ref_ids'}}
    return {'splits': public_splits, 'overlap': overlap}


def category_from_dataset(row: Dict[str, Any]) -> str:
    additional = row.get('additional_paras')
    if isinstance(additional, str):
        try:
            additional = json.loads(additional)
        except Exception:
            additional = {}
    if isinstance(additional, dict) and additional.get('category'):
        return str(additional['category'])
    return 'unknown'


def html_table(rows: List[Dict[str, Any]], max_rows: Optional[int] = None) -> str:
    if max_rows is not None:
        rows = rows[:max_rows]
    if not rows:
        return '<p>无数据</p>'
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    body = ["<div class=\"table-wrap\"><table><thead><tr>"]
    body.extend(f'<th>{html.escape(str(field))}</th>' for field in fields)
    body.append('</tr></thead><tbody>')
    for row in rows:
        body.append('<tr>')
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                text = f'{value:.6f}'
                cls = 'num'
            else:
                text = '' if value is None else str(value)
                cls = ''
            body.append(f"<td class=\"{cls}\">{html.escape(text)}</td>")
        body.append('</tr>')
    body.append('</tbody></table></div>')
    return ''.join(body)


def build_report_html(results: Dict[str, Any]) -> str:
    style = """
    body{margin:0;background:#f6f8fb;color:#182233;
      font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",Arial,sans-serif}
    main{max-width:1480px;margin:0 auto;padding:28px 24px 48px}
    section{background:#fff;border:1px solid #d8e0ea;border-radius:8px;padding:16px;margin:14px 0}
    h1{margin:0 0 8px;font-size:28px} h2{font-size:20px;margin:0 0 10px}
    code{background:#eef2f7;border-radius:4px;padding:1px 5px}
    .table-wrap{overflow:auto;border:1px solid #d8e0ea;border-radius:8px}
    table{border-collapse:collapse;min-width:1120px;width:100%;background:#fff}
    th,td{border-bottom:1px solid #d8e0ea;border-right:1px solid #edf1f5;
      padding:7px 8px;text-align:left;vertical-align:top}
    th{background:#eef3f9;white-space:nowrap}.num{text-align:right;font-variant-numeric:tabular-nums}
    .warn{background:#fff6e5;border-left:4px solid #8a5a12}.good{background:#e9f8f2;border-left:4px solid #176b52}
    """
    bootstrap_rows = [{'comparison': name, **value} for name, value in results['paired_bootstrap'].items()]
    split_overlap_rows = [{'pair': name, **value} for name, value in results['rsvg_split_leakage']['overlap'].items()]
    method_rows = [{'method': name, **summary} for name, summary in results['method_summaries'].items()]
    split_leakage_json = html.escape(json.dumps(results['rsvg_split_leakage']['splits'], ensure_ascii=False, indent=2))
    return f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>MRPD CPU 分析报告</title><style>{style}</style></head>
<body><main>
<h1>MRPD CPU 分析报告</h1>
<p>该报告由 <code>tools/analysis/mrpd_cpu_analysis.py</code> 基于已有 per-sample eval 输出离线生成，不涉及模型推理。</p>
<section><h2>方法总体指标复核</h2>{html_table(method_rows)}</section>
<section><h2>Paired Bootstrap 显著性</h2>
<p>mean_delta = other - base。p 值为 bootstrap 双侧近似，主要用于论文阶段判断差异是否稳定。</p>
{html_table(bootstrap_rows)}</section>
<section><h2>RSVG-DIOR Split 泄漏检查</h2>
{html_table(split_overlap_rows)}<pre>{split_leakage_json}</pre></section>
<section><h2>面积分桶</h2>{html_table(results["bucket_analysis"]["area"])}</section>
<section><h2>长宽比分桶</h2>{html_table(results["bucket_analysis"]["aspect"])}</section>
<section><h2>边缘目标分桶</h2>{html_table(results["bucket_analysis"]["edge"])}</section>
<section><h2>类别分桶 Top 40</h2>{html_table(results["bucket_analysis"]["category"], max_rows=40)}</section>
</main></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, default=REPO_ROOT / 'MRPD' / 'cpu_analysis')
    parser.add_argument('--bootstrap-iters', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20260527)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    methods: Dict[str, Dict[str, Dict[str, Any]]] = {}
    missing: Dict[str, str] = {}
    for name, path in DEFAULT_METHODS.items():
        if path.exists():
            methods[name] = load_method_file(path)
        else:
            missing[name] = str(path)

    method_summaries = {name: metric_summary(rows.values()) for name, rows in methods.items()}

    comparisons = {
        'sft_vs_grpo_only': ('sft', 'grpo_only'),
        'sft_vs_mrpd_metric': ('sft', 'mrpd_metric'),
        'grpo_only_vs_mrpd_metric': ('grpo_only', 'mrpd_metric'),
        'grpo_only_vs_mrpd_ema098': ('grpo_only', 'mrpd_ema098'),
        'mrpd_metric_vs_mrpd_ema098': ('mrpd_metric', 'mrpd_ema098'),
        'mrpd_no_refresh_vs_mrpd_ema098': ('mrpd_no_refresh', 'mrpd_ema098'),
        'mrpd_step80_vs_mrpd_ema098': ('mrpd_step80', 'mrpd_ema098'),
    }
    bootstrap = {}
    for name, (base, other) in comparisons.items():
        if base in methods and other in methods:
            bootstrap[name] = paired_bootstrap(methods[base], methods[other], args.bootstrap_iters, args.seed)

    analysis_methods = {
        key: methods[key]
        for key in ('sft', 'grpo_only', 'mrpd_metric', 'mrpd_ema098') if key in methods
    }
    bucket_analysis = {
        'area':
        grouped_metric_table(analysis_methods, 'sft', lambda row: area_bucket(gt_box(row)) if gt_box(row) else None),
        'aspect':
        grouped_metric_table(analysis_methods, 'sft', lambda row: aspect_bucket(gt_box(row)) if gt_box(row) else None),
        'edge':
        grouped_metric_table(analysis_methods, 'sft', lambda row: edge_bucket(gt_box(row)) if gt_box(row) else None),
        'category':
        grouped_metric_table(analysis_methods, 'sft', category, min_count=20),
    }

    rsvg_paths = {
        'train': DATA_ROOT / 'rsvg_dior_full' / 'rsvg_train.jsonl',
        'val': DATA_ROOT / 'rsvg_dior_full' / 'rsvg_val.jsonl',
        'test': DATA_ROOT / 'rsvg_dior_full' / 'rsvg_test.jsonl',
    }
    rsvg_report = split_leakage_report(rsvg_paths)

    results = {
        'missing_inputs': missing,
        'method_summaries': method_summaries,
        'paired_bootstrap': bootstrap,
        'bucket_analysis': bucket_analysis,
        'rsvg_split_leakage': rsvg_report,
        'config': {
            'bootstrap_iters': args.bootstrap_iters,
            'seed': args.seed,
            'method_files': {k: str(v)
                             for k, v in DEFAULT_METHODS.items()},
        },
    }

    (args.out_dir / 'mrpd_cpu_analysis.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    write_csv(args.out_dir / 'method_summaries.csv', [{'method': k, **v} for k, v in method_summaries.items()])
    write_csv(args.out_dir / 'paired_bootstrap.csv', [{'comparison': k, **v} for k, v in bootstrap.items()])
    for name, rows in bucket_analysis.items():
        write_csv(args.out_dir / f'bucket_{name}.csv', rows)
    (args.out_dir / 'mrpd_cpu_analysis.html').write_text(build_report_html(results), encoding='utf-8')

    print(
        json.dumps(
            {
                'out_dir': str(args.out_dir),
                'missing_inputs': missing,
                'num_methods': len(methods),
                'num_bootstrap_comparisons': len(bootstrap),
                'rsvg_overlap': rsvg_report['overlap'],
            },
            ensure_ascii=False,
            indent=2))


if __name__ == '__main__':
    main()
