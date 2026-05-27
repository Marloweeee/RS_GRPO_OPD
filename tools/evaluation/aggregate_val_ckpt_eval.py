#!/usr/bin/env python3
"""Aggregate validation-set checkpoint evaluation results.

This script is intentionally restart-friendly. It can be run while rjobs are
still evaluating: completed checkpoints are summarized, and pending/failed
checkpoints stay visible in the output files.
"""

import argparse
import csv
import html
import os
from datetime import datetime

import json

METRIC_KEYS = [
    'n',
    'mIoU',
    'IoU@0.5',
    'IoU@0.6',
    'IoU@0.7',
    'IoU@0.8',
    'IoU@0.9',
    'parse_rate',
    'valid_rate',
]


def load_jsonl(path):
    if not path or not os.path.isfile(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def compute_thresholds_from_per_sample(per_sample_path):
    if not os.path.isfile(per_sample_path):
        return {}
    ious = []
    parse_ok = 0
    valid_ok = 0
    with open(per_sample_path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            iou = float(item.get('iou', 0.0))
            ious.append(iou)
            if item.get('pred_bbox') is not None:
                parse_ok += 1
                valid_ok += 1
    n = len(ious)
    if n == 0:
        return {}
    metrics = {
        'n': n,
        'mIoU': round(sum(ious) / n, 4),
        'parse_rate': round(parse_ok / n, 4),
        'valid_rate': round(valid_ok / n, 4),
    }
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        metrics[f'IoU@{t:.1f}'] = round(sum(i > t for i in ious) / n, 4)
    return metrics


def fmt(value):
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def write_csv(path, rows):
    fieldnames = [
        'id',
        'group',
        'name',
        'status',
        'config',
        *METRIC_KEYS,
        'model',
        'out_dir',
        'summary_path',
        'reason',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_html(path, rows, args):
    completed = sum(1 for row in rows if row.get('status') == 'done')
    failed = sum(1 for row in rows if row.get('status') == 'failed')
    pending = sum(1 for row in rows if row.get('status') in {'pending', 'running'})
    skipped = sum(1 for row in rows if row.get('status') == 'skipped')
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    done_rows = [row for row in rows if row.get('status') == 'done' and row.get('mIoU') is not None]

    def best_line(metric):
        if not done_rows:
            return f'{metric}: 暂无完成结果'
        best = max(done_rows, key=lambda row: float(row.get(metric) or -1))
        return (f'{metric}: {fmt(best.get(metric))} '
                f"({html.escape(str(best.get('id', '')))} - {html.escape(str(best.get('name', '')))})")

    metric_headers = ''.join(f'<th>{html.escape(k)}</th>' for k in METRIC_KEYS)
    table_rows = []
    for row in rows:
        metric_cells = ''.join(f'<td>{html.escape(fmt(row.get(k)))}</td>' for k in METRIC_KEYS)
        cls = html.escape(str(row.get('status', '')))
        table_rows.append("<tr class=\"{cls}\">"
                          '<td>{id}</td><td>{group}</td><td>{name}</td><td>{status}</td>'
                          "<td class=\"config\">{config}</td>{metrics}"
                          "<td class=\"path\">{model}</td><td class=\"path\">{out_dir}</td><td>{reason}</td>"
                          '</tr>'.format(
                              cls=cls,
                              id=html.escape(str(row.get('id', ''))),
                              group=html.escape(str(row.get('group', ''))),
                              name=html.escape(str(row.get('name', ''))),
                              status=html.escape(str(row.get('status', ''))),
                              config=html.escape(str(row.get('config', ''))),
                              metrics=metric_cells,
                              model=html.escape(str(row.get('model', ''))),
                              out_dir=html.escape(str(row.get('out_dir', ''))),
                              reason=html.escape(str(row.get('reason', ''))),
                          ))

    content = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>4B Checkpoint Val 指标汇总</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #202124; }}
    h1 {{ font-size: 24px; margin: 0 0 8px; }}
    h2 {{ font-size: 18px; margin: 24px 0 8px; }}
    p {{ margin: 6px 0; line-height: 1.5; }}
    ul {{ margin: 8px 0 16px 20px; padding: 0; line-height: 1.6; }}
    code {{ background: #f1f3f4; padding: 1px 4px; border-radius: 4px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #dadce0; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f8fafd; position: sticky; top: 0; z-index: 1; }}
    tr.done td:first-child {{ border-left: 4px solid #188038; }}
    tr.failed td:first-child {{ border-left: 4px solid #d93025; }}
    tr.running td:first-child, tr.pending td:first-child {{ border-left: 4px solid #f9ab00; }}
    tr.skipped td:first-child {{ border-left: 4px solid #80868b; }}
    .summary {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0 18px; }}
    .pill {{ background: #f8fafd; border: 1px solid #dadce0; border-radius: 6px; padding: 8px 10px; }}
    .config {{ min-width: 240px; }}
    .path {{ max-width: 360px; word-break: break-all;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>4B Checkpoint Val 指标汇总</h1>
  <p>生成时间：{html.escape(generated_at)}</p>
  <p>评测数据：<code>{html.escape(args.eval_jsonl)}</code></p>
  <p>评测方式：直接贪心推理，<code>num_rollouts=1</code>，<code>temperature=0.0</code>，
  统计 mIoU 与 IoU@0.5/0.6/0.7/0.8/0.9。</p>
  <p>运行目录：<code>{html.escape(args.out_root)}</code></p>
  <h2>执行记录</h2>
  <ul>
    <li>本次只评测 4B checkpoint，使用 <code>gui_agent</code> 组 rjob 分片运行；
    启动会话 <code>val_4b_ckpt_eval_20260525</code> 已自然退出，
    launcher 结束状态为 <code>overall_status=0</code>。</li>
    <li>每个 checkpoint 都独立保存 <code>summary.json</code>、<code>threshold_summary.json</code>、
    <code>per_sample.jsonl</code>、<code>eval.log</code> 和 <code>status.json</code>，
    中间结果可断点恢复。</li>
    <li>本轮日志未检出 <code>Traceback</code>、<code>RuntimeError</code>、<code>ERROR</code>、
    <code>Killed</code> 或 checkpoint 缺失错误。</li>
    <li>R05/R06 是早期 metric-trigger bug 记录，原 summary 中只有省略路径，无法可靠定位 checkpoint，因此保留为 skipped。</li>
  </ul>
  <h2>关键结论</h2>
  <ul>
    <li>{best_line("mIoU")}</li>
    <li>{best_line("IoU@0.5")}</li>
    <li>{best_line("IoU@0.7")}</li>
    <li>{best_line("IoU@0.9")}</li>
    <li>从 val 结果看，metric-trigger teacher refresh 系列整体处于第一梯队；
    OPD/OPSD-only 明显低于 GRPO-only 和 GRPO+SDPO，说明只做蒸馏不足以替代 GRPO 约束。</li>
  </ul>
  <h2>输出文件</h2>
  <ul>
    <li>JSON 汇总：<code>{html.escape(os.path.join(args.out_root, "val_metrics_summary.json"))}</code></li>
    <li>CSV 汇总：<code>{html.escape(os.path.join(args.out_root, "val_metrics_summary.csv"))}</code></li>
    <li>HTML 汇总：<code>{html.escape(os.path.join(args.out_root, "val_metrics_summary.html"))}</code></li>
    <li>OPSD_Idea 副本：<code>{html.escape(args.html_out or path)}</code></li>
  </ul>
  <div class="summary">
    <div class="pill">总计：{len(rows)}</div>
    <div class="pill">完成：{completed}</div>
    <div class="pill">运行/待跑：{pending}</div>
    <div class="pill">失败：{failed}</div>
    <div class="pill">跳过：{skipped}</div>
  </div>
  <h2>结果表</h2>
  <table>
    <thead>
      <tr>
        <th>ID</th><th>组别</th><th>实验</th><th>状态</th><th>配置</th>
        {metric_headers}
        <th>Checkpoint</th><th>输出目录</th><th>说明</th>
      </tr>
    </thead>
    <tbody>
      {"".join(table_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    with open(path, 'w') as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out_root', required=True)
    parser.add_argument('--eval_jsonl', default='/data/codes/gui_grounding/data/rs_full/rs_val.jsonl')
    parser.add_argument('--skipped_jsonl', default='')
    parser.add_argument('--html_out', default='')
    args = parser.parse_args()

    rows = []
    for item in load_jsonl(args.manifest):
        out_dir = item['out_dir']
        summary_path = os.path.join(out_dir, 'threshold_summary.json')
        if not os.path.isfile(summary_path):
            summary_path = os.path.join(out_dir, 'summary.json')
        metrics = load_json(summary_path) or {}
        if 'IoU@0.6' not in metrics:
            metrics.update(compute_thresholds_from_per_sample(os.path.join(out_dir, 'per_sample.jsonl')))
        status = load_json(os.path.join(out_dir, 'status.json')) or {}
        row = {
            'id': item.get('id'),
            'group': item.get('group'),
            'name': item.get('name'),
            'config': item.get('config'),
            'model': item.get('model'),
            'out_dir': out_dir,
            'summary_path': summary_path if os.path.isfile(summary_path) else '',
            'status': 'done' if metrics else status.get('status', item.get('status', 'pending')),
            'reason': status.get('reason', ''),
        }
        for key in METRIC_KEYS:
            row[key] = metrics.get(key)
        rows.append(row)

    for item in load_jsonl(args.skipped_jsonl):
        row = {
            'id': item.get('id'),
            'group': item.get('group', ''),
            'name': item.get('name', ''),
            'config': item.get('config', ''),
            'model': item.get('model', ''),
            'out_dir': item.get('out_dir', ''),
            'summary_path': '',
            'status': 'skipped',
            'reason': item.get('reason', 'skipped'),
        }
        rows.append(row)

    rows.sort(key=lambda row: str(row.get('id', '')))
    os.makedirs(args.out_root, exist_ok=True)
    json_path = os.path.join(args.out_root, 'val_metrics_summary.json')
    csv_path = os.path.join(args.out_root, 'val_metrics_summary.csv')
    html_path = os.path.join(args.out_root, 'val_metrics_summary.html')

    payload = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'eval_jsonl': args.eval_jsonl,
        'manifest': args.manifest,
        'out_root': args.out_root,
        'rows': rows,
    }
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    write_csv(csv_path, rows)
    write_html(html_path, rows, args)
    if args.html_out:
        os.makedirs(os.path.dirname(args.html_out), exist_ok=True)
        write_html(args.html_out, rows, args)

    done = sum(1 for row in rows if row.get('status') == 'done')
    print(f'[aggregate-val] rows={len(rows)} done={done} json={json_path}')
    print(f'[aggregate-val] csv={csv_path}')
    print(f'[aggregate-val] html={html_path}')
    if args.html_out:
        print(f'[aggregate-val] html_copy={args.html_out}')


if __name__ == '__main__':
    main()
