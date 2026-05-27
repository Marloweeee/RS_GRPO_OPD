#!/usr/bin/env bash
# Evaluate every N-step checkpoint of one training run on rs_test.jsonl.
# This script is intended to run inside a brainctl GPU container where
# /mnt/jfs/copilot is mounted.
set -euo pipefail

repo_root="${REPO_ROOT:-/data/codes/gui_grounding/GUI-SD-code-main}"
env_root="${GUI_SD_ENV_ROOT:-/data/codes/gui_grounding/conda_envs/GUI-SD}"
cd "$repo_root"

export PATH="${env_root}/bin:${PATH}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONFAULTHANDLER=1
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-10000}"
export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"

python_bin="${PYTHON_BIN:-${env_root}/bin/python}"
run_name="${RUN_NAME:?RUN_NAME is required}"
ckpt_root="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"
test_jsonl="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
out_root="${OUT_ROOT:-/data/codes/gui_grounding/data/logs/eval/${run_name}_test_every50}"
step_interval="${STEP_INTERVAL:-50}"

mkdir -p "$out_root"

echo "[test-every50] host=$(hostname)"
echo "[test-every50] cwd=$PWD"
echo "[test-every50] python=$python_bin"
echo "[test-every50] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[test-every50] run_name=$run_name"
echo "[test-every50] ckpt_root=$ckpt_root"
echo "[test-every50] test_jsonl=$test_jsonl"
echo "[test-every50] out_root=$out_root"
echo "[test-every50] step_interval=$step_interval"

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is empty" >&2
    exit 1
fi
if [ ! -f "$test_jsonl" ]; then
    echo "ERROR: test jsonl not found: $test_jsonl" >&2
    exit 1
fi

"$python_bin" -m py_compile tools/evaluation/eval_student.py

latest_v="$(
    find "${ckpt_root}/${run_name}" -maxdepth 1 -type d -name 'v*' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -1 \
        | cut -d' ' -f2- || true
)"
if [ -z "$latest_v" ]; then
    echo "ERROR: no v* checkpoint directory found under ${ckpt_root}/${run_name}" >&2
    find "${ckpt_root}/${run_name}" -maxdepth 2 -type d 2>/dev/null | sed -n '1,80p' >&2 || true
    exit 1
fi
echo "[test-every50] latest_v=$latest_v"

manifest="${out_root}/test_every50_manifest.jsonl"
summary_json="${out_root}/test_every50_summary.json"
summary_csv="${out_root}/test_every50_summary.csv"
summary_html="${out_root}/test_every50_summary.html"
rm -f "$manifest"

"$python_bin" - "$latest_v" "$out_root" "$manifest" "$step_interval" <<'PY'
import json
import os
import re
import sys

latest_v, out_root, manifest, step_interval = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = []
for name in os.listdir(latest_v):
    match = re.fullmatch(r"checkpoint-(\d+)", name)
    if not match:
        continue
    step = int(match.group(1))
    if step % step_interval != 0:
        continue
    model = os.path.join(latest_v, name)
    if os.path.isdir(model):
        rows.append((step, model))
rows.sort()
if not rows:
    raise SystemExit(f"no checkpoint-* directories divisible by {step_interval} found under {latest_v}")
with open(manifest, "w") as f:
    for step, model in rows:
        item = {
            "step": step,
            "id": f"step-{step:04d}",
            "model": model,
            "out_dir": os.path.join(out_root, f"checkpoint-{step}"),
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"[test-every50] manifest={manifest} records={len(rows)}")
PY

while IFS= read -r line; do
    [ -n "$line" ] || continue
    eval "$("$python_bin" - "$line" <<'PY'
import json
import shlex
import sys

item = json.loads(sys.argv[1])
for key in ("step", "id", "model", "out_dir"):
    print(f"{key.upper()}={shlex.quote(str(item[key]))}")
PY
)"
    mkdir -p "$OUT_DIR"
    status_json="${OUT_DIR}/status.json"
    eval_log="${OUT_DIR}/eval.log"
    summary_path="${OUT_DIR}/summary.json"
    threshold_path="${OUT_DIR}/threshold_summary.json"

    if [ -f "$summary_path" ] && [ -f "$threshold_path" ]; then
        echo "[test-every50][$ID] skip completed $summary_path"
        continue
    fi
    if [ ! -d "$MODEL" ]; then
        echo "{\"id\":\"$ID\",\"step\":$STEP,\"status\":\"failed\",\"reason\":\"checkpoint_not_found\",\"model\":\"$MODEL\"}" > "$status_json"
        echo "[test-every50][$ID] checkpoint not found: $MODEL" >&2
        continue
    fi

    cache_root="/tmp/guisd_test_eval_${ID}_$$"
    export XDG_CACHE_HOME="${cache_root}/xdg"
    export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
    export TRITON_CACHE_DIR="${cache_root}/triton"
    export VLLM_CACHE_ROOT="${cache_root}/vllm"
    mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

    echo "{\"id\":\"$ID\",\"step\":$STEP,\"status\":\"running\",\"model\":\"$MODEL\",\"started_at\":\"$(date '+%F %T')\"}" > "$status_json"
    echo "[test-every50][$ID] start model=$MODEL"
    set +e
    "$python_bin" tools/evaluation/eval_student.py \
        --model "$MODEL" \
        --test_jsonl "$test_jsonl" \
        --out_dir "$OUT_DIR" \
        --tp "${TP:-8}" \
        --max_model_len "${MAX_MODEL_LEN:-12000}" \
        --gpu_mem_util "${GPU_MEM_UTIL:-0.88}" \
        --max_new_tokens "${MAX_NEW_TOKENS:-128}" \
        --seed "${SEED:-42}" \
        --num_rollouts "${NUM_ROLLOUTS:-1}" \
        --rollout_temperature "${ROLLOUT_TEMPERATURE:-0.0}" \
        --top_p "${TOP_P:-0.95}" \
        --eval_batch_size "${EVAL_BATCH_SIZE:-512}" 2>&1 | tee "$eval_log"
    rc=${PIPESTATUS[0]}
    set -e
    rm -rf "$cache_root" 2>/dev/null || true

    if [ "$rc" -ne 0 ]; then
        echo "{\"id\":\"$ID\",\"step\":$STEP,\"status\":\"failed\",\"rc\":$rc,\"model\":\"$MODEL\",\"updated_at\":\"$(date '+%F %T')\"}" > "$status_json"
        echo "[test-every50][$ID] failed rc=$rc" >&2
        continue
    fi

    "$python_bin" - "$OUT_DIR" "$status_json" "$ID" "$STEP" "$MODEL" <<'PY'
import json
import os
import sys
from datetime import datetime

out_dir, status_path, item_id, step, model = sys.argv[1:6]
summary_path = os.path.join(out_dir, "summary.json")
per_sample_path = os.path.join(out_dir, "per_sample.jsonl")
with open(summary_path) as f:
    metrics = json.load(f)
ious = []
with open(per_sample_path) as f:
    for line in f:
        if line.strip():
            ious.append(float(json.loads(line)["iou"]))
for t in (0.5, 0.6, 0.7, 0.8, 0.9):
    metrics[f"IoU@{t:.1f}"] = round(sum(i > t for i in ious) / max(1, len(ious)), 4)
with open(summary_path, "w") as f:
    json.dump(metrics, f, indent=2)
with open(os.path.join(out_dir, "threshold_summary.json"), "w") as f:
    json.dump(metrics, f, indent=2)
status = {
    "id": item_id,
    "step": int(step),
    "status": "done",
    "model": model,
    "summary_path": summary_path,
    "metrics": {
        key: metrics.get(key)
        for key in ("n", "mIoU", "IoU@0.5", "IoU@0.6", "IoU@0.7", "IoU@0.8", "IoU@0.9", "parse_rate", "valid_rate")
    },
    "updated_at": datetime.now().strftime("%F %T"),
}
with open(status_path, "w") as f:
    json.dump(status, f, indent=2)
print("[test-every50] metrics " + json.dumps(status["metrics"], ensure_ascii=False))
PY
done < "$manifest"

"$python_bin" - "$manifest" "$summary_json" "$summary_csv" "$summary_html" "$run_name" "$test_jsonl" "$out_root" <<'PY'
import csv
import html
import json
import os
import sys
from datetime import datetime

manifest, summary_json, summary_csv, summary_html, run_name, test_jsonl, out_root = sys.argv[1:8]
metric_keys = ["n", "mIoU", "IoU@0.5", "IoU@0.6", "IoU@0.7", "IoU@0.8", "IoU@0.9", "parse_rate", "valid_rate"]
rows = []
with open(manifest) as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        status_path = os.path.join(item["out_dir"], "status.json")
        status = {}
        if os.path.isfile(status_path):
            with open(status_path) as sf:
                status = json.load(sf)
        row = {
            "id": item["id"],
            "step": item["step"],
            "status": status.get("status", "pending"),
            "model": item["model"],
            "out_dir": item["out_dir"],
            "summary_path": status.get("summary_path", ""),
            "reason": status.get("reason", ""),
        }
        row.update(status.get("metrics", {}))
        rows.append(row)
rows.sort(key=lambda r: int(r["step"]))
payload = {
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "run_name": run_name,
    "test_jsonl": test_jsonl,
    "out_root": out_root,
    "rows": rows,
}
with open(summary_json, "w") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "step", "status", *metric_keys, "model", "out_dir", "summary_path", "reason"])
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
done = [r for r in rows if r.get("status") == "done" and r.get("mIoU") is not None]
def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)
def best(metric):
    if not done:
        return f"{metric}: 暂无完成结果"
    row = max(done, key=lambda r: float(r.get(metric) or -1))
    return f"{metric}: {fmt(row.get(metric))} ({row['id']})"
body_rows = []
for row in rows:
    metric_cells = "".join(f"<td>{html.escape(fmt(row.get(k)))}</td>" for k in metric_keys)
    body_rows.append(
        f"<tr class='{html.escape(str(row.get('status', '')))}'>"
        f"<td>{html.escape(str(row['id']))}</td><td>{row['step']}</td><td>{html.escape(str(row.get('status', '')))}</td>"
        f"{metric_cells}<td class='path'>{html.escape(row['model'])}</td><td class='path'>{html.escape(row['out_dir'])}</td>"
        f"<td>{html.escape(str(row.get('reason', '')))}</td></tr>"
    )
metric_headers = "".join(f"<th>{html.escape(k)}</th>" for k in metric_keys)
content = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Test Every-50 Checkpoint Metrics</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #202124; }}
    h1 {{ font-size: 24px; margin: 0 0 8px; }}
    h2 {{ font-size: 18px; margin: 24px 0 8px; }}
    code {{ background: #f1f3f4; padding: 1px 4px; border-radius: 4px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #dadce0; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f8fafd; }}
    .path {{ max-width: 420px; word-break: break-all; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
    tr.done td:first-child {{ border-left: 4px solid #188038; }}
    tr.failed td:first-child {{ border-left: 4px solid #d93025; }}
  </style>
</head>
<body>
  <h1>Test Every-50 Checkpoint Metrics</h1>
  <p>生成时间：{html.escape(payload['generated_at'])}</p>
  <p>训练任务：<code>{html.escape(run_name)}</code></p>
  <p>评测数据：<code>{html.escape(test_jsonl)}</code></p>
  <p>输出目录：<code>{html.escape(out_root)}</code></p>
  <h2>最佳点</h2>
  <ul>
    <li>{html.escape(best('mIoU'))}</li>
    <li>{html.escape(best('IoU@0.5'))}</li>
    <li>{html.escape(best('IoU@0.7'))}</li>
    <li>{html.escape(best('IoU@0.9'))}</li>
  </ul>
  <h2>结果表</h2>
  <table>
    <thead><tr><th>ID</th><th>Step</th><th>Status</th>{metric_headers}<th>Checkpoint</th><th>输出目录</th><th>说明</th></tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</body>
</html>
"""
with open(summary_html, "w") as f:
    f.write(content)
print(f"[test-every50] summary_json={summary_json}")
print(f"[test-every50] summary_csv={summary_csv}")
print(f"[test-every50] summary_html={summary_html}")
for line in (best("mIoU"), best("IoU@0.5"), best("IoU@0.7"), best("IoU@0.9")):
    print("[test-every50] " + line)
PY

echo "[test-every50] completed"
