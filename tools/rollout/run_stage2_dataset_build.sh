#!/usr/bin/env bash
set -euo pipefail

cd /data/codes/gui_grounding/GUI-SD-code-main

PYTHON_BIN=${PYTHON_BIN:-/data/codes/gui_grounding/conda_envs/GUI-SD/bin/python}
POOL=${POOL:-/data/codes/gui_grounding/data/rs_sub_dir/stage1_rollout_pool_legacy_ckpt80_train/rollout_pool.jsonl}
OUT_DIR=${OUT_DIR:-/data/codes/gui_grounding/data/rs_sub_dir/stage2_routing_datasets_legacy_ckpt80_train}

export PYTHONPATH=/data/codes/gui_grounding/GUI-SD-code-main:${PYTHONPATH:-}

echo "[stage2-data] cwd=$PWD"
echo "[stage2-data] python=$PYTHON_BIN"
echo "[stage2-data] pool=$POOL"
echo "[stage2-data] out_dir=$OUT_DIR"

"$PYTHON_BIN" -m py_compile tools/rollout/stage2_build_routing_datasets.py
"$PYTHON_BIN" tools/rollout/stage2_build_routing_datasets.py \
    --pool "$POOL" \
    --out_dir "$OUT_DIR" \
    --tau_good "${TAU_GOOD:-0.5}" \
    --max_samples "${MAX_SAMPLES:-0}" \
    --smoke_count "${SMOKE_COUNT:-64}"

"$PYTHON_BIN" - <<PY
import json
import os

summary_path = "${OUT_DIR}/summary.json"
with open(summary_path) as f:
    summary = json.load(f)

print("[stage2-data] summary_path=" + summary_path)
print("[stage2-data] target_source_ratio=" + json.dumps(summary["target_source_ratio"], sort_keys=True))
print("[stage2-data] chosen_IoU@0.5=%.4f rejected_IoU@0.5=%.4f margin=%.4f" % (
    summary["chosen_IoU@0.5"],
    summary["rejected_IoU@0.5"],
    summary["dpo_iou_margin_mean"],
))
for name, path in summary["paths"].items():
    print(f"[stage2-data] {name}: {path} ({sum(1 for _ in open(path)) if os.path.isfile(path) else 0} rows)")
PY
