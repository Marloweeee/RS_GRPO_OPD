#!/usr/bin/env bash
set -euo pipefail

cd /data/codes/gui_grounding/GUI-SD-code-main

PYTHON_BIN=${PYTHON_BIN:-/data/codes/gui_grounding/conda_envs/GUI-SD/bin/python}
MODEL=${MODEL:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-4b_student_rs_full_legacy/v0-20260517-230632/checkpoint-80}
JSONL=${JSONL:-/data/codes/gui_grounding/data/rs_sub_dir/rs_train.jsonl}
OUT_DIR=${OUT_DIR:-/data/codes/gui_grounding/data/rs_sub_dir/stage1_rollout_pool_legacy_ckpt80_train}

export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-10000}
export PYTHONPATH=/data/codes/gui_grounding/GUI-SD-code-main:${PYTHONPATH:-}

echo "[stage1-rjob] cwd=$PWD"
echo "[stage1-rjob] python=$PYTHON_BIN"
echo "[stage1-rjob] model=$MODEL"
echo "[stage1-rjob] jsonl=$JSONL"
echo "[stage1-rjob] out_dir=$OUT_DIR"

"$PYTHON_BIN" -m py_compile tools/rollout/stage1_generate_rollout_pool.py

"$PYTHON_BIN" tools/rollout/stage1_generate_rollout_pool.py \
    --model "$MODEL" \
    --jsonl "$JSONL" \
    --out_dir "$OUT_DIR" \
    --max_samples "${MAX_SAMPLES:-0}" \
    --tp "${TP:-2}" \
    --num_rollouts "${NUM_ROLLOUTS:-8}" \
    --rollout_temperature "${ROLLOUT_TEMPERATURE:-0.7}" \
    --top_p "${TOP_P:-0.95}" \
    --max_model_len "${MAX_MODEL_LEN:-12000}" \
    --gpu_mem_util "${GPU_MEM_UTIL:-0.75}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-128}" \
    --seed "${SEED:-42}" \
    --tau_good "${TAU_GOOD:-0.5}" \
    --tau_fail "${TAU_FAIL:-0.3}" \
    --delta "${ROUTE_DELTA:-0.5}"

"$PYTHON_BIN" - <<PY
import json

summary_path = "${OUT_DIR}/summary.json"
with open(summary_path) as f:
    metrics = json.load(f)

print("[stage1-rjob] summary_path=" + summary_path)
print("[stage1-rjob] rollout_pool=${OUT_DIR}/rollout_pool.jsonl")
print("[stage1-rjob] greedy_IoU@0.5=%.4f best_of_8_IoU@0.5=%.4f gap=%.4f" % (
    metrics["greedy_IoU@0.5_rollout0"],
    metrics["best_of_k_IoU@0.5"],
    metrics["oracle_gap_IoU@0.5"],
))
print("[stage1-rjob] routes=" + json.dumps(metrics["route_ratio"], sort_keys=True))
print("[stage1-rjob] correction=" + json.dumps(metrics["correction_source_ratio"], sort_keys=True))
PY
