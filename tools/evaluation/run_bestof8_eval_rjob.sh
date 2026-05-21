#!/usr/bin/env bash
set -euo pipefail

cd /data/codes/gui_grounding/GUI-SD-code-main

PYTHON_BIN=${PYTHON_BIN:-/data/codes/gui_grounding/conda_envs/GUI-SD/bin/python}
MODEL=${MODEL:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-4b_student_rs_full_legacy/v0-20260517-230632/checkpoint-80}
TEST_JSONL=${TEST_JSONL:-/data/codes/gui_grounding/data/rs_sub_dir/rs_test.jsonl}
OUT_DIR=${OUT_DIR:-/data/codes/gui_grounding/data/rs_sub_dir/eval_bestof8_legacy_ckpt80}

export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-10000}
export PYTHONPATH=/data/codes/gui_grounding/GUI-SD-code-main:${PYTHONPATH:-}

echo "[bestof8] cwd=$PWD"
echo "[bestof8] python=$PYTHON_BIN"
echo "[bestof8] model=$MODEL"
echo "[bestof8] test_jsonl=$TEST_JSONL"
echo "[bestof8] out_dir=$OUT_DIR"

"$PYTHON_BIN" -m py_compile tools/evaluation/eval_student.py

"$PYTHON_BIN" tools/evaluation/eval_student.py \
    --model "$MODEL" \
    --test_jsonl "$TEST_JSONL" \
    --out_dir "$OUT_DIR" \
    --tp "${TP:-1}" \
    --num_rollouts "${NUM_ROLLOUTS:-8}" \
    --rollout_temperature "${ROLLOUT_TEMPERATURE:-0.7}" \
    --top_p "${TOP_P:-0.95}" \
    --max_model_len "${MAX_MODEL_LEN:-12000}" \
    --gpu_mem_util "${GPU_MEM_UTIL:-0.85}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-128}" \
    --seed "${SEED:-42}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-0}"

"$PYTHON_BIN" - <<PY
import json

summary_path = "${OUT_DIR}/summary.json"
with open(summary_path) as f:
    metrics = json.load(f)

print("[bestof8] summary_path=" + summary_path)
print("[bestof8] IoU@0.5=%.4f" % metrics["IoU@0.5"])
print("[bestof8] mIoU=%.4f IoU@0.7=%.4f parse_rate=%.4f valid_rate=%.4f" % (
    metrics["mIoU"],
    metrics["IoU@0.7"],
    metrics["parse_rate"],
    metrics["valid_rate"],
))
PY
