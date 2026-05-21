#!/usr/bin/env bash
set -euo pipefail

cd /data/codes/gui_grounding/GUI-SD-code-main

PYTHON_BIN=${PYTHON_BIN:-/data/codes/gui_grounding/conda_envs/GUI-SD/bin/python}
RUN_ROOT=${RUN_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-8b-base_rsfull_stage_refresh_s800_trefsync_r1}
TEST_JSONL=${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}
OUT_BASE=${OUT_BASE:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd/eval_latest_stage_refresh_test}
TP=${TP:-2}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.75}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-256}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-12000}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}

export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-10000}
export PYTHONPATH=/data/codes/gui_grounding/GUI-SD-code-main:${PYTHONPATH:-}
export PATH=/data/codes/gui_grounding/conda_envs/GUI-SD/bin:${PATH}

if [ ! -d "${RUN_ROOT}" ]; then
    echo "[latest-eval] ERROR: RUN_ROOT not found: ${RUN_ROOT}" >&2
    exit 1
fi

latest_ckpt=$(
    while IFS= read -r d; do
        name=$(basename "${d}")
        step=${name#checkpoint-}
        if [[ "${step}" =~ ^[0-9]+$ ]] && [ -f "${d}/config.json" ] && [ -f "${d}/model.safetensors.index.json" ]; then
            printf '%s %s\n' "${step}" "${d}"
        fi
    done < <(find "${RUN_ROOT}" -mindepth 2 -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null) \
        | sort -n \
        | tail -1 \
        | cut -d' ' -f2-
)

if [ -z "${latest_ckpt}" ]; then
    echo "[latest-eval] ERROR: no complete checkpoint found under ${RUN_ROOT}" >&2
    exit 1
fi

ckpt_tag="$(basename "$(dirname "${latest_ckpt}")")_$(basename "${latest_ckpt}")"
out_dir="${OUT_DIR:-${OUT_BASE}/${ckpt_tag}_greedy}"
mkdir -p "${out_dir}"

echo "[latest-eval] RUN_ROOT=${RUN_ROOT}"
echo "[latest-eval] latest_ckpt=${latest_ckpt}"
echo "[latest-eval] TEST_JSONL=${TEST_JSONL} ($(wc -l < "${TEST_JSONL}") samples)"
echo "[latest-eval] out_dir=${out_dir}"
echo "[latest-eval] PYTHON_BIN=${PYTHON_BIN}"
echo "[latest-eval] TP=${TP}"
echo "[latest-eval] GPU_MEM_UTIL=${GPU_MEM_UTIL}"
echo "[latest-eval] EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"

"${PYTHON_BIN}" tools/evaluation/eval_student.py \
    --model "${latest_ckpt}" \
    --test_jsonl "${TEST_JSONL}" \
    --out_dir "${out_dir}" \
    --tp "${TP}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --gpu_mem_util "${GPU_MEM_UTIL}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed 42

"${PYTHON_BIN}" - <<PY
import json
import os

summary_path = os.path.join("${out_dir}", "summary.json")
with open(summary_path) as f:
    metrics = json.load(f)
print("[latest-eval] summary_path=" + summary_path)
print("[latest-eval] model=" + metrics["model"])
print("[latest-eval] IoU@0.5=%.4f" % metrics["IoU@0.5"])
print("[latest-eval] mIoU=%.4f IoU@0.7=%.4f parse_rate=%.4f valid_rate=%.4f" % (
    metrics["mIoU"], metrics["IoU@0.7"], metrics["parse_rate"], metrics["valid_rate"]))
PY
