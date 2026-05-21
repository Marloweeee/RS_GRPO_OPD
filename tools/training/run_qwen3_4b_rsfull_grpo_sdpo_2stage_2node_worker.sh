#!/usr/bin/env bash
# Two-stage Qwen3-VL-4B full-data GRPO+SDPO worker.
#
# Stage 1: train from the base 4B checkpoint for 80 steps.
# Stage 2: load the latest Stage-1 checkpoint as student/ref/teacher, then train
#          on the full training set for one epoch and evaluate the final model.
#
# This wrapper is intended to run inside a two-replica brainctl launch job. It
# delegates the actual multi-node rendezvous/training to
# run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh, while giving each stage an
# isolated run name and rendezvous id.
set -euo pipefail

repo_root="${REPO_ROOT:-/data/codes/gui_grounding/GUI-SD-code-main}"
cd "$repo_root"

env_root="${GUI_SD_ENV_ROOT:-/data/codes/gui_grounding/conda_envs/GUI-SD}"
export PATH="${env_root}/bin:${PATH}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHON_BIN="${PYTHON_BIN:-${env_root}/bin/python}"

worker_script="${TWO_NODE_WORKER_SCRIPT:-${repo_root}/tools/training/run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh}"
base_run_name="${RUN_NAME:-gui-sd-qwen3-4b-base_rsfull_grpo_sdpo_2stage}"
stage1_run_name="${STAGE1_RUN_NAME:-${base_run_name}_stage1_s80}"
stage2_run_name="${STAGE2_RUN_NAME:-${base_run_name}_stage2_from_s80_e1}"

ckpt_root="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"
artifact_root="${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd}"
base_model_path="${BASE_MODEL_PATH:-/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-4B-Instruct}"
train_jsonl="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}"
test_jsonl="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
stage1_marker_dir="${ARTIFACT_ROOT:-$artifact_root}/two_stage_markers/${base_run_name}"
stage1_ckpt_file="${stage1_marker_dir}/stage1_latest_ckpt.txt"

mkdir -p "$stage1_marker_dir"

log() {
    echo "[4b-2stage-worker][$(date '+%F %T')][host=$(hostname)][rank=${NODE_RANK:-?}] $*"
}

find_latest_checkpoint_for_run() {
    local run_name="$1"
    local latest_v latest_ckpt
    latest_v="$(find "${ckpt_root}/${run_name}" -maxdepth 1 -type d -name 'v*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
    if [ -z "$latest_v" ]; then
        return 1
    fi
    latest_ckpt="$(find "$latest_v" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
    if [ -z "$latest_ckpt" ]; then
        return 1
    fi
    printf '%s\n' "$latest_ckpt"
}

infer_rank_for_wrapper() {
    if [ -n "${NODE_RANK:-}" ]; then
        printf '%s\n' "$NODE_RANK"
        return 0
    fi
    for key in REPLICA_INDEX REPLICA_RANK RANK_INDEX WORKER_INDEX POD_INDEX INDEX; do
        local value="${!key:-}"
        if [[ "$value" =~ ^[0-9]+$ ]]; then
            printf '%s\n' "$value"
            return 0
        fi
    done

    "$PYTHON_BIN" - "$NNODES" "$(hostname)" <<'PY'
import hashlib
import re
import sys

nnodes = int(sys.argv[1])
host = sys.argv[2]
tokens = re.split(r'[-_.]', host)
for token in reversed(tokens):
    if token.isdigit() and int(token) < nnodes:
        print(int(token))
        raise SystemExit(0)

suffix = tokens[-1]
for idx in range(nnodes):
    if hashlib.md5(str(idx).encode()).hexdigest().startswith(suffix):
        print(idx)
        raise SystemExit(0)

raise SystemExit(f"cannot infer NODE_RANK from hostname={host}; set NODE_RANK explicitly")
PY
}

export NNODES="${NNODES:-2}"
export NODE_RANK="$(infer_rank_for_wrapper)"
export NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
export CKPT_ROOT="$ckpt_root"
export CHECKPOINT_ROOT="$ckpt_root"
export ARTIFACT_ROOT="$artifact_root"
export BASE_MODEL_PATH="$base_model_path"
export TRAIN_JSONL="$train_jsonl"
export TEST_JSONL="$test_jsonl"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
export ALLOW_RESUME="${ALLOW_RESUME:-false}"
export AUTO_RESUME="${AUTO_RESUME:-false}"

log "base_run_name=${base_run_name}"
log "stage1_run_name=${stage1_run_name}"
log "stage2_run_name=${stage2_run_name}"
log "ckpt_root=${ckpt_root}"
log "artifact_root=${artifact_root}"
log "base_model_path=${base_model_path}"
log "worker_script=${worker_script}"

if [ ! -x "$worker_script" ]; then
    chmod +x "$worker_script"
fi

log "starting Stage 1: base 4B -> 80 steps"
RUN_NAME="$stage1_run_name" \
RENDEZVOUS_ID="${stage1_run_name}" \
MODEL_PATH="$base_model_path" \
TEACHER_PATH="$base_model_path" \
MAX_STEPS="${STAGE1_MAX_STEPS:-80}" \
NUM_TRAIN_EPOCHS="${STAGE1_NUM_TRAIN_EPOCHS:-1}" \
SAVE_STEPS="${STAGE1_SAVE_STEPS:-40}" \
SAVE_TOTAL_LIMIT="${STAGE1_SAVE_TOTAL_LIMIT:-4}" \
WARMUP_RATIO="${STAGE1_WARMUP_RATIO:-0.03}" \
LR="${STAGE1_LR:-${LR:-2e-6}}" \
PER_DEVICE_TRAIN_BATCH_SIZE="${STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE:-${PER_DEVICE_TRAIN_BATCH_SIZE:-4}}" \
GRADIENT_ACCUMULATION_STEPS="${STAGE1_GRADIENT_ACCUMULATION_STEPS:-${GRADIENT_ACCUMULATION_STEPS:-2}}" \
OPSD_MASK_DIR="${artifact_root}/train_cache/${stage1_run_name}" \
EVAL_OUT="${artifact_root}/eval/${stage1_run_name}" \
SKIP_EVAL=true \
bash "$worker_script"

if [ "$NODE_RANK" = "0" ]; then
    stage1_ckpt="$(find_latest_checkpoint_for_run "$stage1_run_name")" || {
        echo "ERROR: Stage 1 produced no checkpoint under ${ckpt_root}/${stage1_run_name}" >&2
        exit 1
    }
    printf '%s\n' "$stage1_ckpt" > "${stage1_ckpt_file}.tmp"
    mv "${stage1_ckpt_file}.tmp" "$stage1_ckpt_file"
    sync "$stage1_ckpt_file" 2>/dev/null || sync 2>/dev/null || true
    log "stage1_latest_ckpt=${stage1_ckpt}"
fi

start_ts="$(date +%s)"
while [ ! -f "$stage1_ckpt_file" ]; do
    now_ts="$(date +%s)"
    if [ $((now_ts - start_ts)) -gt "${STAGE1_CKPT_WAIT_TIMEOUT_SEC:-600}" ]; then
        echo "ERROR: timed out waiting for ${stage1_ckpt_file}" >&2
        exit 1
    fi
    log "waiting for Stage 1 checkpoint marker"
    sleep 5
done
stage1_ckpt="$(cat "$stage1_ckpt_file")"
if [ ! -d "$stage1_ckpt" ]; then
    echo "ERROR: Stage 1 checkpoint marker points to missing directory: ${stage1_ckpt}" >&2
    exit 1
fi

log "starting Stage 2: ${stage1_ckpt} -> full-data training"
RUN_NAME="$stage2_run_name" \
RENDEZVOUS_ID="${stage2_run_name}" \
MODEL_PATH="$stage1_ckpt" \
TEACHER_PATH="${STAGE2_TEACHER_PATH:-$stage1_ckpt}" \
MAX_STEPS="${STAGE2_MAX_STEPS:--1}" \
NUM_TRAIN_EPOCHS="${STAGE2_NUM_TRAIN_EPOCHS:-1}" \
SAVE_STEPS="${STAGE2_SAVE_STEPS:-250}" \
SAVE_TOTAL_LIMIT="${STAGE2_SAVE_TOTAL_LIMIT:-8}" \
WARMUP_RATIO="${STAGE2_WARMUP_RATIO:-0.03}" \
LR="${STAGE2_LR:-${LR:-2e-6}}" \
PER_DEVICE_TRAIN_BATCH_SIZE="${STAGE2_PER_DEVICE_TRAIN_BATCH_SIZE:-${PER_DEVICE_TRAIN_BATCH_SIZE:-4}}" \
GRADIENT_ACCUMULATION_STEPS="${STAGE2_GRADIENT_ACCUMULATION_STEPS:-${GRADIENT_ACCUMULATION_STEPS:-2}}" \
OPSD_MASK_DIR="${artifact_root}/train_cache/${stage2_run_name}" \
EVAL_OUT="${artifact_root}/eval/${stage2_run_name}" \
SKIP_EVAL="${STAGE2_SKIP_EVAL:-false}" \
bash "$worker_script"

log "two-stage run completed"
