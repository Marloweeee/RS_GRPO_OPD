#!/usr/bin/env bash
# Two-node Qwen3-VL-4B mid-training teacher/ref refresh worker.
#
# Semantics:
#   1. Train from the base 4B checkpoint to REFRESH_STEP (default: 80).
#   2. Save a full trainer checkpoint at that step.
#   3. Restart the worker from that checkpoint with student/ref/teacher all
#      pointing to the REFRESH_STEP weights, and resume trainer state so data
#      order, optimizer, scheduler, and global_step continue instead of
#      restarting the full dataset.
set -euo pipefail

repo_root="${REPO_ROOT:-/data/codes/gui_grounding/GUI-SD-code-main}"
cd "$repo_root"

env_root="${GUI_SD_ENV_ROOT:-/data/codes/gui_grounding/conda_envs/GUI-SD}"
export PATH="${env_root}/bin:${PATH}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHON_BIN="${PYTHON_BIN:-${env_root}/bin/python}"

worker_script="${TWO_NODE_WORKER_SCRIPT:-${repo_root}/tools/training/run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh}"
base_run_name="${RUN_NAME:-gui-sd-qwen3-4b-base_rsfull_grpo_sdpo_midrefresh_s80_e1}"

ckpt_root="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"
artifact_root="${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd}"
base_model_path="${BASE_MODEL_PATH:-/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-4B-Instruct}"
train_jsonl="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}"
test_jsonl="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"

refresh_step="${REFRESH_STEP:-80}"
refresh_marker_dir="${artifact_root}/midrefresh_markers/${base_run_name}"
refresh_ckpt_file="${refresh_marker_dir}/refresh_step_${refresh_step}_ckpt.txt"

mkdir -p "$refresh_marker_dir"

log() {
    echo "[4b-midrefresh-worker][$(date '+%F %T')][host=$(hostname)][rank=${NODE_RANK:-?}] $*"
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

find_checkpoint_for_step() {
    local run_name="$1"
    local step="$2"
    find "${ckpt_root}/${run_name}" -path "*/checkpoint-${step}" -type d -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -1 \
        | cut -d' ' -f2- || true
}

checkpoint_has_trainer_state() {
    local ckpt="$1"
    [ -f "${ckpt}/trainer_state.json" ] || return 1
    find "$ckpt" -maxdepth 3 \( \
        -name 'optimizer.pt' -o \
        -name 'scheduler.pt' -o \
        -name '*optim_states.pt' -o \
        -name 'global_step*' \
    \) -print -quit 2>/dev/null | grep -q .
}

compute_total_max_steps() {
    local requested="${REFRESH_TOTAL_MAX_STEPS:-${STAGE2_MAX_STEPS:-auto}}"
    if [ -n "$requested" ] && [ "$requested" != "auto" ] && [ "$requested" != "-1" ]; then
        printf '%s\n' "$requested"
        return 0
    fi

    local line_count world_size total_batch steps
    line_count="$(wc -l < "$train_jsonl" | tr -d ' ')"
    world_size=$((NNODES * NPROC_PER_NODE))
    total_batch=$((world_size * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
    steps=$((line_count / total_batch))
    if [ "$steps" -lt 1 ]; then
        steps=1
    fi
    printf '%s\n' "$steps"
}

wait_for_refresh_marker() {
    local start now
    start="$(date +%s)"
    while [ ! -f "$refresh_ckpt_file" ]; do
        now="$(date +%s)"
        if [ $((now - start)) -gt "${REFRESH_CKPT_WAIT_TIMEOUT_SEC:-900}" ]; then
            echo "ERROR: timed out waiting for refresh checkpoint marker: ${refresh_ckpt_file}" >&2
            exit 1
        fi
        log "waiting for refresh checkpoint marker"
        sleep 5
    done
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
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export LR="${LR:-2e-6}"

total_max_steps="$(compute_total_max_steps)"
if [ "$total_max_steps" -le "$refresh_step" ]; then
    echo "ERROR: total_max_steps=${total_max_steps} must be greater than refresh_step=${refresh_step}" >&2
    exit 1
fi

log "base_run_name=${base_run_name}"
log "refresh_step=${refresh_step}"
log "total_max_steps=${total_max_steps}"
log "ckpt_root=${ckpt_root}"
log "artifact_root=${artifact_root}"
log "base_model_path=${base_model_path}"
log "worker_script=${worker_script}"
log "refresh_ckpt_file=${refresh_ckpt_file}"

if [ ! -x "$worker_script" ]; then
    chmod +x "$worker_script"
fi

if [ "$NODE_RANK" = "0" ]; then
    rm -f "$refresh_ckpt_file" "${refresh_ckpt_file}.tmp"
fi

log "starting Stage 1: base 4B -> checkpoint-${refresh_step} with full trainer state"
RUN_NAME="$base_run_name" \
RENDEZVOUS_ID="${base_run_name}_stage1_s${refresh_step}" \
MODEL_PATH="$base_model_path" \
TEACHER_PATH="$base_model_path" \
MAX_STEPS="$refresh_step" \
NUM_TRAIN_EPOCHS="${STAGE1_NUM_TRAIN_EPOCHS:-1}" \
SAVE_STEPS="${STAGE1_SAVE_STEPS:-40}" \
SAVE_TOTAL_LIMIT="${STAGE1_SAVE_TOTAL_LIMIT:-4}" \
SAVE_ONLY_MODEL=false \
ALLOW_RESUME=false \
AUTO_RESUME=false \
WARMUP_RATIO="${STAGE1_WARMUP_RATIO:-${WARMUP_RATIO:-0.03}}" \
LR="${STAGE1_LR:-$LR}" \
PER_DEVICE_TRAIN_BATCH_SIZE="${STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE:-$PER_DEVICE_TRAIN_BATCH_SIZE}" \
GRADIENT_ACCUMULATION_STEPS="${STAGE1_GRADIENT_ACCUMULATION_STEPS:-$GRADIENT_ACCUMULATION_STEPS}" \
OPSD_MASK_DIR="${artifact_root}/train_cache/${base_run_name}_stage1_s${refresh_step}" \
EVAL_OUT="${artifact_root}/eval/${base_run_name}_stage1_s${refresh_step}" \
SKIP_EVAL=true \
bash "$worker_script"

if [ "$NODE_RANK" = "0" ]; then
    refresh_ckpt="$(find_checkpoint_for_step "$base_run_name" "$refresh_step")"
    if [ -z "$refresh_ckpt" ]; then
        echo "ERROR: Stage 1 did not produce checkpoint-${refresh_step} under ${ckpt_root}/${base_run_name}" >&2
        exit 1
    fi
    if ! checkpoint_has_trainer_state "$refresh_ckpt"; then
        echo "ERROR: ${refresh_ckpt} does not look like a full trainer checkpoint; cannot safely resume data/optimizer state." >&2
        find "$refresh_ckpt" -maxdepth 2 -printf '%p\n' 2>/dev/null | head -80 >&2 || true
        exit 1
    fi
    printf '%s\n' "$refresh_ckpt" > "${refresh_ckpt_file}.tmp"
    mv "${refresh_ckpt_file}.tmp" "$refresh_ckpt_file"
    sync "$refresh_ckpt_file" 2>/dev/null || sync 2>/dev/null || true
    log "refresh_ckpt=${refresh_ckpt}"
fi

wait_for_refresh_marker
refresh_ckpt="$(cat "$refresh_ckpt_file")"
if [ ! -d "$refresh_ckpt" ]; then
    echo "ERROR: refresh checkpoint marker points to a missing directory: ${refresh_ckpt}" >&2
    exit 1
fi
if ! checkpoint_has_trainer_state "$refresh_ckpt"; then
    echo "ERROR: refresh checkpoint is missing trainer state on this replica: ${refresh_ckpt}" >&2
    exit 1
fi

log "starting Stage 2: resume from ${refresh_ckpt}; student/ref/teacher refreshed to checkpoint-${refresh_step}"
RUN_NAME="$base_run_name" \
RENDEZVOUS_ID="${base_run_name}_stage2_from_s${refresh_step}_continue" \
MODEL_PATH="$refresh_ckpt" \
TEACHER_PATH="${STAGE2_TEACHER_PATH:-$refresh_ckpt}" \
RESUME_FROM_CHECKPOINT="$refresh_ckpt" \
RESUME_ONLY_MODEL=false \
ALLOW_RESUME=true \
AUTO_RESUME=false \
MAX_STEPS="$total_max_steps" \
NUM_TRAIN_EPOCHS="${STAGE2_NUM_TRAIN_EPOCHS:-1}" \
SAVE_STEPS="${STAGE2_SAVE_STEPS:-250}" \
SAVE_TOTAL_LIMIT="${STAGE2_SAVE_TOTAL_LIMIT:-8}" \
SAVE_ONLY_MODEL="${STAGE2_SAVE_ONLY_MODEL:-true}" \
WARMUP_RATIO="${STAGE2_WARMUP_RATIO:-${WARMUP_RATIO:-0.03}}" \
LR="${STAGE2_LR:-$LR}" \
PER_DEVICE_TRAIN_BATCH_SIZE="${STAGE2_PER_DEVICE_TRAIN_BATCH_SIZE:-$PER_DEVICE_TRAIN_BATCH_SIZE}" \
GRADIENT_ACCUMULATION_STEPS="${STAGE2_GRADIENT_ACCUMULATION_STEPS:-$GRADIENT_ACCUMULATION_STEPS}" \
OPSD_MASK_DIR="${artifact_root}/train_cache/${base_run_name}_stage2_from_s${refresh_step}" \
EVAL_OUT="${artifact_root}/eval/${base_run_name}" \
SKIP_EVAL="${STAGE2_SKIP_EVAL:-false}" \
bash "$worker_script"

log "mid-refresh run completed"
