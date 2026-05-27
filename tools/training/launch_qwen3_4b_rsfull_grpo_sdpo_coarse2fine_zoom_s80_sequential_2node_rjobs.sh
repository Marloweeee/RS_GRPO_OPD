#!/usr/bin/env bash
# Coarse-to-fine zoom_in staged restart using two independent rjobs.
#
# Stage 1: base Qwen3-VL-4B -> 80 steps with gaussian hints.
# Stage 2: load Stage-1 checkpoint as student/ref/teacher, then restart full
#          rs_train training with zoom_in hints.
#
# Running each stage in a fresh rjob avoids the observed failure where a second
# training stage inside the same container can no longer see CUDA devices.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
base_run_name="${RUN_NAME:-gui-sd-qwen3-4b-rsfull_grpo_sdpo_coarse2fine_gaussian80_zoom_seq_e1-${timestamp}}"
stage1_run_name="${STAGE1_RUN_NAME:-${base_run_name}_stage1_gaussian_s80}"
stage2_run_name="${STAGE2_RUN_NAME:-${base_run_name}_stage2_zoom_from_s80_e1}"
base_rjob_name="${RJOB_NAME:-q4b-c2f-zs80-${timestamp}}"

worker_script="${TWO_NODE_WORKER_SCRIPT:-${script_dir}/run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh}"
launcher_script="${EXPERIMENT_LAUNCHER_SCRIPT:-${script_dir}/launch_qwen3_4b_rsfull_grpo_sdpo_2node_experiment_rjob.sh}"
ckpt_root="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"
base_model_path="${BASE_MODEL_PATH:-/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-4B-Instruct}"
log_dir="${RJOB_LOG_DIR:-/data/codes/gui_grounding/data/logs/training}"

mkdir -p "$log_dir"
chmod +x "$worker_script" "$launcher_script"

log() {
    echo "[coarse2fine-seq][$(date '+%F %T')] $*"
}

find_checkpoint_for_step() {
    local run_name="$1"
    local step="$2"
    local log_path="${3:-}"
    local checkpoint_path

    checkpoint_path="$(find "${ckpt_root}/${run_name}" -path "*/checkpoint-${step}" -type d -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -1 \
        | cut -d' ' -f2- || true)"
    if [ -n "$checkpoint_path" ]; then
        echo "$checkpoint_path"
        return 0
    fi

    # The local driver may not have /mnt/jfs mounted even though the rjob
    # workers do. In that case, trust the worker log path emitted after save.
    if [ -n "$log_path" ] && [ -f "$log_path" ]; then
        grep -oE "/mnt/jfs/[^[:space:]]*/checkpoint-${step}" "$log_path" \
            | tail -1 || true
    fi
}

run_stage() {
    local stage="$1"
    local run_name="$2"
    local rjob_name="$3"
    local mask_mode="$4"
    local model_path="$5"
    local teacher_path="$6"
    shift 6

    log "launching ${stage}: run_name=${run_name} rjob_name=${rjob_name} mask_mode=${mask_mode}"
    RUN_TIMESTAMP="$timestamp" \
    EXPERIMENT_SLUG="coarse2fine-${stage}" \
    RUN_NAME="$run_name" \
    RJOB_NAME="$rjob_name" \
    RJOB_LOG_PATH="${log_dir}/${rjob_name}.rjob.log" \
    WORKER_SCRIPT="$worker_script" \
    MODEL_PATH="$model_path" \
    TEACHER_PATH="$teacher_path" \
    OPSD_MASK_MODE="$mask_mode" \
    OPSD_HINT_MODE="${OPSD_HINT_MODE:-hint}" \
    SKIP_EVAL=true \
    STAGE2_SKIP_EVAL=true \
    COMPLETION_GREP_PATTERN="4b-2node-worker.*completed" \
    "$@" \
    bash "$launcher_script"
}

export RJOB_GROUP="${RJOB_GROUP:-gui_agent}"
export RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-${RJOB_GROUP}}"
export RJOB_CPU="${RJOB_CPU:-64}"
export RJOB_GPU="${RJOB_GPU:-8}"
export RJOB_MEMORY="${RJOB_MEMORY:-480000}"
export RJOB_POSITIVE_TAGS="${RJOB_POSITIVE_TAGS:-feature/gpfs=yes}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export LR="${LR:-2e-6}"
export SAVE_STEPS="${SAVE_STEPS:-10}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-12}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
export ALLOW_RESUME=false
export AUTO_RESUME=false
export VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.70}"
export ROLLOUT_READY_TIMEOUT_SEC="${ROLLOUT_READY_TIMEOUT_SEC:-900}"
export RENDEZVOUS_TIMEOUT_SEC="${RENDEZVOUS_TIMEOUT_SEC:-1200}"

log "base_run_name=${base_run_name}"
log "stage1_run_name=${stage1_run_name}"
log "stage2_run_name=${stage2_run_name}"
log "group=${RJOB_GROUP} cpu=${RJOB_CPU} gpu=${RJOB_GPU} memory=${RJOB_MEMORY}"
log "ckpt_root=${ckpt_root}"

run_stage \
    "stage1-gaussian-s80" \
    "$stage1_run_name" \
    "${base_rjob_name}-stage1" \
    "gaussian" \
    "$base_model_path" \
    "$base_model_path" \
    env \
        MAX_STEPS="${STAGE1_MAX_STEPS:-80}" \
        NUM_TRAIN_EPOCHS="${STAGE1_NUM_TRAIN_EPOCHS:-1}" \
        SAVE_STEPS="${STAGE1_SAVE_STEPS:-10}" \
        SAVE_TOTAL_LIMIT="${STAGE1_SAVE_TOTAL_LIMIT:-12}" \
        SAVE_ONLY_MODEL=true \
        VLLM_SERVER_PORT="${STAGE1_VLLM_SERVER_PORT:-8533}" \
        MASTER_PORT="${STAGE1_MASTER_PORT:-29533}"

stage1_log_path="${log_dir}/${base_rjob_name}-stage1.rjob.log"
stage1_ckpt="$(find_checkpoint_for_step "$stage1_run_name" "${STAGE1_MAX_STEPS:-80}" "$stage1_log_path")"
if [ -z "$stage1_ckpt" ]; then
    echo "ERROR: Stage 1 did not produce checkpoint-${STAGE1_MAX_STEPS:-80} under ${ckpt_root}/${stage1_run_name}" >&2
    exit 1
fi
log "stage1_ckpt=${stage1_ckpt}"

run_stage \
    "stage2-zoom-full" \
    "$stage2_run_name" \
    "${base_rjob_name}-stage2" \
    "zoom_in" \
    "$stage1_ckpt" \
    "$stage1_ckpt" \
    env \
        MAX_STEPS="${STAGE2_MAX_STEPS:--1}" \
        NUM_TRAIN_EPOCHS="${STAGE2_NUM_TRAIN_EPOCHS:-1}" \
        SAVE_STEPS="${STAGE2_SAVE_STEPS:-10}" \
        SAVE_TOTAL_LIMIT="${STAGE2_SAVE_TOTAL_LIMIT:-12}" \
        SAVE_ONLY_MODEL=true \
        VLLM_SERVER_PORT="${STAGE2_VLLM_SERVER_PORT:-8534}" \
        MASTER_PORT="${STAGE2_MASTER_PORT:-29534}"

log "coarse-to-fine sequential rjobs run completed"
