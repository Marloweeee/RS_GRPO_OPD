#!/usr/bin/env bash
# Qwen3-VL-8B RRSIS-D two-stage training:
#   1) SFT on rs_full/rs_train.jsonl
#   2) GRPO+OPSD initialized from the latest SFT checkpoint
#
# The script is intentionally sequential so stage 2 starts immediately after
# stage 1 produces a usable checkpoint.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
model_path="${BASE_MODEL_PATH:-/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-8B-Instruct}"
train_jsonl="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}"
ckpt_root_sft="${SFT_CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sft}"
artifact_root_sft="${SFT_ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sft}"
ckpt_root_stage2="${STAGE2_CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"
artifact_root_stage2="${STAGE2_ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs/training}"
mkdir -p "$log_root"

sft_run_name="${SFT_RUN_NAME:-gui-sd-qwen3-8b-rrsisd_sft_e1-${timestamp}}"
sft_rjob_name="${SFT_RJOB_NAME:-q8b-rrsisd-sft-2n-${timestamp}}"
sft_log="${SFT_RJOB_LOG_PATH:-${log_root}/${sft_rjob_name}.rjob.log}"

stage2_run_name="${STAGE2_RUN_NAME:-gui-sd-qwen3-8b-rrsisd_sft_grpo_opsd_metric_gaussian_e1-${timestamp}}"
stage2_rjob_name="${STAGE2_RJOB_NAME:-q8b-sftopsd-gauss-e1-${timestamp}}"
stage2_log="${STAGE2_RJOB_LOG_PATH:-${log_root}/${stage2_rjob_name}.rjob.log}"

find_latest_ckpt() {
    local root="$1"
    local run_name="$2"
    local latest_v latest_ckpt
    latest_v="$(
        find "${root}/${run_name}" -maxdepth 1 -type d -name 'v*' -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr | head -1 | cut -d' ' -f2- || true
    )"
    if [ -z "$latest_v" ]; then
        return 1
    fi
    latest_ckpt="$(
        find "$latest_v" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f %p\n' 2>/dev/null \
            | awk '{step=$1; sub(/^checkpoint-/, "", step); if (step ~ /^[0-9]+$/) print step " " $2}' \
            | sort -n | tail -1 | cut -d' ' -f2- || true
    )"
    if [ -z "$latest_ckpt" ]; then
        return 1
    fi
    printf '%s\n' "$latest_ckpt"
}

echo "[q8b-two-stage] timestamp=${timestamp}"
echo "[q8b-two-stage] model=${model_path}"
echo "[q8b-two-stage] train_jsonl=${train_jsonl}"
echo "[q8b-two-stage] sft_run_name=${sft_run_name}"
echo "[q8b-two-stage] stage2_run_name=${stage2_run_name}"

RUN_TIMESTAMP="$timestamp" \
EXPERIMENT_SLUG="rrsisd-8b-sft" \
RUN_NAME="$sft_run_name" \
RJOB_NAME="$sft_rjob_name" \
RJOB_LOG_PATH="$sft_log" \
RJOB_GROUP="${RJOB_GROUP:-gui_agent}" \
RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-gui_agent}" \
RJOB_CPU="${RJOB_CPU:-64}" \
RJOB_MEMORY="${RJOB_MEMORY:-800000}" \
RJOB_GPU="${RJOB_GPU:-8}" \
RJOB_REPLICA="${RJOB_REPLICA:-2}" \
BASE_MODEL_PATH="$model_path" \
MODEL_PATH="$model_path" \
CKPT_ROOT="$ckpt_root_sft" \
ARTIFACT_ROOT="$artifact_root_sft" \
TRAIN_JSONL="$train_jsonl" \
NNODES="${NNODES:-2}" \
NPROC_PER_NODE="${SFT_NPROC_PER_NODE:-8}" \
TRAIN_CUDA_VISIBLE_DEVICES="${SFT_TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
MASTER_PORT="${SFT_MASTER_PORT:-31280}" \
NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-1}" \
MAX_STEPS="${SFT_MAX_STEPS:--1}" \
SAVE_STEPS="${SFT_SAVE_STEPS:-25}" \
SAVE_TOTAL_LIMIT="${SFT_SAVE_TOTAL_LIMIT:-8}" \
PER_DEVICE_TRAIN_BATCH_SIZE="${SFT_PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-8}" \
LR="${SFT_LR:-8e-6}" \
WARMUP_RATIO="${SFT_WARMUP_RATIO:-0.03}" \
MAX_LENGTH="${MAX_LENGTH:-20000}" \
DEEPSPEED_CONFIG="${SFT_DEEPSPEED_CONFIG:-zero2}" \
bash "${script_dir}/launch_qwen3_8b_rrsisd_sft_2node_gui_agent.sh"

sft_ckpt="$(find_latest_ckpt "$ckpt_root_sft" "$sft_run_name")" || {
    echo "ERROR: SFT finished but no checkpoint found under ${ckpt_root_sft}/${sft_run_name}" >&2
    exit 1
}
echo "[q8b-two-stage] sft_latest_ckpt=${sft_ckpt}"

RUN_TIMESTAMP="$timestamp" \
EXPERIMENT_SLUG="q8b-sft-grpo-opsd-metric-gaussian" \
RUN_NAME="$stage2_run_name" \
RJOB_NAME="$stage2_rjob_name" \
RJOB_LOG_PATH="$stage2_log" \
WORKER_SCRIPT="${repo_root}/tools/training/run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh" \
RJOB_GROUP="${RJOB_GROUP:-gui_agent}" \
RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-gui_agent}" \
RJOB_CPU="${RJOB_CPU:-64}" \
RJOB_MEMORY="${RJOB_MEMORY:-800000}" \
RJOB_GPU="${RJOB_GPU:-8}" \
RJOB_REPLICA="${RJOB_REPLICA:-2}" \
COMPLETION_GREP_PATTERN="${COMPLETION_GREP_PATTERN:-\\[4b-2node-worker\\].*completed}" \
COMPLETION_GRACE_SEC="${COMPLETION_GRACE_SEC:-60}" \
BASE_MODEL_PATH="$sft_ckpt" \
MODEL_PATH="$sft_ckpt" \
TEACHER_PATH="$sft_ckpt" \
ROLLOUT_MODEL_PATH="$sft_ckpt" \
CKPT_ROOT="$ckpt_root_stage2" \
ARTIFACT_ROOT="$artifact_root_stage2" \
TRAIN_JSONL="$train_jsonl" \
TEST_JSONL="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}" \
NNODES="${NNODES:-2}" \
NPROC_PER_NODE="${STAGE2_NPROC_PER_NODE:-7}" \
TRAIN_CUDA_VISIBLE_DEVICES="${STAGE2_TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}" \
ROLLOUT_CUDA_VISIBLE_DEVICES="${STAGE2_ROLLOUT_CUDA_VISIBLE_DEVICES:-7}" \
MASTER_PORT="${STAGE2_MASTER_PORT:-31320}" \
VLLM_SERVER_PORT="${STAGE2_VLLM_SERVER_PORT:-9132}" \
NUM_TRAIN_EPOCHS="${STAGE2_NUM_TRAIN_EPOCHS:-1}" \
MAX_STEPS="${STAGE2_MAX_STEPS:--1}" \
SAVE_STEPS="${STAGE2_SAVE_STEPS:-10}" \
SAVE_TOTAL_LIMIT="${STAGE2_SAVE_TOTAL_LIMIT:-20}" \
PER_DEVICE_TRAIN_BATCH_SIZE="${STAGE2_PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
GRADIENT_ACCUMULATION_STEPS="${STAGE2_GRADIENT_ACCUMULATION_STEPS:-4}" \
LR="${STAGE2_LR:-8e-7}" \
WARMUP_RATIO="${STAGE2_WARMUP_RATIO:-0.01}" \
NUM_GENERATIONS="${NUM_GENERATIONS:-8}" \
NUM_ITERATIONS="${NUM_ITERATIONS:-1}" \
MAX_LENGTH="${MAX_LENGTH:-20000}" \
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-64}" \
DEEPSPEED_CONFIG="${STAGE2_DEEPSPEED_CONFIG:-zero2}" \
TEACHER_DEEPSPEED_CONFIG="${TEACHER_DEEPSPEED_CONFIG:-zero3}" \
OFFLOAD_TEACHER_MODEL="${OFFLOAD_TEACHER_MODEL:-false}" \
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.78}" \
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-20000}" \
SDPO_LAMBDA="${SDPO_LAMBDA:-0.25}" \
SDPO_TAU_GOOD="${SDPO_TAU_GOOD:-0.5}" \
SDPO_TAU_FAIL="${SDPO_TAU_FAIL:-0.3}" \
SDPO_DELTA="${SDPO_DELTA:-0.5}" \
SDPO_DISTILL_SCOPE="${SDPO_DISTILL_SCOPE:-failed}" \
SDPO_TARGET="${SDPO_TARGET:-rollout}" \
SDPO_HINT_SOURCE="${SDPO_HINT_SOURCE:-gt}" \
SDPO_TEACHER_REFRESH_MODE="${SDPO_TEACHER_REFRESH_MODE:-metric}" \
SDPO_TEACHER_REFRESH_STEP="${SDPO_TEACHER_REFRESH_STEP:--1}" \
SDPO_TEACHER_REFRESH_WARMUP="${SDPO_TEACHER_REFRESH_WARMUP:-80}" \
SDPO_TEACHER_REFRESH_WINDOW="${SDPO_TEACHER_REFRESH_WINDOW:-50}" \
SDPO_TEACHER_REFRESH_CHECK_INTERVAL="${SDPO_TEACHER_REFRESH_CHECK_INTERVAL:-10}" \
OPSD_MASK_MODE="${OPSD_MASK_MODE:-gaussian}" \
OPSD_HINT_MODE="${OPSD_HINT_MODE:-hint}" \
OPSD_TOKEN_WEIGHT_MODE="${OPSD_TOKEN_WEIGHT_MODE:-uniform-entropy}" \
OPSD_NON_DIGIT_WEIGHT="${OPSD_NON_DIGIT_WEIGHT:-0.05}" \
OPSD_MAX_DIGIT_LEN="${OPSD_MAX_DIGIT_LEN:-3}" \
OPSD_EMA_DECAY="${OPSD_EMA_DECAY:-0.0}" \
GRPO_BETA="${GRPO_BETA:-0.04}" \
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}" \
TOP_P="${TOP_P:-0.95}" \
TOP_K="${TOP_K:-50}" \
SKIP_EVAL=true \
STAGE2_SKIP_EVAL=true \
bash "${repo_root}/tools/training/launch_qwen3_4b_rsfull_grpo_sdpo_2node_experiment_rjob.sh"

stage2_ckpt="$(find_latest_ckpt "$ckpt_root_stage2" "$stage2_run_name")" || {
    echo "ERROR: stage2 finished but no checkpoint found under ${ckpt_root_stage2}/${stage2_run_name}" >&2
    exit 1
}
echo "[q8b-two-stage] stage2_latest_ckpt=${stage2_ckpt}"
echo "[q8b-two-stage] completed"
