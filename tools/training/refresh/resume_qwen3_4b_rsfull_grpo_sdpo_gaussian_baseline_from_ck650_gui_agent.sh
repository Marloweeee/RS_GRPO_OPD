#!/usr/bin/env bash
# Resume the gaussian GRPO+SDPO no-refresh baseline from checkpoint-650.
#
# The first run stopped at step 656 because the vLLM rollout server closed the
# /update_flattened_params connection. This resume keeps the same baseline
# recipe, uses a new run/rendezvous id, and starts from the checkpoint-650
# weights plus trainer state.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
source_ckpt="${RESUME_FROM_CHECKPOINT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rsfull_grpo_sdpo_hint_gaussian_e1-20260524-120936/v0-20260524-121403/checkpoint-650}"
source_run="gui-sd-qwen3-4b-rsfull_grpo_sdpo_hint_gaussian_e1-20260524-120936"

export EXPERIMENT_SLUG="${EXPERIMENT_SLUG:-gaussian-baseline-resume650}"
export RUN_NAME="${RUN_NAME:-gui-sd-qwen3-4b-rsfull_grpo_sdpo_hint_gaussian_resume650_e1-${timestamp}}"
export RJOB_NAME="${RJOB_NAME:-q4b-sdpo-gaussian-resume650-2n-${timestamp}}"
export RJOB_LOG_PATH="${RJOB_LOG_PATH:-/data/codes/gui_grounding/data/logs/training/${RJOB_NAME}.rjob.log}"
export WORKER_SCRIPT="${WORKER_SCRIPT:-${repo_root}/tools/training/run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh}"

export RJOB_GROUP="${RJOB_GROUP:-gui_agent}"
export RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-gui_agent}"
export RJOB_CPU="${RJOB_CPU:-64}"
export RJOB_GPU="${RJOB_GPU:-8}"
export RJOB_MEMORY="${RJOB_MEMORY:-800000}"
export RJOB_REPLICA="${RJOB_REPLICA:-2}"
export RJOB_REPLICA_RESTART="${RJOB_REPLICA_RESTART:-never}"
export RJOB_BACKOFF_LIMIT="${RJOB_BACKOFF_LIMIT:-1}"
export RJOB_MAX_WAIT_DURATION="${RJOB_MAX_WAIT_DURATION:-18h0m0s}"
export RJOB_POSITIVE_TAGS="${RJOB_POSITIVE_TAGS:-feature/gpfs=yes}"

export BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-4B-Instruct}"
export MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH}"
export TEACHER_PATH="${TEACHER_PATH:-$BASE_MODEL_PATH}"
export ROLLOUT_MODEL_PATH="${ROLLOUT_MODEL_PATH:-$BASE_MODEL_PATH}"
export CKPT_ROOT="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd}"
export TRAIN_JSONL="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}"
export TEST_JSONL="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"

export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-7}"
export VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8692}"
export MASTER_PORT="${MASTER_PORT:-29900}"
export ROLLOUT_READY_TIMEOUT_SEC="${ROLLOUT_READY_TIMEOUT_SEC:-900}"
export RENDEZVOUS_TIMEOUT_SEC="${RENDEZVOUS_TIMEOUT_SEC:-1200}"
export RENDEZVOUS_ID="${RENDEZVOUS_ID:-${RUN_NAME}}"

export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export SAVE_STEPS="${SAVE_STEPS:-10}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
export TUNER_TYPE="${TUNER_TYPE:-full}"
export ALLOW_RESUME="${ALLOW_RESUME:-true}"
export AUTO_RESUME="${AUTO_RESUME:-false}"
export RESUME_FROM_CHECKPOINT="$source_ckpt"
export RESUME_ONLY_MODEL="${RESUME_ONLY_MODEL:-true}"
export SKIP_EVAL="${SKIP_EVAL:-true}"

export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export LR="${LR:-2e-6}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
export MAX_LENGTH="${MAX_LENGTH:-20000}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-64}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero2}"
export TEACHER_DEEPSPEED_CONFIG="${TEACHER_DEEPSPEED_CONFIG:-zero3}"
export OFFLOAD_TEACHER_MODEL="${OFFLOAD_TEACHER_MODEL:-false}"

export SDPO_LAMBDA="${SDPO_LAMBDA:-0.25}"
export SDPO_TAU_GOOD="${SDPO_TAU_GOOD:-0.5}"
export SDPO_TAU_FAIL="${SDPO_TAU_FAIL:-0.3}"
export SDPO_DELTA="${SDPO_DELTA:-0.5}"
export SDPO_TARGET="${SDPO_TARGET:-rollout}"
export SDPO_TEACHER_REFRESH_MODE="${SDPO_TEACHER_REFRESH_MODE:-fixed}"
export SDPO_TEACHER_REFRESH_STEP="${SDPO_TEACHER_REFRESH_STEP:--1}"
export OPSD_MASK_MODE="${OPSD_MASK_MODE:-gaussian}"
export OPSD_HINT_MODE="${OPSD_HINT_MODE:-hint}"
export OPSD_TOKEN_WEIGHT_MODE="${OPSD_TOKEN_WEIGHT_MODE:-uniform-entropy}"
export OPSD_NON_DIGIT_WEIGHT="${OPSD_NON_DIGIT_WEIGHT:-0.05}"
export OPSD_MAX_DIGIT_LEN="${OPSD_MAX_DIGIT_LEN:-3}"
export OPSD_ZOOM_RATIO="${OPSD_ZOOM_RATIO:-2.0}"
export OPSD_MIN_AREA_FRAC="${OPSD_MIN_AREA_FRAC:-0.1}"
export OPSD_GAUSSIAN_SIGMA_RATIO="${OPSD_GAUSSIAN_SIGMA_RATIO:-1.5}"
export OPSD_HINT_BOX_COLOR="${OPSD_HINT_BOX_COLOR:-magenta}"
export OPSD_JITTER_RATIO="${OPSD_JITTER_RATIO:-0.2}"

export GRPO_BETA="${GRPO_BETA:-0.04}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-50}"
export VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.70}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-20000}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"
unset VLLM_MAX_NUM_SEQS

export VLLM_CLIENT_POST_RETRIES="${VLLM_CLIENT_POST_RETRIES:-3}"
export VLLM_CLIENT_POST_RETRY_DELAY="${VLLM_CLIENT_POST_RETRY_DELAY:-2}"
export VLLM_CLIENT_POST_TIMEOUT="${VLLM_CLIENT_POST_TIMEOUT:-120}"

mkdir -p /data/codes/gui_grounding/data/logs/training
echo "[gaussian-baseline-resume650] source_run=${source_run}"
echo "[gaussian-baseline-resume650] source_ckpt=${source_ckpt}"
echo "[gaussian-baseline-resume650] run_name=${RUN_NAME}"
echo "[gaussian-baseline-resume650] rjob_name=${RJOB_NAME}"
echo "[gaussian-baseline-resume650] log=${RJOB_LOG_PATH}"

bash "${repo_root}/tools/training/launch_qwen3_4b_rsfull_grpo_sdpo_2node_experiment_rjob.sh"
