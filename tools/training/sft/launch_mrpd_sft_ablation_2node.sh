#!/usr/bin/env bash
# Launch one MRPD SFT-initialized two-node ablation experiment.
#
# Usage:
#   bash tools/training/sft/launch_mrpd_sft_ablation_2node.sh <experiment>
#
# Experiments:
#   sft_grpo_only
#   sft_opsd_only
#   sft_grpo_opsd_zoom
#   sft_grpo_opsd_jitter
#   sft_grpo_opsd_soft_window
#   sft_grpo_opsd_no_hint
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "$repo_root"

experiment="${1:-${EXPERIMENT:-}}"
if [ -z "$experiment" ]; then
    echo "ERROR: experiment is required." >&2
    exit 2
fi

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
source_ckpt="${SOURCE_CKPT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sft/gui-sd-qwen3-4b-rrsisd_sft_e1-20260526-220915/v0-20260526-220853/checkpoint-96}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs}"
mkdir -p "${log_root}/training"

export RUN_TIMESTAMP="$timestamp"
export RJOB_GROUP="${RJOB_GROUP:-gui_agent}"
export RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-gui_agent}"
export RJOB_CPU="${RJOB_CPU:-64}"
export RJOB_GPU="${RJOB_GPU:-8}"
export RJOB_MEMORY="${RJOB_MEMORY:-700000}"
export RJOB_REPLICA="${RJOB_REPLICA:-2}"
export RJOB_REPLICA_RESTART="${RJOB_REPLICA_RESTART:-never}"
export RJOB_BACKOFF_LIMIT="${RJOB_BACKOFF_LIMIT:-1}"
export RJOB_MAX_WAIT_DURATION="${RJOB_MAX_WAIT_DURATION:-18h0m0s}"
export RJOB_POSITIVE_TAGS="${RJOB_POSITIVE_TAGS:-feature/gpfs=yes}"

export BASE_MODEL_PATH="$source_ckpt"
export MODEL_PATH="$source_ckpt"
export TEACHER_PATH="$source_ckpt"
export ROLLOUT_MODEL_PATH="$source_ckpt"
export CKPT_ROOT="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd}"
export TRAIN_JSONL="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}"
export TEST_JSONL="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"

export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-7}"
export MASTER_PORT="${MASTER_PORT:-30120}"
export VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8892}"
export ROLLOUT_READY_TIMEOUT_SEC="${ROLLOUT_READY_TIMEOUT_SEC:-900}"
export RENDEZVOUS_TIMEOUT_SEC="${RENDEZVOUS_TIMEOUT_SEC:-1200}"

export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export SAVE_STEPS="${SAVE_STEPS:-10}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
export TUNER_TYPE="${TUNER_TYPE:-full}"
export ALLOW_RESUME="${ALLOW_RESUME:-false}"
export AUTO_RESUME="${AUTO_RESUME:-false}"
export SKIP_EVAL="${SKIP_EVAL:-true}"
export STAGE2_SKIP_EVAL="${STAGE2_SKIP_EVAL:-true}"

export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export LR="${LR:-1e-6}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.01}"
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
export SDPO_TEACHER_REFRESH_MODE="${SDPO_TEACHER_REFRESH_MODE:-metric}"
export SDPO_TEACHER_REFRESH_STEP="${SDPO_TEACHER_REFRESH_STEP:--1}"
export SDPO_TEACHER_REFRESH_WARMUP="${SDPO_TEACHER_REFRESH_WARMUP:-80}"
export SDPO_TEACHER_REFRESH_WINDOW="${SDPO_TEACHER_REFRESH_WINDOW:-50}"
export SDPO_TEACHER_REFRESH_CHECK_INTERVAL="${SDPO_TEACHER_REFRESH_CHECK_INTERVAL:-10}"
export SDPO_TEACHER_REFRESH_MIN_IOU_IMPROVE="${SDPO_TEACHER_REFRESH_MIN_IOU_IMPROVE:-0.01}"
export SDPO_TEACHER_REFRESH_MAX_FAILED="${SDPO_TEACHER_REFRESH_MAX_FAILED:-0.55}"
export SDPO_TEACHER_REFRESH_MIN_SDPO_LOSS="${SDPO_TEACHER_REFRESH_MIN_SDPO_LOSS:-0.02}"
export SDPO_TEACHER_REFRESH_MAX_KL="${SDPO_TEACHER_REFRESH_MAX_KL:-0.30}"

export OPSD_HINT_MODE="${OPSD_HINT_MODE:-hint}"
export OPSD_TOKEN_WEIGHT_MODE="${OPSD_TOKEN_WEIGHT_MODE:-uniform-entropy}"
export OPSD_NON_DIGIT_WEIGHT="${OPSD_NON_DIGIT_WEIGHT:-0.05}"
export OPSD_MAX_DIGIT_LEN="${OPSD_MAX_DIGIT_LEN:-3}"
export OPSD_EMA_DECAY="${OPSD_EMA_DECAY:-0.0}"
export OPSD_ZOOM_RATIO="${OPSD_ZOOM_RATIO:-2.0}"
export OPSD_MIN_AREA_FRAC="${OPSD_MIN_AREA_FRAC:-0.1}"
export OPSD_GAUSSIAN_SIGMA_RATIO="${OPSD_GAUSSIAN_SIGMA_RATIO:-1.5}"
export OPSD_HINT_BOX_COLOR="${OPSD_HINT_BOX_COLOR:-magenta}"
export OPSD_JITTER_RATIO="${OPSD_JITTER_RATIO:-0.2}"

export GRPO_BETA="${GRPO_BETA:-0.04}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-50}"
export VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.82}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-20000}"
export COMPLETION_GRACE_SEC="${COMPLETION_GRACE_SEC:-60}"

case "$experiment" in
    sft_grpo_only)
        export EXPERIMENT_SLUG="sft-grpo-only"
        export RUN_NAME="${RUN_NAME:-gui-sd-qwen3-4b-rrsisd_sft_grpo_only_e1-${timestamp}}"
        export RJOB_NAME="${RJOB_NAME:-q4b-sft-grpo-only-e1-${timestamp}}"
        export WORKER_SCRIPT="${WORKER_SCRIPT:-${repo_root}/tools/training/run_qwen3_4b_base_rsfull_grpo_only_2node_worker.sh}"
        export COMPLETION_GREP_PATTERN="${COMPLETION_GREP_PATTERN:-\\[4b-grpo-only-2node-worker\\].*completed}"
        ;;
    sft_opsd_only)
        export EXPERIMENT_SLUG="sft-opsd-only-gaussian"
        export RUN_NAME="${RUN_NAME:-gui-sd-qwen3-4b-rrsisd_sft_opsd_only_gaussian_e1-${timestamp}}"
        export RJOB_NAME="${RJOB_NAME:-q4b-sft-opsd-only-gauss-e1-${timestamp}}"
        export WORKER_SCRIPT="${WORKER_SCRIPT:-${repo_root}/tools/training/run_qwen3_4b_base_rsfull_opd_only_2node_worker.sh}"
        export OPSD_MASK_MODE="${OPSD_MASK_MODE:-gaussian}"
        export OPSD_EMA_DECAY="${OPSD_EMA_DECAY:-0.0}"
        export COMPLETION_GREP_PATTERN="${COMPLETION_GREP_PATTERN:-\\[4b-opd-only-2node-worker\\].*completed}"
        ;;
    sft_grpo_opsd_zoom)
        export EXPERIMENT_SLUG="sft-grpo-opsd-zoom"
        export RUN_NAME="${RUN_NAME:-gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_metric_zoom_in_e1-${timestamp}}"
        export RJOB_NAME="${RJOB_NAME:-q4b-sftopsd-zoom-e1-${timestamp}}"
        export WORKER_SCRIPT="${WORKER_SCRIPT:-${repo_root}/tools/training/run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh}"
        export OPSD_MASK_MODE="${OPSD_MASK_MODE:-zoom_in}"
        export COMPLETION_GREP_PATTERN="${COMPLETION_GREP_PATTERN:-\\[4b-2node-worker\\].*completed}"
        ;;
    sft_grpo_opsd_jitter)
        export EXPERIMENT_SLUG="sft-grpo-opsd-jitter"
        export RUN_NAME="${RUN_NAME:-gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_metric_jitter_box_e1-${timestamp}}"
        export RJOB_NAME="${RJOB_NAME:-q4b-sftopsd-jitter-e1-${timestamp}}"
        export WORKER_SCRIPT="${WORKER_SCRIPT:-${repo_root}/tools/training/run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh}"
        export OPSD_MASK_MODE="${OPSD_MASK_MODE:-jitter_box}"
        export COMPLETION_GREP_PATTERN="${COMPLETION_GREP_PATTERN:-\\[4b-2node-worker\\].*completed}"
        ;;
    sft_grpo_opsd_soft_window)
        export EXPERIMENT_SLUG="sft-grpo-opsd-soft-window"
        export RUN_NAME="${RUN_NAME:-gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_metric_soft_window_e1-${timestamp}}"
        export RJOB_NAME="${RJOB_NAME:-q4b-sftopsd-soft-window-e1-${timestamp}}"
        export WORKER_SCRIPT="${WORKER_SCRIPT:-${repo_root}/tools/training/run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh}"
        export OPSD_MASK_MODE="${OPSD_MASK_MODE:-soft_window}"
        export COMPLETION_GREP_PATTERN="${COMPLETION_GREP_PATTERN:-\\[4b-2node-worker\\].*completed}"
        ;;
    sft_grpo_opsd_no_hint)
        export EXPERIMENT_SLUG="sft-grpo-opsd-no-hint"
        export RUN_NAME="${RUN_NAME:-gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_metric_no_hint_e1-${timestamp}}"
        export RJOB_NAME="${RJOB_NAME:-q4b-sftopsd-nohint-e1-${timestamp}}"
        export WORKER_SCRIPT="${WORKER_SCRIPT:-${repo_root}/tools/training/run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh}"
        export OPSD_MASK_MODE="${OPSD_MASK_MODE:-no_mask}"
        export OPSD_HINT_MODE="${OPSD_HINT_MODE:-none}"
        export COMPLETION_GREP_PATTERN="${COMPLETION_GREP_PATTERN:-\\[4b-2node-worker\\].*completed}"
        ;;
    *)
        echo "ERROR: unknown experiment: ${experiment}" >&2
        exit 2
        ;;
esac

export RJOB_LOG_PATH="${RJOB_LOG_PATH:-${log_root}/training/${RJOB_NAME}.rjob.log}"

echo "[mrpd-sft-ablation] experiment=${experiment}"
echo "[mrpd-sft-ablation] timestamp=${timestamp}"
echo "[mrpd-sft-ablation] source_ckpt=${source_ckpt}"
echo "[mrpd-sft-ablation] run_name=${RUN_NAME}"
echo "[mrpd-sft-ablation] rjob_name=${RJOB_NAME}"
echo "[mrpd-sft-ablation] worker=${WORKER_SCRIPT}"
echo "[mrpd-sft-ablation] opsd_mask_mode=${OPSD_MASK_MODE:-none}"
echo "[mrpd-sft-ablation] train_log=${RJOB_LOG_PATH}"

bash "${repo_root}/tools/training/launch_qwen3_4b_rsfull_grpo_sdpo_2node_experiment_rjob.sh"
