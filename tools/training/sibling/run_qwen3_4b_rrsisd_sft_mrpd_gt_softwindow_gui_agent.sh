#!/usr/bin/env bash
# Direct GT-hint MRPD comparison for the sibling-guided experiment.
#
# This keeps the same SFT initialization, rollout, soft-window teacher image,
# and refresh settings as the sibling run. The only intended difference is that
# teacher visual hints come directly from the oracle GT bbox instead of the best
# successful sibling trajectory.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
source_ckpt="${SOURCE_CKPT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sft/gui-sd-qwen3-4b-rrsisd_sft_e1-20260526-220915/v0-20260526-220853/checkpoint-96}"
experiment_slug="${EXPERIMENT_SLUG:-sft-mrpd-gt-hint-softwindow}"
run_name="${RUN_NAME:-gui-sd-qwen3-4b-rrsisd_sft_mrpd_gt_hint_softwindow_e1-${timestamp}}"
rjob_name="${RJOB_NAME:-q4b-sftmrpd-gt-sw-e1-${timestamp}}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs}"
train_log="${TRAIN_RJOB_LOG_PATH:-${log_root}/training/${rjob_name}.rjob.log}"
eval_root="${EVAL_ROOT:-${log_root}/eval/${run_name}_test}"
eval_rjob_name="${EVAL_RJOB_NAME:-q4b-gt-sw-test-${timestamp}}"
eval_log="${EVAL_RJOB_LOG_PATH:-${log_root}/eval/${eval_rjob_name}.submit.log}"
eval_worker_log="${EVAL_WORKER_LOG_PATH:-${log_root}/eval/${eval_rjob_name}.worker.log}"
ckpt_root="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"

mkdir -p "${log_root}/training" "${log_root}/eval" "$eval_root"

if [ ! -d "$source_ckpt" ]; then
    echo "WARNING: SOURCE_CKPT is not visible on the submit host: $source_ckpt" >&2
    echo "WARNING: continuing because the rjob container mounts /mnt/jfs/copilot." >&2
fi

export RUN_TIMESTAMP="$timestamp"
export EXPERIMENT_SLUG="$experiment_slug"
export RUN_NAME="$run_name"
export RJOB_NAME="$rjob_name"
export RJOB_LOG_PATH="$train_log"
export WORKER_SCRIPT="${WORKER_SCRIPT:-${repo_root}/tools/training/run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh}"

export RJOB_GROUP="${RJOB_GROUP:-gui_agent}"
export RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-gui_agent}"
export RJOB_CPU="${RJOB_CPU:-80}"
export RJOB_GPU="${RJOB_GPU:-8}"
export RJOB_MEMORY="${RJOB_MEMORY:-800000}"
export RJOB_REPLICA="${RJOB_REPLICA:-2}"
export RJOB_REPLICA_RESTART="${RJOB_REPLICA_RESTART:-never}"
export RJOB_BACKOFF_LIMIT="${RJOB_BACKOFF_LIMIT:-1}"
export RJOB_MAX_WAIT_DURATION="${RJOB_MAX_WAIT_DURATION:-18h0m0s}"
export RJOB_POSITIVE_TAGS="${RJOB_POSITIVE_TAGS:-feature/gpfs=yes}"
export CLEANUP_RJOB_ON_INTERRUPT="${CLEANUP_RJOB_ON_INTERRUPT:-true}"

export COMPLETION_GREP_PATTERN="${COMPLETION_GREP_PATTERN:-\\[4b-2node-worker\\].*completed}"
export COMPLETION_GRACE_SEC="${COMPLETION_GRACE_SEC:-60}"

export BASE_MODEL_PATH="$source_ckpt"
export MODEL_PATH="$source_ckpt"
export TEACHER_PATH="$source_ckpt"
export ROLLOUT_MODEL_PATH="$source_ckpt"
export CKPT_ROOT="$ckpt_root"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd}"
export TRAIN_JSONL="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}"
export TEST_JSONL="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"

export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-7}"
export VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8894}"
export MASTER_PORT="${MASTER_PORT:-30140}"
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
export SDPO_HINT_SOURCE="${SDPO_HINT_SOURCE:-gt}"
export SDPO_SIBLING_SELECT_METRIC="${SDPO_SIBLING_SELECT_METRIC:-reward}"
export SDPO_SIBLING_FALLBACK="${SDPO_SIBLING_FALLBACK:-gt}"
export SDPO_TEACHER_REFRESH_MODE="${SDPO_TEACHER_REFRESH_MODE:-metric_ewma}"
export SDPO_TEACHER_REFRESH_STEP="${SDPO_TEACHER_REFRESH_STEP:--1}"
export SDPO_TEACHER_REFRESH_WARMUP="${SDPO_TEACHER_REFRESH_WARMUP:-80}"
export SDPO_TEACHER_REFRESH_WINDOW="${SDPO_TEACHER_REFRESH_WINDOW:-50}"
export SDPO_TEACHER_REFRESH_CHECK_INTERVAL="${SDPO_TEACHER_REFRESH_CHECK_INTERVAL:-10}"
export SDPO_TEACHER_REFRESH_MIN_IOU_IMPROVE="${SDPO_TEACHER_REFRESH_MIN_IOU_IMPROVE:-0.01}"
export SDPO_TEACHER_REFRESH_MAX_FAILED="${SDPO_TEACHER_REFRESH_MAX_FAILED:-0.55}"
export SDPO_TEACHER_REFRESH_MIN_SDPO_LOSS="${SDPO_TEACHER_REFRESH_MIN_SDPO_LOSS:-0.02}"
export SDPO_TEACHER_REFRESH_MAX_KL="${SDPO_TEACHER_REFRESH_MAX_KL:-0.30}"
export SDPO_TEACHER_REFRESH_SHORT_WINDOW="${SDPO_TEACHER_REFRESH_SHORT_WINDOW:-20}"
export SDPO_TEACHER_REFRESH_LONG_WINDOW="${SDPO_TEACHER_REFRESH_LONG_WINDOW:-80}"
export SDPO_TEACHER_REFRESH_EWMA_ALPHA="${SDPO_TEACHER_REFRESH_EWMA_ALPHA:-0.10}"
export SDPO_TEACHER_REFRESH_CONSECUTIVE_CHECKS="${SDPO_TEACHER_REFRESH_CONSECUTIVE_CHECKS:-2}"
export SDPO_TEACHER_REFRESH_MIN_SHORT_LONG_IOU_GAIN="${SDPO_TEACHER_REFRESH_MIN_SHORT_LONG_IOU_GAIN:-0.006}"
export SDPO_TEACHER_REFRESH_MIN_EWMA_IOU_GAIN="${SDPO_TEACHER_REFRESH_MIN_EWMA_IOU_GAIN:-0.008}"
export SDPO_TEACHER_REFRESH_MAX_IOU05_DROP="${SDPO_TEACHER_REFRESH_MAX_IOU05_DROP:-0.010}"
export SDPO_TEACHER_REFRESH_MAX_REFRESHES="${SDPO_TEACHER_REFRESH_MAX_REFRESHES:-1}"
export SDPO_TEACHER_REFRESH_COOLDOWN_STEPS="${SDPO_TEACHER_REFRESH_COOLDOWN_STEPS:-0}"

export OPSD_MASK_MODE="${OPSD_MASK_MODE:-soft_window}"
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
unset VLLM_MAX_NUM_SEQS
unset VLLM_ENFORCE_EAGER

echo "[gt-softwindow] timestamp=${timestamp}"
echo "[gt-softwindow] source_ckpt=${source_ckpt}"
echo "[gt-softwindow] run_name=${run_name}"
echo "[gt-softwindow] train_log=${train_log}"
echo "[gt-softwindow] eval_root=${eval_root}"

bash "${repo_root}/tools/training/launch_qwen3_4b_rsfull_grpo_sdpo_2node_experiment_rjob.sh"

latest_ckpt=""
if [ -d "${ckpt_root}/${run_name}" ]; then
    latest_v="$(
        find "${ckpt_root}/${run_name}" -maxdepth 1 -type d -name 'v*' -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr | head -1 | cut -d' ' -f2- || true
    )"
    if [ -n "$latest_v" ]; then
        latest_ckpt="$(
            find "$latest_v" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f %p\n' 2>/dev/null \
                | awk '{step=$1; sub(/^checkpoint-/, "", step); if (step ~ /^[0-9]+$/) print step " " $2}' \
                | sort -n | tail -1 | cut -d' ' -f2- || true
        )"
    fi
fi
if [ -z "$latest_ckpt" ] && [ -f "$train_log" ]; then
    latest_ckpt="$(
        grep -Eo 'latest_ckpt=/[^[:space:]]+' "$train_log" 2>/dev/null \
            | tail -1 | sed 's/^latest_ckpt=//' || true
    )"
fi
if [ -z "$latest_ckpt" ]; then
    echo "ERROR: could not determine latest checkpoint from ${ckpt_root}/${run_name} or ${train_log}" >&2
    exit 1
fi

echo "[gt-softwindow] latest_ckpt=${latest_ckpt}"
echo "[gt-softwindow] launching detached test eval"
MODEL="$latest_ckpt" \
    RJOB_NAME="$eval_rjob_name" \
    RJOB_GROUP="${EVAL_RJOB_GROUP:-gui_agent}" \
    RJOB_CHARGED_GROUP="${EVAL_RJOB_CHARGED_GROUP:-gui_agent}" \
    RJOB_CPU="${EVAL_RJOB_CPU:-80}" \
    RJOB_GPU="${EVAL_RJOB_GPU:-8}" \
    RJOB_MEMORY="${EVAL_RJOB_MEMORY:-800000}" \
    RJOB_MAX_WAIT_DURATION="${EVAL_RJOB_MAX_WAIT_DURATION:-2h0m0s}" \
    EVAL_JSONL="$TEST_JSONL" \
    OUT_DIR="$eval_root" \
    LOG_PATH="$eval_log" \
    WORKER_LOG="$eval_worker_log" \
    TP="${EVAL_TP:-8}" \
    MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-12000}" \
    GPU_MEM_UTIL="${EVAL_GPU_MEM_UTIL:-0.82}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}" \
    MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}" \
    NUM_ROLLOUTS="${EVAL_NUM_ROLLOUTS:-1}" \
    ROLLOUT_TEMPERATURE="${EVAL_ROLLOUT_TEMPERATURE:-0.0}" \
    TOP_P="${EVAL_TOP_P:-0.95}" \
    bash "${repo_root}/tools/evaluation/launch_single_ckpt_split_eval_detached_rjob.sh"

summary_path="${eval_root}/summary.json"
status_path="${eval_root}/status.json"
echo "[gt-softwindow] waiting for eval summary=${summary_path}"
for _ in $(seq 1 "${EVAL_WAIT_POLLS:-240}"); do
    if [ -f "$summary_path" ]; then
        echo "[gt-softwindow] eval summary ready"
        cat "$summary_path"
        echo
        echo "[gt-softwindow] completed"
        exit 0
    fi
    if [ -f "$status_path" ] && grep -q '"status": "failed"' "$status_path"; then
        echo "ERROR: eval failed; status=${status_path}" >&2
        cat "$status_path" >&2
        exit 1
    fi
    sleep "${EVAL_WAIT_INTERVAL_SEC:-30}"
done

echo "ERROR: eval summary not produced within wait budget: ${summary_path}" >&2
exit 1
