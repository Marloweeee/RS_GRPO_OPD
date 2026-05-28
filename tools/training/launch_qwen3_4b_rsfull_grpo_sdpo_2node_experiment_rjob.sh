#!/usr/bin/env bash
# Generic two-node Qwen3-VL-4B rs_full GRPO+SDPO rjob launcher.
#
# User-facing experiment scripts set WORKER_SCRIPT/RUN_NAME/RJOB_NAME and stage
# overrides, then delegate here. The launcher intentionally skips in-worker eval
# by default so training jobs finish cleanly and evaluation can be run as a
# separate GPU job.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
experiment_slug="${EXPERIMENT_SLUG:-zoom-stage-refresh}"
run_name="${RUN_NAME:-gui-sd-qwen3-4b-rsfull_grpo_sdpo_${experiment_slug}_${timestamp}}"
rjob_name="${RJOB_NAME:-q4b-${experiment_slug}-2n-${timestamp}}"
log_path="${RJOB_LOG_PATH:-/data/codes/gui_grounding/data/logs/training/${rjob_name}.rjob.log}"
worker_script="${WORKER_SCRIPT:?WORKER_SCRIPT must be set by the experiment wrapper}"

base_model_path="${BASE_MODEL_PATH:-/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-4B-Instruct}"
train_jsonl="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}"
test_jsonl="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
ckpt_root="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"
artifact_root="${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd}"

group="${RJOB_GROUP:-gui_agent}"
charged_group="${RJOB_CHARGED_GROUP:-${group}}"
namespace="${RJOB_NAMESPACE:-${KUBEBRAIN_NAMESPACE:-shai-core}}"
cpu="${RJOB_CPU:-64}"
gpu="${RJOB_GPU:-8}"
memory="${RJOB_MEMORY:-800000}"
replica="${RJOB_REPLICA:-2}"
replica_restart="${RJOB_REPLICA_RESTART:-never}"
backoff_limit="${RJOB_BACKOFF_LIMIT:-1}"
max_wait_duration="${RJOB_MAX_WAIT_DURATION:-12h0m0s}"
cleanup_on_interrupt="${CLEANUP_RJOB_ON_INTERRUPT:-true}"
positive_tags="${RJOB_POSITIVE_TAGS:-feature/gpfs=yes}"
mount_arg="--mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot"
completion_pattern="${COMPLETION_GREP_PATTERN:-run completed}"
completion_grace_sec="${COMPLETION_GRACE_SEC:-45}"

mkdir -p "$(dirname "$log_path")"
chmod +x "$worker_script"

predict_cmd=(
    brainctl launch
    -P "$replica"
    --replica-prefix
    --cpu "$cpu"
    --gpu "$gpu"
    --memory "$memory"
    --group "$group"
    --charged-group="$charged_group"
    --private-machine=group
    "$mount_arg"
    --custom-resources rdma/mlnx_shared=8
    --predict-only
)
if [ -n "$positive_tags" ]; then
    predict_cmd+=(--positive-tags "$positive_tags")
fi

launch_cmd=(
    brainctl launch
    --name "$rjob_name"
    -P "$replica"
    --replica-prefix
    --replica-restart="$replica_restart"
    --backoff-limit "$backoff_limit"
    --max-wait-duration "$max_wait_duration"
    --cpu "$cpu"
    --gpu "$gpu"
    --memory="$memory"
    --group "$group"
    --charged-group="$charged_group"
    --private-machine=group
    "$mount_arg"
    --custom-resources rdma/mlnx_shared=8
)
if [ -n "$positive_tags" ]; then
    launch_cmd+=(--positive-tags "$positive_tags")
fi

add_env() {
    launch_cmd+=(--set-env "$1=$2")
}

add_env_if_set() {
    local key="$1"
    local value="${!key:-}"
    if [ -n "$value" ]; then
        add_env "$key" "$value"
    fi
}

add_env "DISTRIBUTED_JOB" "true"
add_env "RUN_NAME" "$run_name"
add_env "REPO_ROOT" "$repo_root"
add_env "GUI_SD_ENV_ROOT" "${GUI_SD_ENV_ROOT:-/data/codes/gui_grounding/conda_envs/GUI-SD}"
add_env "BASE_MODEL_PATH" "$base_model_path"
add_env "TRAIN_JSONL" "$train_jsonl"
add_env "TEST_JSONL" "$test_jsonl"
add_env "CKPT_ROOT" "$ckpt_root"
add_env "ARTIFACT_ROOT" "$artifact_root"
add_env "NNODES" "${NNODES:-2}"
add_env "NPROC_PER_NODE" "${NPROC_PER_NODE:-7}"
add_env "TRAIN_CUDA_VISIBLE_DEVICES" "${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
add_env "ROLLOUT_CUDA_VISIBLE_DEVICES" "${ROLLOUT_CUDA_VISIBLE_DEVICES:-7}"
add_env "VLLM_SERVER_PORT" "${VLLM_SERVER_PORT:-8292}"
add_env "MASTER_PORT" "${MASTER_PORT:-29500}"
add_env "NUM_TRAIN_EPOCHS" "${NUM_TRAIN_EPOCHS:-1}"
add_env "MAX_STEPS" "${MAX_STEPS:--1}"
add_env "SAVE_STEPS" "${SAVE_STEPS:-50}"
add_env "SAVE_TOTAL_LIMIT" "${SAVE_TOTAL_LIMIT:-10}"
add_env "SAVE_ONLY_MODEL" "${SAVE_ONLY_MODEL:-true}"
add_env "TUNER_TYPE" "${TUNER_TYPE:-full}"
add_env "ALLOW_RESUME" "${ALLOW_RESUME:-false}"
add_env "AUTO_RESUME" "${AUTO_RESUME:-false}"
add_env "PER_DEVICE_TRAIN_BATCH_SIZE" "${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
add_env "GRADIENT_ACCUMULATION_STEPS" "${GRADIENT_ACCUMULATION_STEPS:-2}"
add_env "LR" "${LR:-2e-6}"
add_env "WARMUP_RATIO" "${WARMUP_RATIO:-0.03}"
add_env "NUM_GENERATIONS" "${NUM_GENERATIONS:-8}"
add_env "NUM_ITERATIONS" "${NUM_ITERATIONS:-1}"
add_env "MAX_LENGTH" "${MAX_LENGTH:-20000}"
add_env "MAX_COMPLETION_LENGTH" "${MAX_COMPLETION_LENGTH:-64}"
add_env "DEEPSPEED_CONFIG" "${DEEPSPEED_CONFIG:-zero2}"
add_env "TEACHER_DEEPSPEED_CONFIG" "${TEACHER_DEEPSPEED_CONFIG:-zero3}"
add_env "OFFLOAD_TEACHER_MODEL" "${OFFLOAD_TEACHER_MODEL:-false}"
add_env "VLLM_GPU_MEMORY_UTIL" "${VLLM_GPU_MEMORY_UTIL:-0.82}"
add_env "VLLM_MAX_MODEL_LEN" "${VLLM_MAX_MODEL_LEN:-20000}"
add_env_if_set "VLLM_MAX_NUM_SEQS"
add_env_if_set "VLLM_ENFORCE_EAGER"
add_env "ROLLOUT_READY_TIMEOUT_SEC" "${ROLLOUT_READY_TIMEOUT_SEC:-900}"
add_env "RENDEZVOUS_TIMEOUT_SEC" "${RENDEZVOUS_TIMEOUT_SEC:-1200}"
add_env "DATASET_NUM_PROC" "${DATASET_NUM_PROC:-8}"
add_env "DATALOADER_NUM_WORKERS" "${DATALOADER_NUM_WORKERS:-8}"
add_env "SDPO_LAMBDA" "${SDPO_LAMBDA:-0.25}"
add_env "SDPO_TAU_GOOD" "${SDPO_TAU_GOOD:-0.5}"
add_env "SDPO_TAU_FAIL" "${SDPO_TAU_FAIL:-0.3}"
add_env "SDPO_DELTA" "${SDPO_DELTA:-0.5}"
add_env "SDPO_GRPO_FAILED_WEIGHT" "${SDPO_GRPO_FAILED_WEIGHT:-1.0}"
add_env "SDPO_DISTILL_SCOPE" "${SDPO_DISTILL_SCOPE:-failed}"
add_env "SDPO_TARGET" "${SDPO_TARGET:-rollout}"
add_env "SDPO_HINT_SOURCE" "${SDPO_HINT_SOURCE:-gt}"
add_env "SDPO_SIBLING_SELECT_METRIC" "${SDPO_SIBLING_SELECT_METRIC:-reward}"
add_env "SDPO_SIBLING_FALLBACK" "${SDPO_SIBLING_FALLBACK:-gt}"
add_env "SDPO_TEACHER_REFRESH_MODE" "${SDPO_TEACHER_REFRESH_MODE:-fixed}"
add_env "SDPO_TEACHER_REFRESH_STEP" "${SDPO_TEACHER_REFRESH_STEP:-${TEACHER_REFRESH_STEP:--1}}"
add_env "SDPO_TEACHER_REFRESH_WARMUP" "${SDPO_TEACHER_REFRESH_WARMUP:-80}"
add_env "SDPO_TEACHER_REFRESH_WINDOW" "${SDPO_TEACHER_REFRESH_WINDOW:-50}"
add_env "SDPO_TEACHER_REFRESH_CHECK_INTERVAL" "${SDPO_TEACHER_REFRESH_CHECK_INTERVAL:-10}"
add_env "SDPO_TEACHER_REFRESH_MIN_IOU_IMPROVE" "${SDPO_TEACHER_REFRESH_MIN_IOU_IMPROVE:-0.01}"
add_env "SDPO_TEACHER_REFRESH_MAX_FAILED" "${SDPO_TEACHER_REFRESH_MAX_FAILED:-0.55}"
add_env "SDPO_TEACHER_REFRESH_MIN_SDPO_LOSS" "${SDPO_TEACHER_REFRESH_MIN_SDPO_LOSS:-0.02}"
add_env "SDPO_TEACHER_REFRESH_MAX_KL" "${SDPO_TEACHER_REFRESH_MAX_KL:-0.30}"
add_env "SDPO_TEACHER_REFRESH_SHORT_WINDOW" "${SDPO_TEACHER_REFRESH_SHORT_WINDOW:-20}"
add_env "SDPO_TEACHER_REFRESH_LONG_WINDOW" "${SDPO_TEACHER_REFRESH_LONG_WINDOW:-80}"
add_env "SDPO_TEACHER_REFRESH_EWMA_ALPHA" "${SDPO_TEACHER_REFRESH_EWMA_ALPHA:-0.10}"
add_env "SDPO_TEACHER_REFRESH_CONSECUTIVE_CHECKS" "${SDPO_TEACHER_REFRESH_CONSECUTIVE_CHECKS:-2}"
add_env "SDPO_TEACHER_REFRESH_MIN_SHORT_LONG_IOU_GAIN" "${SDPO_TEACHER_REFRESH_MIN_SHORT_LONG_IOU_GAIN:-0.006}"
add_env "SDPO_TEACHER_REFRESH_MIN_EWMA_IOU_GAIN" "${SDPO_TEACHER_REFRESH_MIN_EWMA_IOU_GAIN:-0.008}"
add_env "SDPO_TEACHER_REFRESH_MAX_IOU05_DROP" "${SDPO_TEACHER_REFRESH_MAX_IOU05_DROP:-0.010}"
add_env "SDPO_TEACHER_REFRESH_MAX_REFRESHES" "${SDPO_TEACHER_REFRESH_MAX_REFRESHES:-1}"
add_env "SDPO_TEACHER_REFRESH_COOLDOWN_STEPS" "${SDPO_TEACHER_REFRESH_COOLDOWN_STEPS:-0}"
add_env "OPSD_TOKEN_WEIGHT_MODE" "${OPSD_TOKEN_WEIGHT_MODE:-uniform-entropy}"
add_env "OPSD_NON_DIGIT_WEIGHT" "${OPSD_NON_DIGIT_WEIGHT:-0.05}"
add_env "OPSD_MAX_DIGIT_LEN" "${OPSD_MAX_DIGIT_LEN:-3}"
add_env "OPSD_EMA_DECAY" "${OPSD_EMA_DECAY:-0.0}"
add_env "OPSD_MASK_MODE" "${OPSD_MASK_MODE:-zoom_in}"
add_env "OPSD_HINT_MODE" "${OPSD_HINT_MODE:-hint}"
add_env "OPSD_ZOOM_RATIO" "${OPSD_ZOOM_RATIO:-2.0}"
add_env "OPSD_MIN_AREA_FRAC" "${OPSD_MIN_AREA_FRAC:-0.1}"
add_env "OPSD_GAUSSIAN_SIGMA_RATIO" "${OPSD_GAUSSIAN_SIGMA_RATIO:-1.5}"
add_env "OPSD_HINT_BOX_COLOR" "${OPSD_HINT_BOX_COLOR:-magenta}"
add_env "OPSD_JITTER_RATIO" "${OPSD_JITTER_RATIO:-0.2}"
add_env "GRPO_BETA" "${GRPO_BETA:-0.04}"
add_env "ROLLOUT_TEMPERATURE" "${ROLLOUT_TEMPERATURE:-0.7}"
add_env "TOP_P" "${TOP_P:-0.95}"
add_env "TOP_K" "${TOP_K:-50}"
add_env "EVAL_CUDA_VISIBLE_DEVICES" "${EVAL_CUDA_VISIBLE_DEVICES:-0}"
add_env "EVAL_TP" "${EVAL_TP:-1}"
add_env "EVAL_GPU_MEM_UTIL" "${EVAL_GPU_MEM_UTIL:-0.88}"
add_env "EVAL_BATCH_SIZE" "${EVAL_BATCH_SIZE:-512}"
add_env "EVAL_MAX_MODEL_LEN" "${EVAL_MAX_MODEL_LEN:-12000}"
add_env "STAGE2_SKIP_EVAL" "${STAGE2_SKIP_EVAL:-true}"
add_env "SKIP_EVAL" "${SKIP_EVAL:-true}"

for key in \
    MODEL_PATH TEACHER_PATH ROLLOUT_MODEL_PATH RUN_TIMESTAMP \
    TEACHER_REFRESH_STEP \
    SDPO_DISTILL_SCOPE \
    SDPO_HINT_SOURCE SDPO_SIBLING_SELECT_METRIC SDPO_SIBLING_FALLBACK \
    SDPO_TEACHER_REFRESH_MODE SDPO_TEACHER_REFRESH_STEP SDPO_TEACHER_REFRESH_WARMUP \
    SDPO_TEACHER_REFRESH_WINDOW SDPO_TEACHER_REFRESH_CHECK_INTERVAL \
    SDPO_TEACHER_REFRESH_MIN_IOU_IMPROVE SDPO_TEACHER_REFRESH_MAX_FAILED \
    SDPO_TEACHER_REFRESH_MIN_SDPO_LOSS SDPO_TEACHER_REFRESH_MAX_KL \
    SDPO_TEACHER_REFRESH_SHORT_WINDOW SDPO_TEACHER_REFRESH_LONG_WINDOW \
    SDPO_TEACHER_REFRESH_EWMA_ALPHA SDPO_TEACHER_REFRESH_CONSECUTIVE_CHECKS \
    SDPO_TEACHER_REFRESH_MIN_SHORT_LONG_IOU_GAIN SDPO_TEACHER_REFRESH_MIN_EWMA_IOU_GAIN \
    SDPO_TEACHER_REFRESH_MAX_IOU05_DROP \
    SDPO_TEACHER_REFRESH_MAX_REFRESHES SDPO_TEACHER_REFRESH_COOLDOWN_STEPS \
    OPSD_EMA_DECAY \
    STAGE1_RUN_NAME STAGE2_RUN_NAME \
    STAGE1_MAX_STEPS STAGE1_NUM_TRAIN_EPOCHS STAGE1_SAVE_STEPS STAGE1_SAVE_TOTAL_LIMIT \
    STAGE1_LR STAGE1_WARMUP_RATIO STAGE1_PER_DEVICE_TRAIN_BATCH_SIZE STAGE1_GRADIENT_ACCUMULATION_STEPS \
    STAGE2_MAX_STEPS STAGE2_NUM_TRAIN_EPOCHS STAGE2_SAVE_STEPS STAGE2_SAVE_TOTAL_LIMIT \
    STAGE2_LR STAGE2_WARMUP_RATIO STAGE2_PER_DEVICE_TRAIN_BATCH_SIZE STAGE2_GRADIENT_ACCUMULATION_STEPS \
    STAGE2_SAVE_ONLY_MODEL STAGE2_TEACHER_PATH \
    REFRESH_STEP REFRESH_TOTAL_MAX_STEPS \
    STAGE1_OPSD_MASK_MODE STAGE2_OPSD_MASK_MODE STAGE1_OPSD_HINT_MODE STAGE2_OPSD_HINT_MODE \
    RESUME_FROM_CHECKPOINT RESUME_ONLY_MODEL GUI_SD_CACHE_ROOT; do
    add_env_if_set "$key"
done

launch_cmd+=(-- bash "$worker_script")

completed_marker="${log_path}.completed"
rm -f "$completed_marker"

stop_rjob() {
    local reason="$1"
    echo "[launch-qwen3-4b-stage-exp] stopping rjob=${rjob_name}; reason=${reason}"
    if [ -n "$namespace" ]; then
        brainctl -n "$namespace" stop "rjob/${rjob_name}" >/dev/null 2>&1 || true
    else
        brainctl stop "rjob/${rjob_name}" >/dev/null 2>&1 || true
    fi
}

launch_pid=""
completion_watcher_pid=""

cleanup_interrupt() {
    local status=$?
    if [ "$status" -ne 0 ] && [ ! -f "$completed_marker" ] && [ "$cleanup_on_interrupt" = "true" ] && [ -n "$launch_pid" ]; then
        stop_rjob "launcher_exit_status_${status}"
        kill "$launch_pid" >/dev/null 2>&1 || true
    fi
    if [ -n "$completion_watcher_pid" ]; then
        kill "$completion_watcher_pid" >/dev/null 2>&1 || true
    fi
    exit "$status"
}
trap cleanup_interrupt HUP INT TERM

{
    echo "[launch-qwen3-4b-stage-exp] predict: ${predict_cmd[*]}"
    "${predict_cmd[@]}"

    echo "[launch-qwen3-4b-stage-exp] run_name=${run_name}"
    echo "[launch-qwen3-4b-stage-exp] rjob_name=${rjob_name}"
    echo "[launch-qwen3-4b-stage-exp] log_path=${log_path}"
    echo "[launch-qwen3-4b-stage-exp] group=${group}"
    echo "[launch-qwen3-4b-stage-exp] charged_group=${charged_group}"
    echo "[launch-qwen3-4b-stage-exp] namespace=${namespace}"
    echo "[launch-qwen3-4b-stage-exp] max_wait_duration=${max_wait_duration}"
    echo "[launch-qwen3-4b-stage-exp] worker_script=${worker_script}"
    echo "[launch-qwen3-4b-stage-exp] base_model_path=${base_model_path}"
    echo "[launch-qwen3-4b-stage-exp] train_jsonl=${train_jsonl}"
    echo "[launch-qwen3-4b-stage-exp] test_jsonl=${test_jsonl}"
    echo "[launch-qwen3-4b-stage-exp] ckpt_root=${ckpt_root}"
    echo "[launch-qwen3-4b-stage-exp] artifact_root=${artifact_root}"
    echo "[launch-qwen3-4b-stage-exp] opsd_mask_mode=${OPSD_MASK_MODE:-zoom_in}"
    echo "[launch-qwen3-4b-stage-exp] opsd_hint_mode=${OPSD_HINT_MODE:-hint}"
    echo "[launch-qwen3-4b-stage-exp] sdpo_distill_scope=${SDPO_DISTILL_SCOPE:-failed}"
    echo "[launch-qwen3-4b-stage-exp] sdpo_hint_source=${SDPO_HINT_SOURCE:-gt} sdpo_sibling_select_metric=${SDPO_SIBLING_SELECT_METRIC:-reward} sdpo_sibling_fallback=${SDPO_SIBLING_FALLBACK:-gt}"
    echo "[launch-qwen3-4b-stage-exp] sdpo_teacher_refresh_mode=${SDPO_TEACHER_REFRESH_MODE:-fixed}"
    echo "[launch-qwen3-4b-stage-exp] sdpo_teacher_refresh_step=${SDPO_TEACHER_REFRESH_STEP:-${TEACHER_REFRESH_STEP:--1}}"
    echo "[launch-qwen3-4b-stage-exp] sdpo_teacher_refresh_metric warmup=${SDPO_TEACHER_REFRESH_WARMUP:-80} window=${SDPO_TEACHER_REFRESH_WINDOW:-50} check_interval=${SDPO_TEACHER_REFRESH_CHECK_INTERVAL:-10} min_iou_improve=${SDPO_TEACHER_REFRESH_MIN_IOU_IMPROVE:-0.01} max_failed=${SDPO_TEACHER_REFRESH_MAX_FAILED:-0.55} min_sdpo_loss=${SDPO_TEACHER_REFRESH_MIN_SDPO_LOSS:-0.02} max_kl=${SDPO_TEACHER_REFRESH_MAX_KL:-0.30} max_refreshes=${SDPO_TEACHER_REFRESH_MAX_REFRESHES:-1} cooldown_steps=${SDPO_TEACHER_REFRESH_COOLDOWN_STEPS:-0}"
    echo "[launch-qwen3-4b-stage-exp] sdpo_teacher_refresh_metric_ewma short_window=${SDPO_TEACHER_REFRESH_SHORT_WINDOW:-20} long_window=${SDPO_TEACHER_REFRESH_LONG_WINDOW:-80} ewma_alpha=${SDPO_TEACHER_REFRESH_EWMA_ALPHA:-0.10} consecutive_checks=${SDPO_TEACHER_REFRESH_CONSECUTIVE_CHECKS:-2} min_short_long_iou_gain=${SDPO_TEACHER_REFRESH_MIN_SHORT_LONG_IOU_GAIN:-0.006} min_ewma_iou_gain=${SDPO_TEACHER_REFRESH_MIN_EWMA_IOU_GAIN:-0.008} max_iou05_drop=${SDPO_TEACHER_REFRESH_MAX_IOU05_DROP:-0.010}"
    echo "[launch-qwen3-4b-stage-exp] stage1_save_steps=${STAGE1_SAVE_STEPS:-unset}"
    echo "[launch-qwen3-4b-stage-exp] stage2_save_steps=${STAGE2_SAVE_STEPS:-unset}"
    echo "[launch-qwen3-4b-stage-exp] stage2_skip_eval=${STAGE2_SKIP_EVAL:-true}"
    echo "[launch-qwen3-4b-stage-exp] per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
    echo "[launch-qwen3-4b-stage-exp] gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-2}"
    echo "[launch-qwen3-4b-stage-exp] lr=${LR:-2e-6}"
    echo "[launch-qwen3-4b-stage-exp] command: ${launch_cmd[*]}"

    "${launch_cmd[@]}" &
    launch_pid=$!

    (
        while kill -0 "$launch_pid" >/dev/null 2>&1; do
            if [ -f "$log_path" ]; then
                completed_count="$(grep -E "$completion_pattern" "$log_path" 2>/dev/null | wc -l | tr -d ' ')"
                if [ "${completed_count:-0}" -ge "$replica" ]; then
                    echo "[launch-qwen3-4b-stage-exp] detected ${completed_count}/${replica} completion lines; ending local log follower after ${completion_grace_sec}s" | tee -a "$log_path"
                    touch "$completed_marker"
                    sleep "$completion_grace_sec"
                    kill "$launch_pid" >/dev/null 2>&1 || true
                    exit 0
                fi
            fi
            sleep 30
        done
    ) &
    completion_watcher_pid=$!

    set +e
    wait "$launch_pid"
    launch_status=$?
    set -e

    if [ -n "$completion_watcher_pid" ]; then
        kill "$completion_watcher_pid" >/dev/null 2>&1 || true
        wait "$completion_watcher_pid" >/dev/null 2>&1 || true
    fi

    if [ "$launch_status" -ne 0 ]; then
        if [ -f "$completed_marker" ]; then
            echo "[launch-qwen3-4b-stage-exp] brainctl log follower ended after worker completion; treating as success"
            exit 0
        fi
        echo "[launch-qwen3-4b-stage-exp] brainctl launch exited with status=${launch_status}"
        if [ "$cleanup_on_interrupt" = "true" ]; then
            stop_rjob "brainctl_launch_status_${launch_status}"
        fi
        exit "$launch_status"
    fi
} 2>&1 | tee "$log_path"
