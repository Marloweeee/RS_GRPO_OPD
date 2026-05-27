#!/usr/bin/env bash
# Launch one two-node Qwen3-VL-4B rs_full GRPO+SDPO visual-hint ablation run.
#
# Required/typical overrides:
#   OPSD_MASK_MODE=soft_window|jitter_box|zoom_in|no_mask
#   OPSD_HINT_MODE=hint|none
#   RJOB_GROUP=gui_agent|aos
#
# Checkpoints are saved every 10 steps by default to avoid losing all weights
# if the cluster job exits near the end of training.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
mask_mode="${OPSD_MASK_MODE:-soft_window}"
hint_mode="${OPSD_HINT_MODE:-hint}"
mode_slug="${MODE_SLUG:-${mask_mode//_/-}}"
if [ "$hint_mode" = "none" ]; then
    mode_slug="${MODE_SLUG:-no-hint}"
fi

run_name="${RUN_NAME:-gui-sd-qwen3-4b-rsfull_grpo_sdpo_hint_${mode_slug}_e1-${timestamp}}"
rjob_name="${RJOB_NAME:-q4b-sdpo-${mode_slug}-2n-${timestamp}}"
log_path="${RJOB_LOG_PATH:-/data/codes/gui_grounding/data/logs/${rjob_name}.rjob.log}"
worker_script="${WORKER_SCRIPT:-${repo_root}/tools/training/run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh}"

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
max_wait_duration="${RJOB_MAX_WAIT_DURATION:-10m0s}"
replica_creation_timeout_sec="${RJOB_REPLICA_CREATION_TIMEOUT_SEC:-900}"
cleanup_on_interrupt="${CLEANUP_RJOB_ON_INTERRUPT:-true}"
positive_tags="${RJOB_POSITIVE_TAGS:-feature/gpfs=yes}"
mount_arg="--mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot"

predict_cmd=(
    brainctl launch
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

echo "[launch-qwen3-4b-sdpo-hint-ablation] predict: ${predict_cmd[*]}"
"${predict_cmd[@]}"

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
    --set-env "DISTRIBUTED_JOB=true"
    --set-env "RUN_NAME=$run_name"
    --set-env "RENDEZVOUS_ID=$run_name"
    --set-env "REPO_ROOT=$repo_root"
    --set-env "GUI_SD_ENV_ROOT=${GUI_SD_ENV_ROOT:-/data/codes/gui_grounding/conda_envs/GUI-SD}"
    --set-env "BASE_MODEL_PATH=$base_model_path"
    --set-env "MODEL_PATH=${MODEL_PATH:-$base_model_path}"
    --set-env "TEACHER_PATH=${TEACHER_PATH:-${TEACHER_MODEL_PATH:-$base_model_path}}"
    --set-env "TRAIN_JSONL=$train_jsonl"
    --set-env "TEST_JSONL=$test_jsonl"
    --set-env "CKPT_ROOT=$ckpt_root"
    --set-env "ARTIFACT_ROOT=$artifact_root"
    --set-env "NNODES=${NNODES:-2}"
    --set-env "NPROC_PER_NODE=${NPROC_PER_NODE:-7}"
    --set-env "TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
    --set-env "ROLLOUT_CUDA_VISIBLE_DEVICES=${ROLLOUT_CUDA_VISIBLE_DEVICES:-7}"
    --set-env "VLLM_SERVER_PORT=${VLLM_SERVER_PORT:-8292}"
    --set-env "MASTER_PORT=${MASTER_PORT:-29500}"
    --set-env "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}"
    --set-env "MAX_STEPS=${MAX_STEPS:--1}"
    --set-env "SAVE_STEPS=${SAVE_STEPS:-10}"
    --set-env "SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-20}"
    --set-env "SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL:-true}"
    --set-env "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
    --set-env "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}"
    --set-env "LR=${LR:-2e-6}"
    --set-env "WARMUP_RATIO=${WARMUP_RATIO:-0.03}"
    --set-env "NUM_GENERATIONS=${NUM_GENERATIONS:-8}"
    --set-env "NUM_ITERATIONS=${NUM_ITERATIONS:-1}"
    --set-env "MAX_LENGTH=${MAX_LENGTH:-20000}"
    --set-env "MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-64}"
    --set-env "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-zero2}"
    --set-env "TEACHER_DEEPSPEED_CONFIG=${TEACHER_DEEPSPEED_CONFIG:-zero3}"
    --set-env "OFFLOAD_TEACHER_MODEL=${OFFLOAD_TEACHER_MODEL:-false}"
    --set-env "VLLM_GPU_MEMORY_UTIL=${VLLM_GPU_MEMORY_UTIL:-0.82}"
    --set-env "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-20000}"
    --set-env "ROLLOUT_READY_TIMEOUT_SEC=${ROLLOUT_READY_TIMEOUT_SEC:-900}"
    --set-env "RENDEZVOUS_TIMEOUT_SEC=${RENDEZVOUS_TIMEOUT_SEC:-1200}"
    --set-env "DATASET_NUM_PROC=${DATASET_NUM_PROC:-8}"
    --set-env "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}"
    --set-env "SDPO_LAMBDA=${SDPO_LAMBDA:-0.25}"
    --set-env "SDPO_TAU_GOOD=${SDPO_TAU_GOOD:-0.5}"
    --set-env "SDPO_TAU_FAIL=${SDPO_TAU_FAIL:-0.3}"
    --set-env "SDPO_DELTA=${SDPO_DELTA:-0.5}"
    --set-env "SDPO_TARGET=${SDPO_TARGET:-rollout}"
    --set-env "OPSD_TOKEN_WEIGHT_MODE=${OPSD_TOKEN_WEIGHT_MODE:-uniform-entropy}"
    --set-env "OPSD_NON_DIGIT_WEIGHT=${OPSD_NON_DIGIT_WEIGHT:-0.05}"
    --set-env "OPSD_MAX_DIGIT_LEN=${OPSD_MAX_DIGIT_LEN:-3}"
    --set-env "OPSD_MASK_MODE=$mask_mode"
    --set-env "OPSD_HINT_MODE=$hint_mode"
    --set-env "OPSD_ZOOM_RATIO=${OPSD_ZOOM_RATIO:-2.0}"
    --set-env "OPSD_MIN_AREA_FRAC=${OPSD_MIN_AREA_FRAC:-0.1}"
    --set-env "OPSD_GAUSSIAN_SIGMA_RATIO=${OPSD_GAUSSIAN_SIGMA_RATIO:-1.5}"
    --set-env "OPSD_HINT_BOX_COLOR=${OPSD_HINT_BOX_COLOR:-magenta}"
    --set-env "OPSD_JITTER_RATIO=${OPSD_JITTER_RATIO:-0.2}"
    --set-env "GRPO_BETA=${GRPO_BETA:-0.04}"
    --set-env "ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}"
    --set-env "TOP_P=${TOP_P:-0.95}"
    --set-env "TOP_K=${TOP_K:-50}"
    --set-env "EVAL_CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-0}"
    --set-env "EVAL_TP=${EVAL_TP:-1}"
    --set-env "EVAL_GPU_MEM_UTIL=${EVAL_GPU_MEM_UTIL:-0.88}"
    --set-env "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-512}"
    --set-env "EVAL_MAX_MODEL_LEN=${EVAL_MAX_MODEL_LEN:-12000}"
    --set-env "GUI_SD_CACHE_ROOT=${GUI_SD_CACHE_ROOT:-}"
)
if [ -n "$positive_tags" ]; then
    launch_cmd+=(--positive-tags "$positive_tags")
fi

launch_cmd+=(
    --
    bash "$worker_script"
)

mkdir -p "$(dirname "$log_path")"

stop_rjob() {
    local reason="$1"
    echo "[launch-qwen3-4b-sdpo-hint-ablation] stopping rjob=${rjob_name}; reason=${reason}"
    if [ -n "$namespace" ]; then
        brainctl -n "$namespace" stop "rjob/${rjob_name}" >/dev/null 2>&1 || true
    else
        brainctl stop "rjob/${rjob_name}" >/dev/null 2>&1 || true
    fi
}

launch_pid=""
watchdog_pid=""

cleanup_interrupt() {
    local status=$?
    if [ "$status" -ne 0 ] && [ "$cleanup_on_interrupt" = "true" ] && [ -n "$launch_pid" ]; then
        stop_rjob "launcher_exit_status_${status}"
        kill "$launch_pid" >/dev/null 2>&1 || true
    fi
    if [ -n "$watchdog_pid" ]; then
        kill "$watchdog_pid" >/dev/null 2>&1 || true
    fi
    exit "$status"
}
trap cleanup_interrupt HUP INT TERM

echo "[launch-qwen3-4b-sdpo-hint-ablation] run_name=${run_name}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] rjob_name=${rjob_name}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] log_path=${log_path}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] group=${group}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] charged_group=${charged_group}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] namespace=${namespace}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] max_wait_duration=${max_wait_duration}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] replica_creation_timeout_sec=${replica_creation_timeout_sec}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] worker_script=${worker_script}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] base_model_path=${base_model_path}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] train_jsonl=${train_jsonl}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] test_jsonl=${test_jsonl}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] ckpt_root=${ckpt_root}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] artifact_root=${artifact_root}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] opsd_mask_mode=${mask_mode}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] opsd_hint_mode=${hint_mode}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] save_steps=${SAVE_STEPS:-10}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] save_total_limit=${SAVE_TOTAL_LIMIT:-20}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-2}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] lr=${LR:-2e-6}"
echo "[launch-qwen3-4b-sdpo-hint-ablation] command: ${launch_cmd[*]}"

"${launch_cmd[@]}" > >(tee "$log_path") 2>&1 &
launch_pid=$!

if [ "$replica_creation_timeout_sec" -gt 0 ]; then
    (
        sleep "$replica_creation_timeout_sec"
        if kill -0 "$launch_pid" >/dev/null 2>&1; then
            if ! grep -Eq 'cuda_available=True|Start time of running main|Train:|checkpoint-|completed|failed|CUDA is not ready|Traceback|ERROR:' "$log_path" 2>/dev/null; then
                echo "[launch-qwen3-4b-sdpo-hint-ablation] ERROR: no worker progress after ${replica_creation_timeout_sec}s; stopping rjob=${rjob_name}" | tee -a "$log_path"
                stop_rjob "replica_creation_timeout"
                kill "$launch_pid" >/dev/null 2>&1 || true
            fi
        fi
    ) &
    watchdog_pid=$!
fi

set +e
wait "$launch_pid"
launch_status=$?
set -e

if [ -n "$watchdog_pid" ]; then
    kill "$watchdog_pid" >/dev/null 2>&1 || true
    wait "$watchdog_pid" >/dev/null 2>&1 || true
fi

if [ "$launch_status" -ne 0 ]; then
    echo "[launch-qwen3-4b-sdpo-hint-ablation] brainctl launch exited with status=${launch_status}"
    if [ "$cleanup_on_interrupt" = "true" ]; then
        stop_rjob "brainctl_launch_status_${launch_status}"
    fi
    exit "$launch_status"
fi
