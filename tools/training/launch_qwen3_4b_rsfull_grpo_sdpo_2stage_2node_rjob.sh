#!/usr/bin/env bash
# Launch a two-node, two-stage Qwen3-VL-4B full-data GRPO+SDPO experiment.
#
# The script probes gui_agent first. If launch fails, it retries on aos unless
# DISABLE_AOS_FALLBACK=true is set.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

timestamp="$(date +%Y%m%d-%H%M%S)"
run_name="${RUN_NAME:-gui-sd-qwen3-4b-base_rsfull_grpo_sdpo_2stage_s80_e1-${timestamp}}"
rjob_name="${RJOB_NAME:-qwen3-4b-rsfull-2stage-2n-${timestamp}}"
log_path="${RJOB_LOG_PATH:-/data/codes/gui_grounding/data/${rjob_name}.rjob.log}"
worker_script="${WORKER_SCRIPT:-${repo_root}/tools/training/run_qwen3_4b_rsfull_grpo_sdpo_2stage_2node_worker.sh}"

base_model_path="${BASE_MODEL_PATH:-/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-4B-Instruct}"
train_jsonl="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}"
test_jsonl="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
ckpt_root="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"
artifact_root="${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd}"

cpu="${RJOB_CPU:-64}"
gpu="${RJOB_GPU:-8}"
memory="${RJOB_MEMORY:-800000}"
replica="${RJOB_REPLICA:-2}"
replica_restart="${RJOB_REPLICA_RESTART:-never}"
backoff_limit="${RJOB_BACKOFF_LIMIT:-1}"
positive_tags="${RJOB_POSITIVE_TAGS:-}"
mount_arg="--mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot"

build_predict_cmd() {
    local group="$1"
    local charged_group="${2:-$group}"
    PREDICT_CMD=(
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
        PREDICT_CMD+=(--positive-tags "$positive_tags")
    fi
}

build_launch_cmd() {
    local group="$1"
    local charged_group="${2:-$group}"
    LAUNCH_CMD=(
        brainctl launch
        --name "$rjob_name"
        -P "$replica"
        --replica-prefix
        --replica-restart="$replica_restart"
        --backoff-limit "$backoff_limit"
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
        --set-env "REPO_ROOT=$repo_root"
        --set-env "GUI_SD_ENV_ROOT=${GUI_SD_ENV_ROOT:-/data/codes/gui_grounding/conda_envs/GUI-SD}"
        --set-env "BASE_MODEL_PATH=$base_model_path"
        --set-env "TRAIN_JSONL=$train_jsonl"
        --set-env "TEST_JSONL=$test_jsonl"
        --set-env "CKPT_ROOT=$ckpt_root"
        --set-env "ARTIFACT_ROOT=$artifact_root"
        --set-env "NNODES=${NNODES:-2}"
        --set-env "NPROC_PER_NODE=${NPROC_PER_NODE:-7}"
        --set-env "TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
        --set-env "ROLLOUT_CUDA_VISIBLE_DEVICES=${ROLLOUT_CUDA_VISIBLE_DEVICES:-7}"
        --set-env "VLLM_SERVER_PORT=${VLLM_SERVER_PORT:-8392}"
        --set-env "MASTER_PORT=${MASTER_PORT:-29600}"
        --set-env "STAGE1_MAX_STEPS=${STAGE1_MAX_STEPS:-80}"
        --set-env "STAGE1_SAVE_STEPS=${STAGE1_SAVE_STEPS:-40}"
        --set-env "STAGE1_SAVE_TOTAL_LIMIT=${STAGE1_SAVE_TOTAL_LIMIT:-4}"
        --set-env "STAGE2_NUM_TRAIN_EPOCHS=${STAGE2_NUM_TRAIN_EPOCHS:-1}"
        --set-env "STAGE2_MAX_STEPS=${STAGE2_MAX_STEPS:--1}"
        --set-env "STAGE2_SAVE_STEPS=${STAGE2_SAVE_STEPS:-250}"
        --set-env "STAGE2_SAVE_TOTAL_LIMIT=${STAGE2_SAVE_TOTAL_LIMIT:-8}"
        --set-env "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
        --set-env "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}"
        --set-env "LR=${LR:-2e-6}"
        --set-env "STAGE1_LR=${STAGE1_LR:-${LR:-2e-6}}"
        --set-env "STAGE2_LR=${STAGE2_LR:-${LR:-2e-6}}"
        --set-env "WARMUP_RATIO=${WARMUP_RATIO:-0.03}"
        --set-env "NUM_GENERATIONS=${NUM_GENERATIONS:-8}"
        --set-env "NUM_ITERATIONS=${NUM_ITERATIONS:-1}"
        --set-env "MAX_LENGTH=${MAX_LENGTH:-20000}"
        --set-env "MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-64}"
        --set-env "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-zero2}"
        --set-env "TEACHER_DEEPSPEED_CONFIG=${TEACHER_DEEPSPEED_CONFIG:-zero3}"
        --set-env "OFFLOAD_TEACHER_MODEL=${OFFLOAD_TEACHER_MODEL:-false}"
        --set-env "VLLM_GPU_MEMORY_UTIL=${VLLM_GPU_MEMORY_UTIL:-0.88}"
        --set-env "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-20000}"
        --set-env "ROLLOUT_READY_TIMEOUT_SEC=${ROLLOUT_READY_TIMEOUT_SEC:-900}"
        --set-env "RENDEZVOUS_TIMEOUT_SEC=${RENDEZVOUS_TIMEOUT_SEC:-1200}"
        --set-env "DATASET_NUM_PROC=${DATASET_NUM_PROC:-8}"
        --set-env "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}"
        --set-env "EVAL_CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-0}"
        --set-env "EVAL_TP=${EVAL_TP:-1}"
        --set-env "EVAL_GPU_MEM_UTIL=${EVAL_GPU_MEM_UTIL:-0.88}"
        --set-env "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-512}"
    )
    if [ -n "$positive_tags" ]; then
        LAUNCH_CMD+=(--positive-tags "$positive_tags")
    fi
    LAUNCH_CMD+=(-- bash "$worker_script")
}

run_predict() {
    local output status
    set +e
    output="$("${PREDICT_CMD[@]}" 2>&1)"
    status=$?
    set -e
    printf '%s\n' "$output"
    if [ "$status" -ne 0 ]; then
        return "$status"
    fi
    if printf '%s\n' "$output" | grep -Eiq 'fail to pass quota check|quota check|not enough|insufficient|cannot .*schedule|no .*resource|resource .*not .*enough'; then
        return 1
    fi
    return 0
}

run_group() {
    local group="$1"
    local charged_group="${2:-$group}"
    echo "[launch-4b-2stage] probing group=${group} charged_group=${charged_group}"
    build_predict_cmd "$group" "$charged_group"
    echo "[launch-4b-2stage] predict: ${PREDICT_CMD[*]}"
    if ! run_predict; then
        echo "[launch-4b-2stage] predict failed for group=${group}; skip launch on this group"
        return 1
    fi

    build_launch_cmd "$group" "$charged_group"
    echo "[launch-4b-2stage] run_name=${run_name}"
    echo "[launch-4b-2stage] rjob_name=${rjob_name}"
    echo "[launch-4b-2stage] log_path=${log_path}"
    echo "[launch-4b-2stage] group=${group}"
    echo "[launch-4b-2stage] charged_group=${charged_group}"
    echo "[launch-4b-2stage] worker_script=${worker_script}"
    echo "[launch-4b-2stage] base_model_path=${base_model_path}"
    echo "[launch-4b-2stage] train_jsonl=${train_jsonl}"
    echo "[launch-4b-2stage] test_jsonl=${test_jsonl}"
    echo "[launch-4b-2stage] ckpt_root=${ckpt_root}"
    echo "[launch-4b-2stage] artifact_root=${artifact_root}"
    echo "[launch-4b-2stage] stage1_max_steps=${STAGE1_MAX_STEPS:-80}"
    echo "[launch-4b-2stage] stage2_epochs=${STAGE2_NUM_TRAIN_EPOCHS:-1}"
    echo "[launch-4b-2stage] per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
    echo "[launch-4b-2stage] gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-2}"
    echo "[launch-4b-2stage] lr=${LR:-2e-6}"
    echo "[launch-4b-2stage] command: ${LAUNCH_CMD[*]}"
    "${LAUNCH_CMD[@]}"
}

mkdir -p "$(dirname "$log_path")"
chmod +x "$worker_script"

primary_group="${RJOB_GROUP:-gui_agent}"
primary_charged_group="${RJOB_CHARGED_GROUP:-$primary_group}"
fallback_group="${RJOB_FALLBACK_GROUP:-aos}"
fallback_charged_group="${RJOB_FALLBACK_CHARGED_GROUP:-$fallback_group}"

{
    primary_status=0
    run_group "$primary_group" "$primary_charged_group" || primary_status=$?
    if [ "$primary_status" -eq 0 ]; then
        exit 0
    fi
    echo "[launch-4b-2stage] launch on ${primary_group} failed with status=${primary_status}"
    if [ "${DISABLE_AOS_FALLBACK:-false}" = "true" ]; then
        exit "$primary_status"
    fi
    echo "[launch-4b-2stage] retrying on fallback group=${fallback_group}"
    run_group "$fallback_group" "$fallback_charged_group"
} 2>&1 | tee "$log_path"
