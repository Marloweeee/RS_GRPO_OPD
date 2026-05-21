#!/usr/bin/env bash
# Launch the two-node Qwen3-VL-4B full rs_full GRPO+SDPO run.
#
# This script runs a predict-only resource probe first, then launches two rjob
# replicas. Each replica runs run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

timestamp="$(date +%Y%m%d-%H%M%S)"
run_name="${RUN_NAME:-gui-sd-qwen3-4b-base_rsfull_grpo_sdpo_2node_bsz4_gacc2_lr2e6_e1-${timestamp}}"
rjob_name="${RJOB_NAME:-qwen3-4b-rsfull-2n-${timestamp}}"
log_path="${RJOB_LOG_PATH:-/data/codes/gui_grounding/data/${rjob_name}.rjob.log}"
worker_script="${WORKER_SCRIPT:-${repo_root}/tools/training/run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh}"

base_model_path="${BASE_MODEL_PATH:-/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-4B-Instruct}"
train_jsonl="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}"
test_jsonl="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
ckpt_root="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"
artifact_root="${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd}"

group="${RJOB_GROUP:-gui_agent}"
charged_group="${RJOB_CHARGED_GROUP:-${group}}"
cpu="${RJOB_CPU:-64}"
gpu="${RJOB_GPU:-8}"
memory="${RJOB_MEMORY:-800000}"
replica="${RJOB_REPLICA:-2}"
replica_restart="${RJOB_REPLICA_RESTART:-never}"
backoff_limit="${RJOB_BACKOFF_LIMIT:-1}"
positive_tags="${RJOB_POSITIVE_TAGS:-}"
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

echo "[launch-qwen3-4b-2node] predict: ${predict_cmd[*]}"
"${predict_cmd[@]}"

launch_cmd=(
    brainctl launch
    --name "$rjob_name"
    -P "$replica"
    --replica-prefix
    --replica-restart="$replica_restart"
    --backoff-limit "$backoff_limit"
    --cpu "$cpu"
    --gpu "$gpu"
    --memory="$memory"
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
    --set-env "SAVE_STEPS=${SAVE_STEPS:-250}"
    --set-env "SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-8}"
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
    launch_cmd+=(--positive-tags "$positive_tags")
fi

launch_cmd+=(
    --
    bash "$worker_script"
)

echo "[launch-qwen3-4b-2node] run_name=${run_name}"
echo "[launch-qwen3-4b-2node] rjob_name=${rjob_name}"
echo "[launch-qwen3-4b-2node] log_path=${log_path}"
echo "[launch-qwen3-4b-2node] group=${group}"
echo "[launch-qwen3-4b-2node] charged_group=${charged_group}"
echo "[launch-qwen3-4b-2node] worker_script=${worker_script}"
echo "[launch-qwen3-4b-2node] base_model_path=${base_model_path}"
echo "[launch-qwen3-4b-2node] train_jsonl=${train_jsonl}"
echo "[launch-qwen3-4b-2node] test_jsonl=${test_jsonl}"
echo "[launch-qwen3-4b-2node] ckpt_root=${ckpt_root}"
echo "[launch-qwen3-4b-2node] artifact_root=${artifact_root}"
echo "[launch-qwen3-4b-2node] per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
echo "[launch-qwen3-4b-2node] gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-2}"
echo "[launch-qwen3-4b-2node] lr=${LR:-2e-6}"
echo "[launch-qwen3-4b-2node] command: ${launch_cmd[*]}"

"${launch_cmd[@]}" 2>&1 | tee "$log_path"
