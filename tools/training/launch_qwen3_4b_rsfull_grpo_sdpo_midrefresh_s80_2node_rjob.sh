#!/usr/bin/env bash
# Launch the two-node Qwen3-VL-4B mid-training teacher/ref refresh experiment.
#
# The launcher always probes gui_agent with --predict-only first, then submits a
# two-replica rjob. Each replica runs the mid-refresh worker:
#   base -> 80 steps -> refresh student/ref/teacher to checkpoint-80 -> resume
#   trainer state to the one-epoch target.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

timestamp="$(date +%Y%m%d-%H%M%S)"
run_name="${RUN_NAME:-gui-sd-qwen3-4b-base_rsfull_grpo_sdpo_midrefresh_s80_e1-${timestamp}}"
rjob_name="${RJOB_NAME:-qwen3-4b-rsfull-midrefresh-s80-2n-${timestamp}}"
log_path="${RJOB_LOG_PATH:-/data/codes/gui_grounding/data/${rjob_name}.rjob.log}"
worker_script="${WORKER_SCRIPT:-${repo_root}/tools/training/run_qwen3_4b_rsfull_grpo_sdpo_midrefresh_s80_2node_worker.sh}"

base_model_path="${BASE_MODEL_PATH:-/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-4B-Instruct}"
train_jsonl="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}"
test_jsonl="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
ckpt_root="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"
artifact_root="${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd}"

group="${RJOB_GROUP:-gui_agent}"
charged_group="${RJOB_CHARGED_GROUP:-$group}"
cpu="${RJOB_CPU:-64}"
gpu="${RJOB_GPU:-8}"
memory="${RJOB_MEMORY:-800000}"
replica="${RJOB_REPLICA:-2}"
replica_restart="${RJOB_REPLICA_RESTART:-never}"
backoff_limit="${RJOB_BACKOFF_LIMIT:-1}"
positive_tags="${RJOB_POSITIVE_TAGS:-}"
mount_arg="--mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot"

refresh_step="${REFRESH_STEP:-80}"
total_max_steps="${REFRESH_TOTAL_MAX_STEPS:-870}"

mkdir -p "$(dirname "$log_path")"
chmod +x "$worker_script"

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
    --set-env "VLLM_SERVER_PORT=${VLLM_SERVER_PORT:-8492}"
    --set-env "MASTER_PORT=${MASTER_PORT:-29700}"
    --set-env "REFRESH_STEP=$refresh_step"
    --set-env "REFRESH_TOTAL_MAX_STEPS=$total_max_steps"
    --set-env "STAGE1_SAVE_STEPS=${STAGE1_SAVE_STEPS:-40}"
    --set-env "STAGE1_SAVE_TOTAL_LIMIT=${STAGE1_SAVE_TOTAL_LIMIT:-4}"
    --set-env "STAGE2_SAVE_STEPS=${STAGE2_SAVE_STEPS:-250}"
    --set-env "STAGE2_SAVE_TOTAL_LIMIT=${STAGE2_SAVE_TOTAL_LIMIT:-8}"
    --set-env "STAGE2_SAVE_ONLY_MODEL=${STAGE2_SAVE_ONLY_MODEL:-true}"
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
    launch_cmd+=(--positive-tags "$positive_tags")
fi
launch_cmd+=(-- bash "$worker_script")

{
    echo "[launch-4b-midrefresh] predict: ${predict_cmd[*]}"
    "${predict_cmd[@]}"

    echo "[launch-4b-midrefresh] run_name=${run_name}"
    echo "[launch-4b-midrefresh] rjob_name=${rjob_name}"
    echo "[launch-4b-midrefresh] log_path=${log_path}"
    echo "[launch-4b-midrefresh] group=${group}"
    echo "[launch-4b-midrefresh] charged_group=${charged_group}"
    echo "[launch-4b-midrefresh] worker_script=${worker_script}"
    echo "[launch-4b-midrefresh] base_model_path=${base_model_path}"
    echo "[launch-4b-midrefresh] train_jsonl=${train_jsonl}"
    echo "[launch-4b-midrefresh] test_jsonl=${test_jsonl}"
    echo "[launch-4b-midrefresh] ckpt_root=${ckpt_root}"
    echo "[launch-4b-midrefresh] artifact_root=${artifact_root}"
    echo "[launch-4b-midrefresh] refresh_step=${refresh_step}"
    echo "[launch-4b-midrefresh] total_max_steps=${total_max_steps}"
    echo "[launch-4b-midrefresh] per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
    echo "[launch-4b-midrefresh] gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-2}"
    echo "[launch-4b-midrefresh] lr=${LR:-2e-6}"
    echo "[launch-4b-midrefresh] command: ${launch_cmd[*]}"
    "${launch_cmd[@]}"
} 2>&1 | tee "$log_path"
