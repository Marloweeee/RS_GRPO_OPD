#!/usr/bin/env bash
# Launch Qwen3-VL-8B SFT on RRSIS-D rs_full with two gui_agent 8-GPU nodes.
#
# This is the first stage for the planned 8B SFT -> GRPO+OPSD run.  It reuses
# the existing two-node SFT worker; the worker is model-size agnostic through
# MODEL_PATH/BASE_MODEL_PATH.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
export EXPERIMENT_SLUG="${EXPERIMENT_SLUG:-rrsisd-8b-sft}"
export RUN_NAME="${RUN_NAME:-gui-sd-qwen3-8b-rrsisd_sft_e1-${timestamp}}"
export RJOB_NAME="${RJOB_NAME:-q8b-rrsisd-sft-2n-${timestamp}}"
export RJOB_LOG_PATH="${RJOB_LOG_PATH:-/data/codes/gui_grounding/data/logs/training/${RJOB_NAME}.rjob.log}"
export WORKER_SCRIPT="${WORKER_SCRIPT:-${repo_root}/tools/training/sft/run_qwen3_8b_rrsisd_sft_2node_worker.sh}"

export RJOB_GROUP="${RJOB_GROUP:-gui_agent}"
export RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-gui_agent}"
export RJOB_CPU="${RJOB_CPU:-80}"
export RJOB_GPU="${RJOB_GPU:-8}"
export RJOB_MEMORY="${RJOB_MEMORY:-900000}"
export RJOB_REPLICA="${RJOB_REPLICA:-2}"
export RJOB_REPLICA_RESTART="${RJOB_REPLICA_RESTART:-never}"
export RJOB_BACKOFF_LIMIT="${RJOB_BACKOFF_LIMIT:-1}"
export RJOB_MAX_WAIT_DURATION="${RJOB_MAX_WAIT_DURATION:-16h0m0s}"
export RJOB_POSITIVE_TAGS="${RJOB_POSITIVE_TAGS:-feature/gpfs=yes}"

# End the local brainctl log follower after both replicas reach the worker's
# normal completion line, so the tmux pane does not keep a finished job alive.
export COMPLETION_GREP_PATTERN="${COMPLETION_GREP_PATTERN:-run completed}"
export COMPLETION_GRACE_SEC="${COMPLETION_GRACE_SEC:-60}"

export BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-8B-Instruct}"
export MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH}"
export CKPT_ROOT="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sft}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sft}"
export TRAIN_JSONL="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}"

export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-30280}"
export RENDEZVOUS_TIMEOUT_SEC="${RENDEZVOUS_TIMEOUT_SEC:-1200}"

export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export SAVE_STEPS="${SAVE_STEPS:-25}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
export TUNER_TYPE="${TUNER_TYPE:-full}"
export ALLOW_RESUME="${ALLOW_RESUME:-false}"
export AUTO_RESUME="${AUTO_RESUME:-false}"

# 8B full SFT uses the same effective batch as the 4B two-node SFT baseline:
# 2 nodes * 8 GPUs * micro-batch 1 * grad-acc 8 = 128 samples/update.
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export LR="${LR:-8e-6}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export MAX_LENGTH="${MAX_LENGTH:-20000}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero2}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"

mkdir -p /data/codes/gui_grounding/data/logs/training

echo "[q8b-sft-launch] timestamp=${timestamp}"
echo "[q8b-sft-launch] run_name=${RUN_NAME}"
echo "[q8b-sft-launch] rjob_name=${RJOB_NAME}"
echo "[q8b-sft-launch] model=${MODEL_PATH}"
echo "[q8b-sft-launch] train_jsonl=${TRAIN_JSONL}"
echo "[q8b-sft-launch] ckpt_root=${CKPT_ROOT}"

bash "${repo_root}/tools/training/launch_qwen3_4b_rsfull_grpo_sdpo_2node_experiment_rjob.sh"
