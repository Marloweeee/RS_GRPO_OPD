#!/usr/bin/env bash
set -euo pipefail

cd /data/codes/gui_grounding/GUI-SD-code-main

# Full rs_full-data GRPO+SDPO experiment from the Qwen3-VL-4B-Instruct base model.
# This script only prepares the run configuration; tools/training/run_grpo_sdpo_worker.sh
# owns rollout server startup, training, checkpoint selection, and final greedy eval.

base_model_path=${BASE_MODEL_PATH:-/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-4B-Instruct}
train_jsonl=${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}
test_jsonl=${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}

if [ ! -d "${base_model_path}" ]; then
    echo "[qwen3-4b-base-rsfull] ERROR: base model not found: ${base_model_path}" >&2
    echo "[qwen3-4b-base-rsfull] Set BASE_MODEL_PATH to the local Qwen3-VL-4B-Instruct path." >&2
    exit 1
fi

if [ ! -f "${train_jsonl}" ]; then
    echo "[qwen3-4b-base-rsfull] ERROR: train jsonl not found: ${train_jsonl}" >&2
    exit 1
fi

if [ ! -f "${test_jsonl}" ]; then
    echo "[qwen3-4b-base-rsfull] ERROR: test jsonl not found: ${test_jsonl}" >&2
    exit 1
fi

export RUN_NAME="${RUN_NAME:-gui-sd-qwen3-4b-base_rsfull_grpo_sdpo_topkkl_e1_zero2_bsz2_lr1p5e6_r1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export SAVE_STEPS="${SAVE_STEPS:-250}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"

# Start this experiment from the base model by default. Set ALLOW_RESUME=true and
# RESUME_FROM_CHECKPOINT/AUTO_RESUME explicitly if a failed run should be resumed.
if [ "${ALLOW_RESUME:-false}" != "true" ]; then
    unset RESUME_FROM_CHECKPOINT
    export AUTO_RESUME=false
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-7}"
export VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8192}"

export CKPT_ROOT="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd}"
export MODEL_PATH="${MODEL_PATH:-${base_model_path}}"
export TEACHER_PATH="${TEACHER_PATH:-${base_model_path}}"
export TRAIN_JSONL="${train_jsonl}"
export TEST_JSONL="${test_jsonl}"
export OPSD_MASK_DIR="${OPSD_MASK_DIR:-${ARTIFACT_ROOT}/train_cache/${RUN_NAME}}"
export EVAL_OUT="${EVAL_OUT:-${ARTIFACT_ROOT}/eval/${RUN_NAME}}"

# 4B can carry a larger per-device micro-batch than the 8B runs while keeping the
# same effective global batch as the previous 8B baseline: 7 * 2 * 4 = 56.
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export LR="${LR:-1.5e-6}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero2}"
export TEACHER_DEEPSPEED_CONFIG="${TEACHER_DEEPSPEED_CONFIG:-zero3}"
export OFFLOAD_TEACHER_MODEL="${OFFLOAD_TEACHER_MODEL:-false}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"

export MAX_LENGTH="${MAX_LENGTH:-20000}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-64}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-20000}"
export VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.85}"

export EVAL_TP="${EVAL_TP:-1}"
export EVAL_GPU_MEM_UTIL="${EVAL_GPU_MEM_UTIL:-0.85}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-384}"

export SDPO_LAMBDA="${SDPO_LAMBDA:-0.25}"
export SDPO_TAU_GOOD="${SDPO_TAU_GOOD:-0.5}"
export SDPO_TAU_FAIL="${SDPO_TAU_FAIL:-0.3}"
export SDPO_DELTA="${SDPO_DELTA:-0.5}"
export SDPO_TARGET="${SDPO_TARGET:-rollout}"
export OPSD_TOKEN_WEIGHT_MODE="${OPSD_TOKEN_WEIGHT_MODE:-uniform-entropy}"
export OPSD_NON_DIGIT_WEIGHT="${OPSD_NON_DIGIT_WEIGHT:-0.05}"
export OPSD_MAX_DIGIT_LEN="${OPSD_MAX_DIGIT_LEN:-3}"
export OPSD_MASK_MODE="${OPSD_MASK_MODE:-gaussian}"
export OPSD_HINT_MODE="${OPSD_HINT_MODE:-hint}"
export OPSD_HINT_BOX_COLOR="${OPSD_HINT_BOX_COLOR:-magenta}"
export OPSD_JITTER_RATIO="${OPSD_JITTER_RATIO:-0.2}"
export GRPO_BETA="${GRPO_BETA:-0.04}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-50}"

export PYTHONPATH="/data/codes/gui_grounding/GUI-SD-code-main:${PYTHONPATH:-}"

echo "[qwen3-4b-base-rsfull] RUN_NAME=${RUN_NAME}"
echo "[qwen3-4b-base-rsfull] BASE_MODEL_PATH=${base_model_path}"
echo "[qwen3-4b-base-rsfull] MODEL_PATH=${MODEL_PATH}"
echo "[qwen3-4b-base-rsfull] TEACHER_PATH=${TEACHER_PATH}"
echo "[qwen3-4b-base-rsfull] REF_MODEL_PATH=${MODEL_PATH} (run_grpo_sdpo_worker.sh uses MODEL_PATH as ref_model)"
echo "[qwen3-4b-base-rsfull] TRAIN_JSONL=${TRAIN_JSONL} ($(wc -l < "${TRAIN_JSONL}") samples)"
echo "[qwen3-4b-base-rsfull] TEST_JSONL=${TEST_JSONL} ($(wc -l < "${TEST_JSONL}") samples)"
echo "[qwen3-4b-base-rsfull] CKPT_ROOT=${CKPT_ROOT}"
echo "[qwen3-4b-base-rsfull] ARTIFACT_ROOT=${ARTIFACT_ROOT}"
echo "[qwen3-4b-base-rsfull] OPSD_MASK_DIR=${OPSD_MASK_DIR}"
echo "[qwen3-4b-base-rsfull] EVAL_OUT=${EVAL_OUT}"
echo "[qwen3-4b-base-rsfull] PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "[qwen3-4b-base-rsfull] GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "[qwen3-4b-base-rsfull] LR=${LR}"
echo "[qwen3-4b-base-rsfull] DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"
echo "[qwen3-4b-base-rsfull] TEACHER_DEEPSPEED_CONFIG=${TEACHER_DEEPSPEED_CONFIG}"
echo "[qwen3-4b-base-rsfull] OFFLOAD_TEACHER_MODEL=${OFFLOAD_TEACHER_MODEL}"
echo "[qwen3-4b-base-rsfull] VLLM_GPU_MEMORY_UTIL=${VLLM_GPU_MEMORY_UTIL}"
echo "[qwen3-4b-base-rsfull] EVAL_TP=${EVAL_TP}"
echo "[qwen3-4b-base-rsfull] EVAL_GPU_MEM_UTIL=${EVAL_GPU_MEM_UTIL}"
echo "[qwen3-4b-base-rsfull] EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "[qwen3-4b-base-rsfull] AUTO_RESUME=${AUTO_RESUME:-<unset>}"

bash tools/training/run_grpo_sdpo_worker.sh
