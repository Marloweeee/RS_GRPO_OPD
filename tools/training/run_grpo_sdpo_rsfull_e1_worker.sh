#!/usr/bin/env bash
set -euo pipefail

cd /data/codes/gui_grounding/GUI-SD-code-main

# Full rs_full-data checkpoint-80 warm-start GRPO+SDPO experiment.
# Default is one full pass over /data/codes/gui_grounding/data/rs_full/rs_train.jsonl.

export RUN_NAME="${RUN_NAME:-gui-sd-qwen3-8b-ck80_rsfull_grpo_sdpo_topkkl_e1_zero3_offload_r1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export SAVE_STEPS="${SAVE_STEPS:-250}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-7}"

export CKPT_ROOT="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"
export MODEL_PATH="${MODEL_PATH:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-4b_student_rs_full_legacy/v0-20260517-230632/checkpoint-80}"
export TEACHER_PATH="${TEACHER_PATH:-${MODEL_PATH}}"
export TRAIN_JSONL="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}"
export TEST_JSONL="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
export EVAL_OUT="${EVAL_OUT:-/data/codes/gui_grounding/data/rs_full/eval_${RUN_NAME}}"
export EVAL_TP="${EVAL_TP:-2}"
export EVAL_GPU_MEM_UTIL="${EVAL_GPU_MEM_UTIL:-0.75}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"

# Keep the stable 8B training setup used by the successful 1/10-data runs.
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero3}"
export TEACHER_DEEPSPEED_CONFIG="${TEACHER_DEEPSPEED_CONFIG:-zero3}"
export OFFLOAD_TEACHER_MODEL="${OFFLOAD_TEACHER_MODEL:-true}"

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
export LR="${LR:-1e-6}"
export GRPO_BETA="${GRPO_BETA:-0.04}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-50}"

export PYTHONPATH="/data/codes/gui_grounding/GUI-SD-code-main:${PYTHONPATH:-}"

echo "[ck80-rsfull-grpo-sdpo] RUN_NAME=${RUN_NAME}"
echo "[ck80-rsfull-grpo-sdpo] NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "[ck80-rsfull-grpo-sdpo] MODEL_PATH=${MODEL_PATH}"
echo "[ck80-rsfull-grpo-sdpo] TEACHER_PATH=${TEACHER_PATH}"
echo "[ck80-rsfull-grpo-sdpo] TRAIN_JSONL=${TRAIN_JSONL} ($(wc -l < "${TRAIN_JSONL}") samples)"
echo "[ck80-rsfull-grpo-sdpo] TEST_JSONL=${TEST_JSONL} ($(wc -l < "${TEST_JSONL}") samples)"
echo "[ck80-rsfull-grpo-sdpo] EVAL_OUT=${EVAL_OUT}"
echo "[ck80-rsfull-grpo-sdpo] EVAL_TP=${EVAL_TP}"
echo "[ck80-rsfull-grpo-sdpo] EVAL_GPU_MEM_UTIL=${EVAL_GPU_MEM_UTIL}"
echo "[ck80-rsfull-grpo-sdpo] EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"

bash tools/training/run_grpo_sdpo_worker.sh
