#!/usr/bin/env bash
set -euo pipefail

cd /data/codes/gui_grounding/GUI-SD-code-main

# Stage-refresh validation:
# continue from the rs_full base e1 checkpoint, and refresh both teacher/ref to
# that same checkpoint for the next 800-step GRPO+SDPO stage.

stage_ckpt=${STAGE_CKPT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-8b-base_rsfull_grpo_srpo_topkkl_e1_zero3_offload_r1/v0-20260520-165157/checkpoint-1740}
if [ ! -d "${stage_ckpt}" ]; then
    echo "[stage-refresh-base-s800] ERROR: stage checkpoint not found: ${stage_ckpt}" >&2
    exit 1
fi

export RUN_NAME="${RUN_NAME:-gui-sd-qwen3-8b-base_rsfull_stage_refresh_s800_trefsync_r1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:-800}"
export SAVE_STEPS="${SAVE_STEPS:-200}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-7}"
export VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8192}"

export CKPT_ROOT="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd}"
export MODEL_PATH="${MODEL_PATH:-${stage_ckpt}}"
export TEACHER_PATH="${TEACHER_PATH:-${stage_ckpt}}"
export TRAIN_JSONL="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}"
export TEST_JSONL="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
export OPSD_MASK_DIR="${OPSD_MASK_DIR:-${ARTIFACT_ROOT}/train_cache/${RUN_NAME}}"
export EVAL_OUT="${EVAL_OUT:-${ARTIFACT_ROOT}/eval/${RUN_NAME}}"
export EVAL_TP="${EVAL_TP:-2}"
export EVAL_GPU_MEM_UTIL="${EVAL_GPU_MEM_UTIL:-0.75}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"

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

echo "[stage-refresh-base-s800] RUN_NAME=${RUN_NAME}"
echo "[stage-refresh-base-s800] STAGE_CKPT=${stage_ckpt}"
echo "[stage-refresh-base-s800] MODEL_PATH=${MODEL_PATH}"
echo "[stage-refresh-base-s800] TEACHER_PATH=${TEACHER_PATH}"
echo "[stage-refresh-base-s800] REF_MODEL_PATH=${MODEL_PATH} (run_grpo_sdpo_worker.sh uses MODEL_PATH as ref_model)"
echo "[stage-refresh-base-s800] MAX_STEPS=${MAX_STEPS}"
echo "[stage-refresh-base-s800] TRAIN_JSONL=${TRAIN_JSONL} ($(wc -l < "${TRAIN_JSONL}") samples)"
echo "[stage-refresh-base-s800] TEST_JSONL=${TEST_JSONL} ($(wc -l < "${TEST_JSONL}") samples)"
echo "[stage-refresh-base-s800] CKPT_ROOT=${CKPT_ROOT}"
echo "[stage-refresh-base-s800] ARTIFACT_ROOT=${ARTIFACT_ROOT}"
echo "[stage-refresh-base-s800] OPSD_MASK_DIR=${OPSD_MASK_DIR}"
echo "[stage-refresh-base-s800] EVAL_OUT=${EVAL_OUT}"
echo "[stage-refresh-base-s800] DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"
echo "[stage-refresh-base-s800] TEACHER_DEEPSPEED_CONFIG=${TEACHER_DEEPSPEED_CONFIG}"
echo "[stage-refresh-base-s800] OFFLOAD_TEACHER_MODEL=${OFFLOAD_TEACHER_MODEL}"
echo "[stage-refresh-base-s800] EVAL_TP=${EVAL_TP}"
echo "[stage-refresh-base-s800] EVAL_GPU_MEM_UTIL=${EVAL_GPU_MEM_UTIL}"
echo "[stage-refresh-base-s800] EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"

bash tools/training/run_grpo_sdpo_worker.sh
