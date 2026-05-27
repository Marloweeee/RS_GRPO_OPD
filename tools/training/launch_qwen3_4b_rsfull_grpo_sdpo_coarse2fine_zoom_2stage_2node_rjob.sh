#!/usr/bin/env bash
# Experiment C: coarse-to-fine staged restart.
#
# Stage 1 uses gaussian visual hints for 80 steps, then Stage 2 refreshes
# student/ref/teacher from checkpoint-80 and restarts full-data training with
# zoom_in hints. This keeps the early policy less brittle while using zoom_in
# for late bbox refinement.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"

export EXPERIMENT_SLUG="${EXPERIMENT_SLUG:-coarse2fine-zoom-s80}"
export RUN_TIMESTAMP="$timestamp"
export RUN_NAME="${RUN_NAME:-gui-sd-qwen3-4b-rsfull_grpo_sdpo_coarse2fine_gaussian80_zoom_e1-${timestamp}}"
export RJOB_NAME="${RJOB_NAME:-q4b-coarse2fine-zoom-s80-2n-${timestamp}}"
export WORKER_SCRIPT="${WORKER_SCRIPT:-${script_dir}/run_qwen3_4b_rsfull_grpo_sdpo_coarse2fine_zoom_2stage_2node_worker.sh}"

export STAGE1_OPSD_MASK_MODE="${STAGE1_OPSD_MASK_MODE:-gaussian}"
export STAGE2_OPSD_MASK_MODE="${STAGE2_OPSD_MASK_MODE:-zoom_in}"
export STAGE1_OPSD_HINT_MODE="${STAGE1_OPSD_HINT_MODE:-hint}"
export STAGE2_OPSD_HINT_MODE="${STAGE2_OPSD_HINT_MODE:-hint}"
export OPSD_MASK_MODE="${OPSD_MASK_MODE:-zoom_in}"
export OPSD_HINT_MODE="${OPSD_HINT_MODE:-hint}"
export STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-80}"
export STAGE1_SAVE_STEPS="${STAGE1_SAVE_STEPS:-10}"
export STAGE1_SAVE_TOTAL_LIMIT="${STAGE1_SAVE_TOTAL_LIMIT:-12}"
export STAGE2_NUM_TRAIN_EPOCHS="${STAGE2_NUM_TRAIN_EPOCHS:-1}"
export STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:--1}"
export STAGE2_SAVE_STEPS="${STAGE2_SAVE_STEPS:-10}"
export STAGE2_SAVE_TOTAL_LIMIT="${STAGE2_SAVE_TOTAL_LIMIT:-12}"
export STAGE2_SKIP_EVAL="${STAGE2_SKIP_EVAL:-true}"
export VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8523}"
export MASTER_PORT="${MASTER_PORT:-29523}"
export RJOB_GROUP="${RJOB_GROUP:-gui_agent}"
export RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-gui_agent}"
export COMPLETION_GREP_PATTERN="${COMPLETION_GREP_PATTERN:-coarse-to-fine run completed}"

exec "${script_dir}/launch_qwen3_4b_rsfull_grpo_sdpo_2node_experiment_rjob.sh"
