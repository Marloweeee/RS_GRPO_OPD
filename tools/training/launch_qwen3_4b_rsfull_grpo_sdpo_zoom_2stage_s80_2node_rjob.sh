#!/usr/bin/env bash
# Experiment A: zoom_in staged restart.
#
# Stage 1 trains 80 steps from the base 4B model. Stage 2 loads checkpoint-80
# as student/ref/teacher and restarts full rs_train training from the beginning.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"

export EXPERIMENT_SLUG="${EXPERIMENT_SLUG:-zoom-2stage-s80}"
export RUN_TIMESTAMP="$timestamp"
export RUN_NAME="${RUN_NAME:-gui-sd-qwen3-4b-rsfull_grpo_sdpo_zoom_2stage_s80_e1-${timestamp}}"
export RJOB_NAME="${RJOB_NAME:-q4b-zoom-2stage-s80-2n-${timestamp}}"
export WORKER_SCRIPT="${WORKER_SCRIPT:-${script_dir}/run_qwen3_4b_rsfull_grpo_sdpo_2stage_2node_worker.sh}"

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
export VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8521}"
export MASTER_PORT="${MASTER_PORT:-29521}"
export RJOB_GROUP="${RJOB_GROUP:-gui_agent}"
export RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-gui_agent}"
export COMPLETION_GREP_PATTERN="${COMPLETION_GREP_PATTERN:-two-stage run completed}"

exec "${script_dir}/launch_qwen3_4b_rsfull_grpo_sdpo_2node_experiment_rjob.sh"
