#!/usr/bin/env bash
# Experiment B: zoom_in mid-training teacher/ref refresh.
#
# Stage 1 trains to checkpoint-80 with full trainer state. Stage 2 reloads
# checkpoint-80 as student/ref/teacher and resumes trainer state, continuing
# from global step 80 to the one-epoch target instead of replaying the dataset.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"

export EXPERIMENT_SLUG="${EXPERIMENT_SLUG:-zoom-midrefresh-s80}"
export RUN_TIMESTAMP="$timestamp"
export RUN_NAME="${RUN_NAME:-gui-sd-qwen3-4b-rsfull_grpo_sdpo_zoom_midrefresh_s80_e1-${timestamp}}"
export RJOB_NAME="${RJOB_NAME:-q4b-zoom-midrefresh-s80-2n-${timestamp}}"
export WORKER_SCRIPT="${WORKER_SCRIPT:-${script_dir}/run_qwen3_4b_rsfull_grpo_sdpo_midrefresh_s80_2node_worker.sh}"

export OPSD_MASK_MODE="${OPSD_MASK_MODE:-zoom_in}"
export OPSD_HINT_MODE="${OPSD_HINT_MODE:-hint}"
export REFRESH_STEP="${REFRESH_STEP:-80}"
export REFRESH_TOTAL_MAX_STEPS="${REFRESH_TOTAL_MAX_STEPS:-870}"
export STAGE1_SAVE_STEPS="${STAGE1_SAVE_STEPS:-10}"
export STAGE1_SAVE_TOTAL_LIMIT="${STAGE1_SAVE_TOTAL_LIMIT:-4}"
export STAGE2_SAVE_STEPS="${STAGE2_SAVE_STEPS:-10}"
export STAGE2_SAVE_TOTAL_LIMIT="${STAGE2_SAVE_TOTAL_LIMIT:-12}"
export STAGE2_SAVE_ONLY_MODEL="${STAGE2_SAVE_ONLY_MODEL:-true}"
export STAGE2_SKIP_EVAL="${STAGE2_SKIP_EVAL:-true}"
export VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8522}"
export MASTER_PORT="${MASTER_PORT:-29522}"
export RJOB_GROUP="${RJOB_GROUP:-gui_agent}"
export RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-gui_agent}"
export COMPLETION_GREP_PATTERN="${COMPLETION_GREP_PATTERN:-mid-refresh run completed}"

exec "${script_dir}/launch_qwen3_4b_rsfull_grpo_sdpo_2node_experiment_rjob.sh"
