#!/usr/bin/env bash
# SFT-initialized GRPO+OPSD gaussian ablation: fixed teacher, no refresh.
#
# Purpose:
#   Baseline for teacher-refresh studies. The teacher starts from the same SFT
#   checkpoint and remains unchanged: no hard refresh and no EMA.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"

export RUN_TIMESTAMP="$timestamp"
export EXPERIMENT_SLUG="${EXPERIMENT_SLUG:-sft-grpo-opsd-no-refresh-gaussian}"
export RUN_NAME="${RUN_NAME:-gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_no_refresh_gaussian_e1-${timestamp}}"
export RJOB_NAME="${RJOB_NAME:-q4b-sftopsd-norefresh-gauss-e1-${timestamp}}"
export TRAIN_RJOB_LOG_PATH="${TRAIN_RJOB_LOG_PATH:-/data/codes/gui_grounding/data/logs/training/${RJOB_NAME}.rjob.log}"
export EVAL_ROOT="${EVAL_ROOT:-/data/codes/gui_grounding/data/logs/eval/${RUN_NAME}_test}"
export EVAL_RJOB_NAME="${EVAL_RJOB_NAME:-eval-sftopsd-norefresh-gauss-${timestamp}}"

export RJOB_GROUP="${RJOB_GROUP:-gui_agent}"
export RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-gui_agent}"
export RJOB_CPU="${RJOB_CPU:-64}"
export RJOB_MEMORY="${RJOB_MEMORY:-700000}"
export RJOB_GPU="${RJOB_GPU:-8}"
export RJOB_REPLICA="${RJOB_REPLICA:-2}"

export EVAL_RJOB_GROUP="${EVAL_RJOB_GROUP:-gui_agent}"
export EVAL_RJOB_CHARGED_GROUP="${EVAL_RJOB_CHARGED_GROUP:-gui_agent}"
export EVAL_RJOB_CPU="${EVAL_RJOB_CPU:-28}"
export EVAL_RJOB_MEMORY="${EVAL_RJOB_MEMORY:-600000}"

export MASTER_PORT="${MASTER_PORT:-30330}"
export VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-9012}"

export LR="${LR:-1e-6}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.01}"
export SAVE_STEPS="${SAVE_STEPS:-10}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
export OPSD_MASK_MODE="${OPSD_MASK_MODE:-gaussian}"
export OPSD_HINT_MODE="${OPSD_HINT_MODE:-hint}"
export OPSD_EMA_DECAY="${OPSD_EMA_DECAY:-0.0}"
export SDPO_TEACHER_REFRESH_MODE="${SDPO_TEACHER_REFRESH_MODE:-off}"
export SDPO_TEACHER_REFRESH_STEP="${SDPO_TEACHER_REFRESH_STEP:--1}"

bash "${script_dir}/run_qwen3_4b_rrsisd_sft_to_grpo_opsd_metric_gaussian_gui_agent.sh"
