#!/usr/bin/env bash
# SFT-initialized MRPD gaussian ablation:
# good/ambiguous use normal GRPO; failed uses OPSD only (0x failed GRPO).
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"

export RUN_TIMESTAMP="$timestamp"
export EXPERIMENT_SLUG="${EXPERIMENT_SLUG:-sft-mrpd-failed-opsd-only-gaussian}"
export RUN_NAME="${RUN_NAME:-gui-sd-qwen3-4b-rrsisd_sft_mrpd_failed_opsd_only_gaussian_e1-${timestamp}}"
export RJOB_NAME="${RJOB_NAME:-q4b-sft-mrpd-fopsdonly-gauss-e1-${timestamp}}"
export TRAIN_RJOB_LOG_PATH="${TRAIN_RJOB_LOG_PATH:-/data/codes/gui_grounding/data/logs/training/${RJOB_NAME}.rjob.log}"
export EVAL_ROOT="${EVAL_ROOT:-/data/codes/gui_grounding/data/logs/eval/${RUN_NAME}_test}"
export EVAL_RJOB_NAME="${EVAL_RJOB_NAME:-eval-sft-mrpd-fopsdonly-gauss-${timestamp}}"

export RJOB_GROUP="${RJOB_GROUP:-gui_agent}"
export RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-gui_agent}"
export RJOB_CPU="${RJOB_CPU:-80}"
export RJOB_MEMORY="${RJOB_MEMORY:-800000}"
export RJOB_GPU="${RJOB_GPU:-8}"
export RJOB_REPLICA="${RJOB_REPLICA:-2}"

export EVAL_RJOB_GROUP="${EVAL_RJOB_GROUP:-gui_agent}"
export EVAL_RJOB_CHARGED_GROUP="${EVAL_RJOB_CHARGED_GROUP:-gui_agent}"
export EVAL_RJOB_CPU="${EVAL_RJOB_CPU:-80}"
export EVAL_RJOB_MEMORY="${EVAL_RJOB_MEMORY:-800000}"

export MASTER_PORT="${MASTER_PORT:-30460}"
export VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-9060}"

export SDPO_DISTILL_SCOPE="${SDPO_DISTILL_SCOPE:-failed}"
export SDPO_GRPO_FAILED_WEIGHT="${SDPO_GRPO_FAILED_WEIGHT:-0.0}"
export SDPO_TEACHER_REFRESH_MODE="${SDPO_TEACHER_REFRESH_MODE:-metric}"
export SDPO_TEACHER_REFRESH_MAX_REFRESHES="${SDPO_TEACHER_REFRESH_MAX_REFRESHES:-1}"
export SDPO_TEACHER_REFRESH_COOLDOWN_STEPS="${SDPO_TEACHER_REFRESH_COOLDOWN_STEPS:-0}"
export OPSD_MASK_MODE="${OPSD_MASK_MODE:-gaussian}"
export OPSD_HINT_MODE="${OPSD_HINT_MODE:-hint}"
export OPSD_EMA_DECAY="${OPSD_EMA_DECAY:-0.0}"

bash "${script_dir}/run_qwen3_4b_rrsisd_sft_to_grpo_opsd_metric_gaussian_gui_agent.sh"
