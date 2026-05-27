#!/usr/bin/env bash
# Launch Qwen3-VL-4B train+val GRPO+SDPO mid-refresh s80 experiment on aos.
set -euo pipefail

repo_root="/data/codes/gui_grounding/GUI-SD-code-main"
launcher="${repo_root}/tools/training/launch_qwen3_4b_rsfull_grpo_sdpo_midrefresh_s80_2node_rjob.sh"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
run_name="${RUN_NAME:-gui-sd-qwen3-4b-base_rstrainval_grpo_sdpo_midrefresh_s80_e1-${timestamp}}"
rjob_name="${RJOB_NAME:-qwen3-4b-trainval-midrefresh-s80-2n-${timestamp}}"
log_path="${RJOB_LOG_PATH:-/data/codes/gui_grounding/data/logs/${rjob_name}.rjob.log}"

cd "$repo_root"

RJOB_GROUP="${RJOB_GROUP:-aos}" \
RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-aos}" \
RJOB_CPU="${RJOB_CPU:-24}" \
RJOB_MEMORY="${RJOB_MEMORY:-560000}" \
RJOB_GPU="${RJOB_GPU:-8}" \
RJOB_REPLICA="${RJOB_REPLICA:-2}" \
RJOB_POSITIVE_TAGS="${RJOB_POSITIVE_TAGS:-feature/gpfs=yes}" \
RUN_NAME="$run_name" \
RJOB_NAME="$rjob_name" \
RJOB_LOG_PATH="$log_path" \
TRAIN_JSONL="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train_val.jsonl}" \
TEST_JSONL="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}" \
REFRESH_STEP="${REFRESH_STEP:-80}" \
REFRESH_TOTAL_MAX_STEPS="${REFRESH_TOTAL_MAX_STEPS:-994}" \
STAGE1_SAVE_STEPS="${STAGE1_SAVE_STEPS:-10}" \
STAGE1_SAVE_TOTAL_LIMIT="${STAGE1_SAVE_TOTAL_LIMIT:-20}" \
STAGE2_SAVE_STEPS="${STAGE2_SAVE_STEPS:-10}" \
STAGE2_SAVE_TOTAL_LIMIT="${STAGE2_SAVE_TOTAL_LIMIT:-20}" \
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}" \
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}" \
bash "$launcher"
