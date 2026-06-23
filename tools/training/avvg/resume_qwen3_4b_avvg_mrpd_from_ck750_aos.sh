#!/usr/bin/env bash
# Continue the AVVG-refGeo MRPD stage on AOS from the last saved gui_agent
# checkpoint before the gui_agent quota became unavailable.
#
# The interrupted gui_agent run used save_only_model=true, so this script treats
# checkpoint-750 as the new student/teacher/rollout/ref initialization and runs
# the remaining steps in a fresh output directory.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
source_ckpt="${SOURCE_CKPT:-/mnt/jfs/copilot/lhb/checkpoint/rs/avvg-mrpd/gui-sd-qwen3-4b-avvg_refgeo_sft_mrpd_metric_gaussian_e1-20260527-2043-avvgmrpd/v0-20260527-204813/checkpoint-750}"
train_jsonl="${TRAIN_JSONL:-/data/codes/gui_grounding/data/avvg_refgeo_full/avvg_train.jsonl}"
test_jsonl="${TEST_JSONL:-/data/codes/gui_grounding/data/avvg_refgeo_full/avvg_test.jsonl}"

run_name="${RUN_NAME:-gui-sd-qwen3-4b-avvg_refgeo_sft_mrpd_metric_gaussian_e1_resume750_aos-${timestamp}}"
rjob_name="${RJOB_NAME:-q4b-avvg-mrpd-resume750-aos-${timestamp}}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs}"
train_log="${RJOB_LOG_PATH:-${log_root}/training/${rjob_name}.rjob.log}"

mkdir -p "${log_root}/training"

echo "[avvg-mrpd-aos-resume] timestamp=${timestamp}"
echo "[avvg-mrpd-aos-resume] source_ckpt=${source_ckpt}"
echo "[avvg-mrpd-aos-resume] run_name=${run_name}"
echo "[avvg-mrpd-aos-resume] rjob_name=${rjob_name}"
echo "[avvg-mrpd-aos-resume] train_jsonl=${train_jsonl}"
echo "[avvg-mrpd-aos-resume] test_jsonl=${test_jsonl}"
echo "[avvg-mrpd-aos-resume] train_log=${train_log}"

RUN_TIMESTAMP="$timestamp" \
EXPERIMENT_SLUG="avvg-refgeo-mrpd-resume750-aos" \
RUN_NAME="$run_name" \
RJOB_NAME="$rjob_name" \
RJOB_LOG_PATH="$train_log" \
WORKER_SCRIPT="${repo_root}/tools/training/run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh" \
RJOB_GROUP="${RJOB_GROUP:-aos}" \
RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-aos}" \
RJOB_CPU="${RJOB_CPU:-64}" \
RJOB_GPU="${RJOB_GPU:-8}" \
RJOB_MEMORY="${RJOB_MEMORY:-800000}" \
RJOB_REPLICA="${RJOB_REPLICA:-2}" \
RJOB_REPLICA_RESTART="${RJOB_REPLICA_RESTART:-never}" \
RJOB_BACKOFF_LIMIT="${RJOB_BACKOFF_LIMIT:-1}" \
RJOB_MAX_WAIT_DURATION="${RJOB_MAX_WAIT_DURATION:-24h0m0s}" \
RJOB_POSITIVE_TAGS="${RJOB_POSITIVE_TAGS:-feature/gpfs=yes}" \
COMPLETION_GREP_PATTERN="\\[4b-2node-worker\\].*completed" \
COMPLETION_GRACE_SEC="${COMPLETION_GRACE_SEC:-60}" \
BASE_MODEL_PATH="$source_ckpt" \
MODEL_PATH="$source_ckpt" \
TEACHER_PATH="$source_ckpt" \
ROLLOUT_MODEL_PATH="$source_ckpt" \
CKPT_ROOT="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/avvg-mrpd}" \
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/avvg-mrpd}" \
TRAIN_JSONL="$train_jsonl" \
TEST_JSONL="$test_jsonl" \
NNODES="${NNODES:-2}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-7}" \
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}" \
ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-7}" \
MASTER_PORT="${MASTER_PORT:-30720}" \
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-9072}" \
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}" \
MAX_STEPS="${MAX_STEPS:-1141}" \
SAVE_STEPS="${SAVE_STEPS:-10}" \
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}" \
SAVE_ONLY_MODEL=true \
TUNER_TYPE=full \
ALLOW_RESUME=false \
AUTO_RESUME=false \
SKIP_EVAL=true \
STAGE2_SKIP_EVAL=true \
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}" \
LR="${LR:-1e-6}" \
WARMUP_RATIO="${WARMUP_RATIO:-0.01}" \
NUM_GENERATIONS="${NUM_GENERATIONS:-8}" \
NUM_ITERATIONS="${NUM_ITERATIONS:-1}" \
MAX_LENGTH="${MAX_LENGTH:-20000}" \
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-64}" \
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero2}" \
TEACHER_DEEPSPEED_CONFIG="${TEACHER_DEEPSPEED_CONFIG:-zero3}" \
OFFLOAD_TEACHER_MODEL="${OFFLOAD_TEACHER_MODEL:-false}" \
SDPO_LAMBDA="${SDPO_LAMBDA:-0.25}" \
SDPO_TAU_GOOD="${SDPO_TAU_GOOD:-0.5}" \
SDPO_TAU_FAIL="${SDPO_TAU_FAIL:-0.3}" \
SDPO_DELTA="${SDPO_DELTA:-0.5}" \
SDPO_TARGET="${SDPO_TARGET:-rollout}" \
SDPO_TEACHER_REFRESH_MODE="${SDPO_TEACHER_REFRESH_MODE:-metric}" \
SDPO_TEACHER_REFRESH_STEP="${SDPO_TEACHER_REFRESH_STEP:--1}" \
SDPO_TEACHER_REFRESH_WARMUP="${SDPO_TEACHER_REFRESH_WARMUP:-80}" \
SDPO_TEACHER_REFRESH_WINDOW="${SDPO_TEACHER_REFRESH_WINDOW:-50}" \
SDPO_TEACHER_REFRESH_CHECK_INTERVAL="${SDPO_TEACHER_REFRESH_CHECK_INTERVAL:-10}" \
SDPO_TEACHER_REFRESH_MIN_IOU_IMPROVE="${SDPO_TEACHER_REFRESH_MIN_IOU_IMPROVE:-0.01}" \
SDPO_TEACHER_REFRESH_MAX_FAILED="${SDPO_TEACHER_REFRESH_MAX_FAILED:-0.55}" \
SDPO_TEACHER_REFRESH_MIN_SDPO_LOSS="${SDPO_TEACHER_REFRESH_MIN_SDPO_LOSS:-0.02}" \
SDPO_TEACHER_REFRESH_MAX_KL="${SDPO_TEACHER_REFRESH_MAX_KL:-0.30}" \
OPSD_MASK_MODE="${OPSD_MASK_MODE:-gaussian}" \
OPSD_HINT_MODE="${OPSD_HINT_MODE:-hint}" \
OPSD_TOKEN_WEIGHT_MODE="${OPSD_TOKEN_WEIGHT_MODE:-uniform-entropy}" \
OPSD_NON_DIGIT_WEIGHT="${OPSD_NON_DIGIT_WEIGHT:-0.05}" \
OPSD_MAX_DIGIT_LEN="${OPSD_MAX_DIGIT_LEN:-3}" \
OPSD_EMA_DECAY="${OPSD_EMA_DECAY:-0.0}" \
OPSD_ZOOM_RATIO="${OPSD_ZOOM_RATIO:-2.0}" \
OPSD_MIN_AREA_FRAC="${OPSD_MIN_AREA_FRAC:-0.1}" \
OPSD_GAUSSIAN_SIGMA_RATIO="${OPSD_GAUSSIAN_SIGMA_RATIO:-1.5}" \
OPSD_HINT_BOX_COLOR="${OPSD_HINT_BOX_COLOR:-magenta}" \
OPSD_JITTER_RATIO="${OPSD_JITTER_RATIO:-0.2}" \
GRPO_BETA="${GRPO_BETA:-0.04}" \
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}" \
TOP_P="${TOP_P:-0.95}" \
TOP_K="${TOP_K:-50}" \
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.82}" \
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-20000}" \
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}" \
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}" \
bash "${repo_root}/tools/training/launch_qwen3_4b_rsfull_grpo_sdpo_2node_experiment_rjob.sh"
