#!/usr/bin/env bash
# Train-only rollout-count ablation for SFT-initialized MRPD.
# The only experimental variable should be NUM_GENERATIONS.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <num_generations: 2|4|16>" >&2
    exit 2
fi

num_generations="$1"
case "$num_generations" in
    2|4|16) ;;
    *)
        echo "ERROR: num_generations must be one of 2, 4, 16; got ${num_generations}" >&2
        exit 2
        ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
source_ckpt="${SOURCE_CKPT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sft/gui-sd-qwen3-4b-rrsisd_sft_e1-20260526-220915/v0-20260526-220853/checkpoint-96}"
experiment_slug="rollout-n-ablation-ngen${num_generations}"
run_name="${RUN_NAME:-gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_metric_gaussian_ngen${num_generations}_e1-${timestamp}}"
rjob_name="${RJOB_NAME:-q4b-rrsisd-ngen${num_generations}-${timestamp}}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs/rollout_n_ablation}"
train_log="${RJOB_LOG_PATH:-${log_root}/${rjob_name}.rjob.log}"
env_log="${log_root}/${rjob_name}.env"
preflight_log="${log_root}/${rjob_name}.rjob-predict.log"
ckpt_root="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"

mkdir -p "$log_root"

if [ ! -d "$source_ckpt" ]; then
    echo "WARNING: SOURCE_CKPT is not visible on the submit host: $source_ckpt" >&2
    echo "WARNING: continuing because the rjob container mounts /mnt/jfs/copilot." >&2
fi

export RUN_TIMESTAMP="$timestamp"
export EXPERIMENT_SLUG="$experiment_slug"
export RUN_NAME="$run_name"
export RJOB_NAME="$rjob_name"
export RJOB_LOG_PATH="$train_log"
export WORKER_SCRIPT="${WORKER_SCRIPT:-${repo_root}/tools/training/run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh}"

export RJOB_GROUP="${RJOB_GROUP:-aos}"
export RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-aos}"
export RJOB_CPU="${RJOB_CPU:-64}"
export RJOB_GPU="${RJOB_GPU:-8}"
export RJOB_MEMORY="${RJOB_MEMORY:-800000}"
export RJOB_REPLICA="${RJOB_REPLICA:-2}"
export RJOB_REPLICA_RESTART="${RJOB_REPLICA_RESTART:-never}"
export RJOB_BACKOFF_LIMIT="${RJOB_BACKOFF_LIMIT:-1}"
export RJOB_MAX_WAIT_DURATION="${RJOB_MAX_WAIT_DURATION:-24h0m0s}"
export RJOB_POSITIVE_TAGS="${RJOB_POSITIVE_TAGS:-feature/gpfs=yes}"

# Keep the local log follower from holding a finished rjob open.
export COMPLETION_GREP_PATTERN="${COMPLETION_GREP_PATTERN:-\\[4b-2node-worker\\].*completed}"
export COMPLETION_GRACE_SEC="${COMPLETION_GRACE_SEC:-60}"

# Student, teacher, ref, and rollout server all start from the same clean SFT ckpt.
export BASE_MODEL_PATH="$source_ckpt"
export MODEL_PATH="$source_ckpt"
export REF_MODEL_PATH="$source_ckpt"
export TEACHER_PATH="$source_ckpt"
export ROLLOUT_MODEL_PATH="$source_ckpt"
export CKPT_ROOT="$ckpt_root"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd}"
export TRAIN_JSONL="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}"
export TEST_JSONL="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"

export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-7}"
export VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8892}"
export MASTER_PORT="${MASTER_PORT:-30120}"
export ROLLOUT_READY_TIMEOUT_SEC="${ROLLOUT_READY_TIMEOUT_SEC:-900}"
export RENDEZVOUS_TIMEOUT_SEC="${RENDEZVOUS_TIMEOUT_SEC:-1200}"

export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export SAVE_STEPS="${SAVE_STEPS:-10}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
export TUNER_TYPE="${TUNER_TYPE:-full}"
export ALLOW_RESUME="${ALLOW_RESUME:-false}"
export AUTO_RESUME="${AUTO_RESUME:-false}"
export SKIP_EVAL="${SKIP_EVAL:-true}"
export STAGE2_SKIP_EVAL="${STAGE2_SKIP_EVAL:-true}"

export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export LR="${LR:-1e-6}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.01}"
export NUM_GENERATIONS="$num_generations"
export NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
export MAX_LENGTH="${MAX_LENGTH:-20000}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-64}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero2}"
export TEACHER_DEEPSPEED_CONFIG="${TEACHER_DEEPSPEED_CONFIG:-zero3}"
export OFFLOAD_TEACHER_MODEL="${OFFLOAD_TEACHER_MODEL:-false}"

export SDPO_LAMBDA="${SDPO_LAMBDA:-0.25}"
export SDPO_TAU_GOOD="${SDPO_TAU_GOOD:-0.5}"
export SDPO_TAU_FAIL="${SDPO_TAU_FAIL:-0.3}"
export SDPO_DELTA="${SDPO_DELTA:-0.5}"
export SDPO_GRPO_FAILED_WEIGHT="${SDPO_GRPO_FAILED_WEIGHT:-1.0}"
export SDPO_DISTILL_SCOPE="${SDPO_DISTILL_SCOPE:-failed}"
export SDPO_TARGET="${SDPO_TARGET:-rollout}"
export SDPO_HINT_SOURCE="${SDPO_HINT_SOURCE:-gt}"
export SDPO_SIBLING_SELECT_METRIC="${SDPO_SIBLING_SELECT_METRIC:-reward}"
export SDPO_SIBLING_FALLBACK="${SDPO_SIBLING_FALLBACK:-gt}"
export SDPO_TEACHER_REFRESH_MODE="${SDPO_TEACHER_REFRESH_MODE:-metric}"
export SDPO_TEACHER_REFRESH_STEP="${SDPO_TEACHER_REFRESH_STEP:--1}"
export SDPO_TEACHER_REFRESH_WARMUP="${SDPO_TEACHER_REFRESH_WARMUP:-80}"
export SDPO_TEACHER_REFRESH_WINDOW="${SDPO_TEACHER_REFRESH_WINDOW:-50}"
export SDPO_TEACHER_REFRESH_CHECK_INTERVAL="${SDPO_TEACHER_REFRESH_CHECK_INTERVAL:-10}"
export SDPO_TEACHER_REFRESH_MIN_IOU_IMPROVE="${SDPO_TEACHER_REFRESH_MIN_IOU_IMPROVE:-0.01}"
export SDPO_TEACHER_REFRESH_MAX_FAILED="${SDPO_TEACHER_REFRESH_MAX_FAILED:-0.55}"
export SDPO_TEACHER_REFRESH_MIN_SDPO_LOSS="${SDPO_TEACHER_REFRESH_MIN_SDPO_LOSS:-0.02}"
export SDPO_TEACHER_REFRESH_MAX_KL="${SDPO_TEACHER_REFRESH_MAX_KL:-0.30}"
export SDPO_TEACHER_REFRESH_MAX_REFRESHES="${SDPO_TEACHER_REFRESH_MAX_REFRESHES:-1}"

export OPSD_MASK_MODE="${OPSD_MASK_MODE:-gaussian}"
export OPSD_HINT_MODE="${OPSD_HINT_MODE:-hint}"
export OPSD_TOKEN_WEIGHT_MODE="${OPSD_TOKEN_WEIGHT_MODE:-uniform-entropy}"
export OPSD_NON_DIGIT_WEIGHT="${OPSD_NON_DIGIT_WEIGHT:-0.05}"
export OPSD_MAX_DIGIT_LEN="${OPSD_MAX_DIGIT_LEN:-3}"
export OPSD_EMA_DECAY="${OPSD_EMA_DECAY:-0.0}"
export OPSD_ZOOM_RATIO="${OPSD_ZOOM_RATIO:-2.0}"
export OPSD_MIN_AREA_FRAC="${OPSD_MIN_AREA_FRAC:-0.1}"
export OPSD_GAUSSIAN_SIGMA_RATIO="${OPSD_GAUSSIAN_SIGMA_RATIO:-1.5}"
export OPSD_HINT_BOX_COLOR="${OPSD_HINT_BOX_COLOR:-magenta}"
export OPSD_JITTER_RATIO="${OPSD_JITTER_RATIO:-0.2}"

export GRPO_BETA="${GRPO_BETA:-0.04}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-50}"
export VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.82}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-20000}"
unset VLLM_MAX_NUM_SEQS
unset VLLM_ENFORCE_EAGER

cat > "$env_log" <<EOF
RUN_TIMESTAMP=${RUN_TIMESTAMP}
RUN_NAME=${RUN_NAME}
RJOB_NAME=${RJOB_NAME}
RJOB_GROUP=${RJOB_GROUP}
RJOB_CHARGED_GROUP=${RJOB_CHARGED_GROUP}
RJOB_REPLICA=${RJOB_REPLICA}
RJOB_GPU=${RJOB_GPU}
RJOB_CPU=${RJOB_CPU}
RJOB_MEMORY=${RJOB_MEMORY}
SOURCE_CKPT=${source_ckpt}
TRAIN_JSONL=${TRAIN_JSONL}
CKPT_ROOT=${CKPT_ROOT}
NUM_GENERATIONS=${NUM_GENERATIONS}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
LR=${LR}
SDPO_DISTILL_SCOPE=${SDPO_DISTILL_SCOPE}
SDPO_LAMBDA=${SDPO_LAMBDA}
SDPO_TEACHER_REFRESH_MODE=${SDPO_TEACHER_REFRESH_MODE}
OPSD_MASK_MODE=${OPSD_MASK_MODE}
OPSD_TOKEN_WEIGHT_MODE=${OPSD_TOKEN_WEIGHT_MODE}
SKIP_EVAL=${SKIP_EVAL}
STAGE2_SKIP_EVAL=${STAGE2_SKIP_EVAL}
EOF

echo "[rollout-n-ablation] timestamp=${timestamp}"
echo "[rollout-n-ablation] num_generations=${NUM_GENERATIONS}"
echo "[rollout-n-ablation] source_ckpt=${source_ckpt}"
echo "[rollout-n-ablation] run_name=${run_name}"
echo "[rollout-n-ablation] rjob_name=${rjob_name}"
echo "[rollout-n-ablation] train_log=${train_log}"
echo "[rollout-n-ablation] env_log=${env_log}"
echo "[rollout-n-ablation] brainctl-rjob predict log=${preflight_log}"

set +e
brainctl rjob launch \
    -P "$RJOB_REPLICA" \
    --replica-prefix \
    --cpu "$RJOB_CPU" \
    --gpu "$RJOB_GPU" \
    --memory "$RJOB_MEMORY" \
    --charged-group="$RJOB_CHARGED_GROUP" \
    --private-machine=group \
    --positive-tags "$RJOB_POSITIVE_TAGS" \
    --mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot \
    --custom-resources rdma/mlnx_shared=8 \
    --predict-only \
    -- bash -lc 'true' 2>&1 | tee "$preflight_log"
predict_status=${PIPESTATUS[0]}
set -e
if [ "$predict_status" -ne 0 ]; then
    echo "[rollout-n-ablation] WARNING: brainctl rjob predict-only exited with status=${predict_status}; continuing to the existing launcher, which performs its own predict-only check." >&2
fi

bash "${repo_root}/tools/training/launch_qwen3_4b_rsfull_grpo_sdpo_2node_experiment_rjob.sh"

echo "[rollout-n-ablation] training launcher exited for ${run_name}"
echo "[rollout-n-ablation] expected checkpoint root: ${ckpt_root}/${run_name}"
