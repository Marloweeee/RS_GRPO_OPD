#!/usr/bin/env bash
# RSVG-DIOR full-object experiment pipeline:
#   1. SFT Qwen3-VL-4B from the original base checkpoint on rsvg_train.jsonl.
#   2. Continue from the SFT checkpoint with the current best stable MRPD setup
#      (GRPO + OPSD, metric teacher refresh, gaussian hint) on rsvg_train.jsonl.
#   3. Launch a detached greedy evaluation on rsvg_test.jsonl for the MRPD ckpt.
#
# The pipeline is intended to run inside a tmux session on the submit host. The
# actual training/eval jobs are submitted as AOS rjobs, so GPU resources are
# released by the cluster once each scripted job exits.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs}"
train_log_root="${TRAIN_LOG_ROOT:-${log_root}/training}"
eval_log_root="${EVAL_LOG_ROOT:-${log_root}/eval}"
mkdir -p "$train_log_root" "$eval_log_root"

base_model="${BASE_MODEL_PATH:-/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-4B-Instruct}"
train_jsonl="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rsvg_dior_full/rsvg_train.jsonl}"
val_jsonl="${VAL_JSONL:-/data/codes/gui_grounding/data/rsvg_dior_full/rsvg_val.jsonl}"
test_jsonl="${TEST_JSONL:-/data/codes/gui_grounding/data/rsvg_dior_full/rsvg_test.jsonl}"

sft_run_name="${SFT_RUN_NAME:-gui-sd-qwen3-4b-rsvg_dior_sft_e1-${timestamp}}"
mrpd_run_name="${MRPD_RUN_NAME:-gui-sd-qwen3-4b-rsvg_dior_sft_mrpd_metric_gaussian_e1-${timestamp}}"
sft_rjob_name="${SFT_RJOB_NAME:-q4b-rsvg-sft-2n-${timestamp}}"
mrpd_rjob_name="${MRPD_RJOB_NAME:-q4b-rsvg-sftmrpd-2n-${timestamp}}"

sft_ckpt_root="${SFT_CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rsvg-sft}"
mrpd_ckpt_root="${MRPD_CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rsvg-mrpd}"
sft_artifact_root="${SFT_ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rsvg-sft}"
mrpd_artifact_root="${MRPD_ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rsvg-mrpd}"

sft_log="${SFT_RJOB_LOG_PATH:-${train_log_root}/${sft_rjob_name}.rjob.log}"
mrpd_log="${MRPD_RJOB_LOG_PATH:-${train_log_root}/${mrpd_rjob_name}.rjob.log}"
pipeline_log="${PIPELINE_LOG_PATH:-${train_log_root}/q4b-rsvg-sft-then-mrpd-${timestamp}.pipeline.log}"

latest_checkpoint() {
    local root="$1"
    local run_name="$2"
    local log_path="${3:-}"
    local ckpt
    ckpt="$(
        find "${root}/${run_name}" -path '*/checkpoint-*' -type d -printf '%f %p\n' 2>/dev/null \
            | awk '{step=$1; sub(/^checkpoint-/, "", step); if (step ~ /^[0-9]+$/) print step " " $2}' \
            | sort -n \
            | tail -1 \
            | cut -d' ' -f2- || true
    )"
    if [ -z "$ckpt" ] && [ -n "$log_path" ] && [ -f "$log_path" ]; then
        ckpt="$(
            grep -E 'latest_ckpt=|last_model_checkpoint: ' "$log_path" 2>/dev/null \
                | sed -E 's/^.*latest_ckpt=//; s/^.*last_model_checkpoint: //' \
                | grep -E '/checkpoint-[0-9]+$' \
                | tail -1 || true
        )"
    fi
    [ -n "$ckpt" ] || return 1
    printf '%s\n' "$ckpt"
}

wait_for_summary() {
    local summary_path="$1"
    local status_path="${2:-}"
    local max_polls="${3:-240}"
    local interval="${4:-30}"
    local i
    for i in $(seq 1 "$max_polls"); do
        if [ -f "$summary_path" ]; then
            return 0
        fi
        if [ -n "$status_path" ] && [ -f "$status_path" ] && grep -q '"status": "failed"' "$status_path"; then
            echo "ERROR: eval failed; status_path=${status_path}" >&2
            cat "$status_path" >&2 || true
            return 1
        fi
        sleep "$interval"
    done
    return 1
}

{
    echo "[rsvg-pipeline] timestamp=${timestamp}"
    echo "[rsvg-pipeline] base_model=${base_model}"
    echo "[rsvg-pipeline] train_jsonl=${train_jsonl}"
    echo "[rsvg-pipeline] val_jsonl=${val_jsonl}"
    echo "[rsvg-pipeline] test_jsonl=${test_jsonl}"
    echo "[rsvg-pipeline] sft_run_name=${sft_run_name}"
    echo "[rsvg-pipeline] mrpd_run_name=${mrpd_run_name}"
    echo "[rsvg-pipeline] sft_log=${sft_log}"
    echo "[rsvg-pipeline] mrpd_log=${mrpd_log}"

    if [ ! -f "$train_jsonl" ] || [ ! -f "$test_jsonl" ]; then
        echo "ERROR: RSVG jsonl files are missing." >&2
        exit 1
    fi

    echo "[rsvg-pipeline] stage1=SFT submit"
    RUN_TIMESTAMP="$timestamp" \
    EXPERIMENT_SLUG="rsvg-dior-sft" \
    RUN_NAME="$sft_run_name" \
    RJOB_NAME="$sft_rjob_name" \
    RJOB_LOG_PATH="$sft_log" \
    RJOB_GROUP="${RJOB_GROUP:-aos}" \
    RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-aos}" \
    RJOB_CPU="${SFT_RJOB_CPU:-28}" \
    RJOB_GPU="${SFT_RJOB_GPU:-8}" \
    RJOB_MEMORY="${SFT_RJOB_MEMORY:-600000}" \
    RJOB_REPLICA="${SFT_RJOB_REPLICA:-2}" \
    RJOB_MAX_WAIT_DURATION="${SFT_RJOB_MAX_WAIT_DURATION:-12h0m0s}" \
    BASE_MODEL_PATH="$base_model" \
    MODEL_PATH="$base_model" \
    CKPT_ROOT="$sft_ckpt_root" \
    ARTIFACT_ROOT="$sft_artifact_root" \
    TRAIN_JSONL="$train_jsonl" \
    NNODES=2 \
    NPROC_PER_NODE=8 \
    TRAIN_CUDA_VISIBLE_DEVICES="${SFT_TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
    MASTER_PORT="${SFT_MASTER_PORT:-29810}" \
    NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-1}" \
    MAX_STEPS="${SFT_MAX_STEPS:--1}" \
    SAVE_STEPS="${SFT_SAVE_STEPS:-50}" \
    SAVE_TOTAL_LIMIT="${SFT_SAVE_TOTAL_LIMIT:-8}" \
    SAVE_ONLY_MODEL=true \
    TUNER_TYPE=full \
    ALLOW_RESUME="${SFT_ALLOW_RESUME:-false}" \
    AUTO_RESUME="${SFT_AUTO_RESUME:-false}" \
    PER_DEVICE_TRAIN_BATCH_SIZE="${SFT_PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
    GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-4}" \
    LR="${SFT_LR:-1e-5}" \
    WARMUP_RATIO="${SFT_WARMUP_RATIO:-0.03}" \
    MAX_LENGTH="${SFT_MAX_LENGTH:-20000}" \
    DEEPSPEED_CONFIG="${SFT_DEEPSPEED_CONFIG:-zero2}" \
    DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}" \
    DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}" \
    bash "${repo_root}/tools/training/sft/launch_qwen3_4b_rrsisd_sft_2node_aos.sh"

    sft_ckpt="$(latest_checkpoint "$sft_ckpt_root" "$sft_run_name" "$sft_log")" || {
        echo "ERROR: could not locate SFT checkpoint for ${sft_run_name}" >&2
        exit 1
    }
    echo "[rsvg-pipeline] sft_ckpt=${sft_ckpt}"

    echo "[rsvg-pipeline] stage2=MRPD submit"
    RUN_TIMESTAMP="$timestamp" \
    EXPERIMENT_SLUG="rsvg-dior-sft-mrpd-metric-gaussian" \
    RUN_NAME="$mrpd_run_name" \
    RJOB_NAME="$mrpd_rjob_name" \
    RJOB_LOG_PATH="$mrpd_log" \
    WORKER_SCRIPT="${repo_root}/tools/training/run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh" \
    RJOB_GROUP="${RJOB_GROUP:-aos}" \
    RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-aos}" \
    RJOB_CPU="${MRPD_RJOB_CPU:-28}" \
    RJOB_GPU="${MRPD_RJOB_GPU:-8}" \
    RJOB_MEMORY="${MRPD_RJOB_MEMORY:-600000}" \
    RJOB_REPLICA="${MRPD_RJOB_REPLICA:-2}" \
    RJOB_MAX_WAIT_DURATION="${MRPD_RJOB_MAX_WAIT_DURATION:-12h0m0s}" \
    COMPLETION_GREP_PATTERN="\\[4b-2node-worker\\].*completed" \
    BASE_MODEL_PATH="$sft_ckpt" \
    MODEL_PATH="$sft_ckpt" \
    TEACHER_PATH="$sft_ckpt" \
    ROLLOUT_MODEL_PATH="$sft_ckpt" \
    CKPT_ROOT="$mrpd_ckpt_root" \
    ARTIFACT_ROOT="$mrpd_artifact_root" \
    TRAIN_JSONL="$train_jsonl" \
    TEST_JSONL="$test_jsonl" \
    NNODES=2 \
    NPROC_PER_NODE=7 \
    TRAIN_CUDA_VISIBLE_DEVICES="${MRPD_TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}" \
    ROLLOUT_CUDA_VISIBLE_DEVICES="${MRPD_ROLLOUT_CUDA_VISIBLE_DEVICES:-7}" \
    MASTER_PORT="${MRPD_MASTER_PORT:-29820}" \
    VLLM_SERVER_PORT="${MRPD_VLLM_SERVER_PORT:-8622}" \
    NUM_TRAIN_EPOCHS="${MRPD_NUM_TRAIN_EPOCHS:-1}" \
    MAX_STEPS="${MRPD_MAX_STEPS:--1}" \
    SAVE_STEPS="${MRPD_SAVE_STEPS:-10}" \
    SAVE_TOTAL_LIMIT="${MRPD_SAVE_TOTAL_LIMIT:-20}" \
    SAVE_ONLY_MODEL=true \
    TUNER_TYPE=full \
    ALLOW_RESUME="${MRPD_ALLOW_RESUME:-false}" \
    AUTO_RESUME="${MRPD_AUTO_RESUME:-false}" \
    SKIP_EVAL=true \
    STAGE2_SKIP_EVAL=true \
    PER_DEVICE_TRAIN_BATCH_SIZE="${MRPD_PER_DEVICE_TRAIN_BATCH_SIZE:-4}" \
    GRADIENT_ACCUMULATION_STEPS="${MRPD_GRADIENT_ACCUMULATION_STEPS:-2}" \
    LR="${MRPD_LR:-1e-6}" \
    WARMUP_RATIO="${MRPD_WARMUP_RATIO:-0.01}" \
    NUM_GENERATIONS="${MRPD_NUM_GENERATIONS:-8}" \
    NUM_ITERATIONS="${MRPD_NUM_ITERATIONS:-1}" \
    MAX_LENGTH="${MRPD_MAX_LENGTH:-20000}" \
    MAX_COMPLETION_LENGTH="${MRPD_MAX_COMPLETION_LENGTH:-64}" \
    DEEPSPEED_CONFIG="${MRPD_DEEPSPEED_CONFIG:-zero2}" \
    TEACHER_DEEPSPEED_CONFIG="${MRPD_TEACHER_DEEPSPEED_CONFIG:-zero3}" \
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

    mrpd_ckpt="$(latest_checkpoint "$mrpd_ckpt_root" "$mrpd_run_name" "$mrpd_log")" || {
        echo "ERROR: could not locate MRPD checkpoint for ${mrpd_run_name}" >&2
        exit 1
    }
    echo "[rsvg-pipeline] mrpd_ckpt=${mrpd_ckpt}"

    eval_slug="q4b-rsvg-sft-mrpd-${timestamp}-test"
    eval_out="${eval_log_root}/${eval_slug}"
    eval_rjob="${EVAL_RJOB_NAME:-eval-rsvg-mrpd-$(date +%m%d%H%M)}"
    echo "[rsvg-pipeline] stage3=test eval submit eval_rjob=${eval_rjob} eval_out=${eval_out}"
    MODEL="$mrpd_ckpt" \
    RJOB_NAME="$eval_rjob" \
    RJOB_GROUP="${EVAL_RJOB_GROUP:-aos}" \
    RJOB_CHARGED_GROUP="${EVAL_RJOB_CHARGED_GROUP:-aos}" \
    RJOB_CPU="${EVAL_RJOB_CPU:-28}" \
    RJOB_GPU="${EVAL_RJOB_GPU:-8}" \
    RJOB_MEMORY="${EVAL_RJOB_MEMORY:-600000}" \
    RJOB_MAX_WAIT_DURATION="${EVAL_RJOB_MAX_WAIT_DURATION:-3h0m0s}" \
    EVAL_JSONL="$test_jsonl" \
    OUT_DIR="$eval_out" \
    LOG_PATH="${eval_log_root}/${eval_rjob}.submit.log" \
    WORKER_LOG="${eval_log_root}/${eval_rjob}.worker.log" \
    TP="${EVAL_TP:-8}" \
    MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-12000}" \
    GPU_MEM_UTIL="${EVAL_GPU_MEM_UTIL:-0.82}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}" \
    MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}" \
    NUM_ROLLOUTS=1 \
    ROLLOUT_TEMPERATURE=0.0 \
    TOP_P="${EVAL_TOP_P:-0.95}" \
    bash "${repo_root}/tools/evaluation/launch_single_ckpt_split_eval_detached_rjob.sh"

    summary="${eval_out}/summary.json"
    echo "[rsvg-pipeline] waiting for eval summary=${summary}"
    if wait_for_summary "$summary" "${eval_out}/status.json" "${EVAL_WAIT_POLLS:-360}" "${EVAL_WAIT_INTERVAL_SEC:-30}"; then
        echo "[rsvg-pipeline] eval summary ready"
        cat "$summary"
        echo
    else
        echo "ERROR: eval summary not ready within wait budget: ${summary}" >&2
        exit 1
    fi

    cat <<EOF
[rsvg-pipeline] completed
[rsvg-pipeline] sft_ckpt=${sft_ckpt}
[rsvg-pipeline] mrpd_ckpt=${mrpd_ckpt}
[rsvg-pipeline] test_summary=${summary}
EOF
} 2>&1 | tee "$pipeline_log"
