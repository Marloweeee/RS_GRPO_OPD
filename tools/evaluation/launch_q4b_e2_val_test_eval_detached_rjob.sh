#!/usr/bin/env bash
# Detached val+test evaluation for the best-1epoch continued-to-2epoch 4B checkpoint.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
rjob_name="${RJOB_NAME:-q4b-e2-val-test-eval-${timestamp}}"
group="${RJOB_GROUP:-aos}"
charged_group="${RJOB_CHARGED_GROUP:-$group}"
cpu="${RJOB_CPU:-64}"
gpu="${RJOB_GPU:-8}"
memory="${RJOB_MEMORY:-600000}"
max_wait_duration="${RJOB_MAX_WAIT_DURATION:-30m0s}"
positive_tags="${RJOB_POSITIVE_TAGS:-feature/gpfs=yes}"
mount_arg="--mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot"

model="${MODEL:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rsfull_grpo_opsd_best1e_continue_e2_metric_gaussian_20260526-092528/v0-20260526-093208/checkpoint-870}"
val_jsonl="${VAL_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_val.jsonl}"
test_jsonl="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
val_out_dir="${VAL_OUT_DIR:-/data/codes/gui_grounding/data/logs/eval/qwen3-4b-best1e-e2-metric-gaussian-val-ck870}"
test_out_dir="${TEST_OUT_DIR:-/data/codes/gui_grounding/data/logs/eval/qwen3-4b-best1e-e2-metric-gaussian-test-ck870}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs/eval}"
submit_log="${LOG_PATH:-${log_root}/${rjob_name}.submit.log}"
worker_log="${WORKER_LOG:-${log_root}/${rjob_name}.worker.log}"
worker_script="${WORKER_SCRIPT:-${repo_root}/tools/evaluation/run_two_split_eval_worker.sh}"

mkdir -p "$log_root" "$val_out_dir" "$test_out_dir"
chmod +x "$worker_script" "${repo_root}/tools/evaluation/run_single_ckpt_eval_worker.sh"

{
    echo "[q4b-e2-val-test-detached] timestamp=${timestamp}"
    echo "[q4b-e2-val-test-detached] rjob_name=${rjob_name}"
    echo "[q4b-e2-val-test-detached] group=${group} charged_group=${charged_group}"
    echo "[q4b-e2-val-test-detached] cpu=${cpu} gpu=${gpu} memory=${memory} positive_tags=${positive_tags}"
    echo "[q4b-e2-val-test-detached] model=${model}"
    echo "[q4b-e2-val-test-detached] val_jsonl=${val_jsonl}"
    echo "[q4b-e2-val-test-detached] test_jsonl=${test_jsonl}"
    echo "[q4b-e2-val-test-detached] val_out_dir=${val_out_dir}"
    echo "[q4b-e2-val-test-detached] test_out_dir=${test_out_dir}"
    echo "[q4b-e2-val-test-detached] submit_log=${submit_log}"
    echo "[q4b-e2-val-test-detached] worker_log=${worker_log}"

    brainctl rjob launch \
        --cpu "$cpu" \
        --gpu "$gpu" \
        --memory "$memory" \
        --group "$group" \
        --charged-group="$charged_group" \
        --private-machine=group \
        "$mount_arg" \
        --positive-tags "$positive_tags" \
        --predict-only

    brainctl launch \
        -d \
        --name "$rjob_name" \
        --replica-restart=never \
        --backoff-limit 1 \
        --max-wait-duration "$max_wait_duration" \
        --cpu "$cpu" \
        --gpu "$gpu" \
        --memory="$memory" \
        --group "$group" \
        --charged-group="$charged_group" \
        --private-machine=group \
        "$mount_arg" \
        --positive-tags "$positive_tags" \
        --set-env "REPO_ROOT=$repo_root" \
        --set-env "GUI_SD_ENV_ROOT=${GUI_SD_ENV_ROOT:-/data/codes/gui_grounding/conda_envs/GUI-SD}" \
        --set-env "MODEL=$model" \
        --set-env "VAL_JSONL=$val_jsonl" \
        --set-env "TEST_JSONL=$test_jsonl" \
        --set-env "VAL_OUT_DIR=$val_out_dir" \
        --set-env "TEST_OUT_DIR=$test_out_dir" \
        --set-env "EVAL_CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
        --set-env "TP=${TP:-8}" \
        --set-env "MAX_MODEL_LEN=${MAX_MODEL_LEN:-12000}" \
        --set-env "GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.88}" \
        --set-env "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-512}" \
        --set-env "MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}" \
        --set-env "NUM_ROLLOUTS=${NUM_ROLLOUTS:-1}" \
        --set-env "ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.0}" \
        --set-env "TOP_P=${TOP_P:-0.95}" \
        --set-env "WORKER_LOG=$worker_log" \
        -- \
        bash -lc '
            set -euo pipefail
            mkdir -p "$(dirname "$WORKER_LOG")"
            {
                echo "[detached-worker] start=$(date "+%F %T %Z")"
                echo "[detached-worker] host=$(hostname)"
                bash "$REPO_ROOT/tools/evaluation/run_two_split_eval_worker.sh"
                status=$?
                echo "[detached-worker] status=${status}"
                echo "[detached-worker] end=$(date "+%F %T %Z")"
                exit "$status"
            } > "$WORKER_LOG" 2>&1
        '
} 2>&1 | tee "$submit_log"

echo "[q4b-e2-val-test-detached] submitted"
echo "[q4b-e2-val-test-detached] worker_log=${worker_log}"
echo "[q4b-e2-val-test-detached] val_summary=${val_out_dir}/threshold_summary.json"
echo "[q4b-e2-val-test-detached] test_summary=${test_out_dir}/threshold_summary.json"
