#!/usr/bin/env bash
# Detached teacher-hint pass@K evaluation launcher.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
rjob_name="${RJOB_NAME:-hint-pass8-eval-${timestamp}}"
group="${RJOB_GROUP:-aos}"
charged_group="${RJOB_CHARGED_GROUP:-$group}"
cpu="${RJOB_CPU:-28}"
gpu="${RJOB_GPU:-8}"
memory="${RJOB_MEMORY:-600000}"
max_wait_duration="${RJOB_MAX_WAIT_DURATION:-1h0m0s}"
positive_tags="${RJOB_POSITIVE_TAGS:-feature/gpfs=yes}"
mount_arg="--mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot"

model="${MODEL:?MODEL is required}"
eval_jsonl="${EVAL_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
out_dir="${OUT_DIR:?OUT_DIR is required}"
hint_mode="${HINT_MODE:-gaussian}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs/eval}"
submit_log="${LOG_PATH:-${log_root}/${rjob_name}.submit.log}"
worker_log="${WORKER_LOG:-${log_root}/${rjob_name}.worker.log}"
worker_script="${WORKER_SCRIPT:-${repo_root}/tools/evaluation/run_teacher_hint_pass8_worker.sh}"

mkdir -p "$log_root" "$out_dir"
chmod +x "$worker_script"

{
    echo "[hint-pass8-launch] timestamp=${timestamp}"
    echo "[hint-pass8-launch] rjob_name=${rjob_name}"
    echo "[hint-pass8-launch] group=${group} charged_group=${charged_group}"
    echo "[hint-pass8-launch] cpu=${cpu} gpu=${gpu} memory=${memory} positive_tags=${positive_tags}"
    echo "[hint-pass8-launch] model=${model}"
    echo "[hint-pass8-launch] eval_jsonl=${eval_jsonl}"
    echo "[hint-pass8-launch] out_dir=${out_dir}"
    echo "[hint-pass8-launch] hint_mode=${hint_mode}"
    echo "[hint-pass8-launch] submit_log=${submit_log}"
    echo "[hint-pass8-launch] worker_log=${worker_log}"

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
        --set-env "EVAL_JSONL=$eval_jsonl" \
        --set-env "OUT_DIR=$out_dir" \
        --set-env "HINT_MODE=$hint_mode" \
        --set-env "EVAL_CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
        --set-env "TP=${TP:-8}" \
        --set-env "MAX_MODEL_LEN=${MAX_MODEL_LEN:-12000}" \
        --set-env "GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.88}" \
        --set-env "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}" \
        --set-env "MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}" \
        --set-env "NUM_ROLLOUTS=${NUM_ROLLOUTS:-8}" \
        --set-env "ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}" \
        --set-env "TOP_P=${TOP_P:-0.95}" \
        --set-env "OPSD_GAUSSIAN_SIGMA_RATIO=${OPSD_GAUSSIAN_SIGMA_RATIO:-1.5}" \
        --set-env "OPSD_MIN_AREA_FRAC=${OPSD_MIN_AREA_FRAC:-0.1}" \
        --set-env "OPSD_ZOOM_RATIO=${OPSD_ZOOM_RATIO:-2.0}" \
        --set-env "OPSD_JITTER_RATIO=${OPSD_JITTER_RATIO:-0.2}" \
        --set-env "OPSD_HINT_BOX_COLOR=${OPSD_HINT_BOX_COLOR:-magenta}" \
        --set-env "WORKER_LOG=$worker_log" \
        -- \
        bash -lc '
            set -euo pipefail
            mkdir -p "$(dirname "$WORKER_LOG")"
            {
                echo "[detached-hint-pass8-worker] start=$(date "+%F %T %Z")"
                echo "[detached-hint-pass8-worker] host=$(hostname)"
                bash "$REPO_ROOT/tools/evaluation/run_teacher_hint_pass8_worker.sh"
                status=$?
                echo "[detached-hint-pass8-worker] status=${status}"
                echo "[detached-hint-pass8-worker] end=$(date "+%F %T %Z")"
                exit "$status"
            } > "$WORKER_LOG" 2>&1
        '
} 2>&1 | tee "$submit_log"

echo "[hint-pass8-launch] submitted"
echo "[hint-pass8-launch] worker_log=${worker_log}"
echo "[hint-pass8-launch] summary=${out_dir}/summary.json"
