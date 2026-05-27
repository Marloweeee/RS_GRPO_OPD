#!/usr/bin/env bash
# Detached launcher for one checkpoint on one jsonl split.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
rjob_name="${RJOB_NAME:-single-split-eval-${timestamp}}"
group="${RJOB_GROUP:-aos}"
charged_group="${RJOB_CHARGED_GROUP:-$group}"
cpu="${RJOB_CPU:-28}"
gpu="${RJOB_GPU:-8}"
memory="${RJOB_MEMORY:-600000}"
max_wait_duration="${RJOB_MAX_WAIT_DURATION:-1h0m0s}"
positive_tags="${RJOB_POSITIVE_TAGS:-feature/gpfs=yes}"
mount_arg="--mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot"

model="${MODEL:?MODEL is required}"
eval_jsonl="${EVAL_JSONL:?EVAL_JSONL is required}"
out_dir="${OUT_DIR:?OUT_DIR is required}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs/eval}"
submit_log="${LOG_PATH:-${log_root}/${rjob_name}.submit.log}"
worker_log="${WORKER_LOG:-${log_root}/${rjob_name}.worker.log}"
worker_script="${WORKER_SCRIPT:-${repo_root}/tools/evaluation/run_single_ckpt_eval_worker.sh}"

mkdir -p "$log_root" "$out_dir"
chmod +x "$worker_script"

{
    echo "[single-split-detached] timestamp=${timestamp}"
    echo "[single-split-detached] rjob_name=${rjob_name}"
    echo "[single-split-detached] group=${group} charged_group=${charged_group}"
    echo "[single-split-detached] cpu=${cpu} gpu=${gpu} memory=${memory} positive_tags=${positive_tags}"
    echo "[single-split-detached] model=${model}"
    echo "[single-split-detached] eval_jsonl=${eval_jsonl}"
    echo "[single-split-detached] out_dir=${out_dir}"
    echo "[single-split-detached] submit_log=${submit_log}"
    echo "[single-split-detached] worker_log=${worker_log}"

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
                echo "[detached-single-split-worker] start=$(date "+%F %T %Z")"
                echo "[detached-single-split-worker] host=$(hostname)"
                bash "$REPO_ROOT/tools/evaluation/run_single_ckpt_eval_worker.sh"
                status=$?
                echo "[detached-single-split-worker] status=${status}"
                echo "[detached-single-split-worker] end=$(date "+%F %T %Z")"
                exit "$status"
            } > "$WORKER_LOG" 2>&1
        '
} 2>&1 | tee "$submit_log"

echo "[single-split-detached] submitted"
echo "[single-split-detached] worker_log=${worker_log}"
echo "[single-split-detached] summary=${out_dir}/summary.json"
