#!/usr/bin/env bash
# Launch one GPU rjob to evaluate a single checkpoint on one jsonl split.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

split="${1:?usage: $0 <split-name> <eval-jsonl> <out-dir> [node-tag]}"
eval_jsonl="${2:?usage: $0 <split-name> <eval-jsonl> <out-dir> [node-tag]}"
out_dir="${3:?usage: $0 <split-name> <eval-jsonl> <out-dir> [node-tag]}"
node_tag="${4:-${RJOB_NODE_TAG:-}}"

model="${MODEL:?MODEL is required}"
timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
rjob_name="${RJOB_NAME:-q4b-e2-${split}-eval-${timestamp}}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs/eval}"
log_path="${log_root}/${rjob_name}.rjob.log"
worker_script="${WORKER_SCRIPT:-${repo_root}/tools/evaluation/run_single_ckpt_eval_worker.sh}"

group="${RJOB_GROUP:-aos}"
charged_group="${RJOB_CHARGED_GROUP:-$group}"
cpu="${RJOB_CPU:-64}"
gpu="${RJOB_GPU:-8}"
memory="${RJOB_MEMORY:-600000}"
max_wait_duration="${RJOB_MAX_WAIT_DURATION:-30m0s}"
job_timeout="${RJOB_TIMEOUT:-4h}"
mount_arg="--mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot"

mkdir -p "$log_root" "$out_dir"

positive_tag_args=()
if [ -n "$node_tag" ]; then
    positive_tag_args+=(--positive-tags "node/${node_tag}")
else
    positive_tag_args+=(--positive-tags "feature/gpfs=yes")
fi

echo "[single-split-launch] split=${split}"
echo "[single-split-launch] model=${model}"
echo "[single-split-launch] eval_jsonl=${eval_jsonl}"
echo "[single-split-launch] out_dir=${out_dir}"
echo "[single-split-launch] rjob_name=${rjob_name}"
echo "[single-split-launch] log_path=${log_path}"
echo "[single-split-launch] group=${group} cpu=${cpu} gpu=${gpu} memory=${memory} node_tag=${node_tag:-auto}"

set +e
{
    timeout --kill-after=2m "$job_timeout" \
        brainctl launch \
            --name "$rjob_name" \
            --replica-restart=never \
            --backoff-limit 1 \
            --max-wait-duration "$max_wait_duration" \
            --cpu "$cpu" \
            --gpu "$gpu" \
            --memory "$memory" \
            --group "$group" \
            --charged-group "$charged_group" \
            --private-machine=group \
            "$mount_arg" \
            "${positive_tag_args[@]}" \
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
            -- \
            bash "$worker_script"
} 2>&1 | tee "$log_path"
status=${PIPESTATUS[0]}
set -e

if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
    echo "[single-split-launch] timed out; stopping rjob/${rjob_name}"
    brainctl -n shai-core stop "rjob/${rjob_name}" >/dev/null 2>&1 || true
elif [ "$status" -ne 0 ]; then
    echo "[single-split-launch] failed status=${status}; stopping rjob/${rjob_name} if needed"
    brainctl -n shai-core stop "rjob/${rjob_name}" >/dev/null 2>&1 || true
fi

echo "[single-split-launch] completed status=${status}"
exit "$status"
