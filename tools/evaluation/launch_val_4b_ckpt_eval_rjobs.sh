#!/usr/bin/env bash
# Launch sharded validation-set evaluation for existing 4B checkpoints.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
run_root="${OUT_ROOT:-/data/codes/gui_grounding/data/logs/eval/val_ckpt_eval_${timestamp}}"
manifest="${MANIFEST:-${run_root}/val_ckpt_eval_manifest.jsonl}"
skipped_jsonl="${SKIPPED_JSONL:-${run_root}/val_ckpt_eval_skipped.jsonl}"
eval_jsonl="${EVAL_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_val.jsonl}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs/eval}"
html_out="${HTML_OUT:-${repo_root}/OPSD_Idea/val_4b_ckpt_metrics_summary.html}"

group="${RJOB_GROUP:-gui_agent}"
charged_group="${RJOB_CHARGED_GROUP:-$group}"
cpu="${RJOB_CPU:-16}"
gpu="${RJOB_GPU:-1}"
memory="${RJOB_MEMORY:-180000}"
num_shards="${NUM_SHARDS:-8}"
max_wait_duration="${RJOB_MAX_WAIT_DURATION:-30m0s}"
per_shard_timeout="${PER_SHARD_TIMEOUT:-4h}"
mount_arg="--mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot"
positive_tags="${RJOB_POSITIVE_TAGS:-feature/gpfs=yes}"
worker_script="${WORKER_SCRIPT:-${repo_root}/tools/evaluation/run_val_ckpt_eval_worker.sh}"
aggregate_script="${AGGREGATE_SCRIPT:-${repo_root}/tools/evaluation/aggregate_val_ckpt_eval.py}"

mkdir -p "$run_root" "$log_root"
chmod +x "$worker_script" "$aggregate_script"

if [ ! -f "$manifest" ]; then
    echo "[val-4b-launch] ERROR: manifest not found: $manifest" >&2
    exit 1
fi
if [ ! -f "$eval_jsonl" ]; then
    echo "[val-4b-launch] ERROR: eval jsonl not found: $eval_jsonl" >&2
    exit 1
fi

cat > "${run_root}/launch_config.json" <<JSON
{
  "timestamp": "${timestamp}",
  "run_root": "${run_root}",
  "manifest": "${manifest}",
  "skipped_jsonl": "${skipped_jsonl}",
  "eval_jsonl": "${eval_jsonl}",
  "group": "${group}",
  "charged_group": "${charged_group}",
  "cpu": "${cpu}",
  "gpu": "${gpu}",
  "memory": "${memory}",
  "num_shards": "${num_shards}",
  "max_wait_duration": "${max_wait_duration}",
  "per_shard_timeout": "${per_shard_timeout}",
  "positive_tags": "${positive_tags}"
}
JSON

aggregate_now() {
    python3 "$aggregate_script" \
        --manifest "$manifest" \
        --out_root "$run_root" \
        --eval_jsonl "$eval_jsonl" \
        --skipped_jsonl "$skipped_jsonl" \
        --html_out "$html_out"
}

stop_rjob_quietly() {
    local rjob_name="$1"
    brainctl -n shai-core stop "rjob/${rjob_name}" >/dev/null 2>&1 || true
}

predict_resources() {
    echo "[val-4b-launch] predict resources: group=${group} gpu=${gpu} cpu=${cpu} memory=${memory}"
    brainctl launch \
        --cpu "$cpu" \
        --gpu "$gpu" \
        --memory "$memory" \
        --group "$group" \
        --charged-group="$charged_group" \
        --private-machine=group \
        "$mount_arg" \
        --positive-tags "$positive_tags" \
        --predict-only
}

launch_shard() {
    local shard_index="$1"
    local shard_tag
    shard_tag="$(printf 's%02d' "$shard_index")"
    local rjob_name="${RJOB_NAME_PREFIX:-val4b-ckpt-eval}-${shard_tag}-${timestamp}"
    local log_path="${log_root}/${rjob_name}.rjob.log"

    echo "[val-4b-launch] launch shard=${shard_index}/${num_shards} rjob=${rjob_name}"
    echo "[val-4b-launch] log_path=${log_path}"
    set +e
    {
        timeout --kill-after=2m "$per_shard_timeout" \
            brainctl launch \
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
                --set-env "MANIFEST=$manifest" \
                --set-env "OUT_ROOT=$run_root" \
                --set-env "EVAL_JSONL=$eval_jsonl" \
                --set-env "SHARD_INDEX=$shard_index" \
                --set-env "NUM_SHARDS=$num_shards" \
                --set-env "EVAL_CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-0}" \
                --set-env "TP=${TP:-1}" \
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
    local status=${PIPESTATUS[0]}
    set -e

    if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
        echo "[val-4b-launch] shard=${shard_index} timed out; stopping ${rjob_name}"
        stop_rjob_quietly "$rjob_name"
    elif [ "$status" -ne 0 ]; then
        echo "[val-4b-launch] shard=${shard_index} failed status=${status}; stopping ${rjob_name} if needed"
        stop_rjob_quietly "$rjob_name"
    fi
    aggregate_now || true
    return "$status"
}

echo "[val-4b-launch] timestamp=${timestamp}"
echo "[val-4b-launch] run_root=${run_root}"
echo "[val-4b-launch] manifest=$(wc -l < "$manifest") records"
echo "[val-4b-launch] eval_jsonl=$(wc -l < "$eval_jsonl") samples"
echo "[val-4b-launch] html_out=${html_out}"

aggregate_now || true
predict_resources

pids=()
for shard_index in $(seq 0 $((num_shards - 1))); do
    launch_shard "$shard_index" &
    pids+=("$!")
    sleep "${LAUNCH_STAGGER_SECONDS:-8}"
done

overall_status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        overall_status=1
    fi
done

aggregate_now
echo "[val-4b-launch] completed overall_status=${overall_status}"
exit "$overall_status"
