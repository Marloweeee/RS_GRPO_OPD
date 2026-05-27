#!/usr/bin/env bash
# Launch full-test evaluation for the four teacher-refresh checkpoints.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
group="${RJOB_GROUP:-aos}"
charged_group="${RJOB_CHARGED_GROUP:-$group}"
cpu="${RJOB_CPU:-16}"
gpu="${RJOB_GPU:-1}"
memory="${RJOB_MEMORY:-160000}"
max_wait_duration="${RJOB_MAX_WAIT_DURATION:-10m0s}"
per_job_timeout="${PER_JOB_TIMEOUT:-45m}"
mount_arg="--mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot"
worker_script="${WORKER_SCRIPT:-${repo_root}/tools/evaluation/run_teacher_refresh_eval_worker.sh}"

test_jsonl="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
out_root="${OUT_ROOT:-/data/codes/gui_grounding/data/logs/eval/teacher_refresh_eval_${timestamp}}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs/eval}"

node_tag_0="${NODE_TAG_0:-node/gpu-a800-0026.host.platform.shaipower.com}"
node_tag_1="${NODE_TAG_1:-node/gpu-a800-0022.host.platform.shaipower.com}"
node_tag_2="${NODE_TAG_2:-node/gpu-a800-0031.host.platform.shaipower.com}"
node_tag_3="${NODE_TAG_3:-node/gpu-a800-0028.host.platform.shaipower.com}"

model_s80_zoom="${MODEL_S80_ZOOM:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rsfull_grpo_sdpo_teacher_refresh_s80_zoom_in_e1-20260523-133039/v0-20260523-133631/checkpoint-870}"
model_s80_gaussian="${MODEL_S80_GAUSSIAN:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rsfull_grpo_sdpo_teacher_refresh_s80_gaussian_e1-20260523-133039/v0-20260523-133630/checkpoint-870}"
model_metric_gaussian="${MODEL_METRIC_GAUSSIAN:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rsfull_grpo_sdpo_teacher_refresh_metric_gaussian_e1-20260523-141744/v0-20260523-142213/checkpoint-80}"
model_metric_zoom="${MODEL_METRIC_ZOOM:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rsfull_grpo_sdpo_teacher_refresh_metric_zoom_in_e1-20260523-142445/v0-20260523-142900/checkpoint-80}"

mkdir -p "$out_root" "$log_root"
chmod +x "$worker_script"

predict_one() {
    local name="$1"
    local node_tag="$2"
    echo "[teacher-refresh-launch] predict ${name}: node_tag=${node_tag}"
    brainctl launch \
        --cpu "$cpu" \
        --gpu "$gpu" \
        --memory "$memory" \
        --group "$group" \
        --charged-group="$charged_group" \
        --private-machine=group \
        "$mount_arg" \
        --positive-tags "$node_tag" \
        --predict-only
}

stop_rjob_quietly() {
    local rjob_name="$1"
    brainctl -n shai-core stop "rjob/${rjob_name}" >/dev/null 2>&1 || true
}

launch_one() {
    local name="$1"
    local model="$2"
    local node_tag="$3"
    local rjob_name="$4"
    local out_dir="${out_root}/${name}"
    local log_path="${log_root}/${rjob_name}.rjob.log"

    mkdir -p "$out_dir"
    echo "[teacher-refresh-launch] launch ${name}: rjob=${rjob_name} node_tag=${node_tag}"
    echo "[teacher-refresh-launch] ${name} model=${model}"
    echo "[teacher-refresh-launch] ${name} out_dir=${out_dir}"
    echo "[teacher-refresh-launch] ${name} log_path=${log_path}"

    set +e
    {
        timeout --kill-after=2m "$per_job_timeout" \
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
                --positive-tags "$node_tag" \
                --set-env "REPO_ROOT=$repo_root" \
                --set-env "GUI_SD_ENV_ROOT=${GUI_SD_ENV_ROOT:-/data/codes/gui_grounding/conda_envs/GUI-SD}" \
                --set-env "MODEL=$model" \
                --set-env "TEST_JSONL=$test_jsonl" \
                --set-env "OUT_DIR=$out_dir" \
                --set-env "EVAL_NAME=$name" \
                --set-env "EVAL_CUDA_VISIBLE_DEVICES=0" \
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
        echo "[teacher-refresh-launch] ${name} timed out, stopping ${rjob_name}"
        stop_rjob_quietly "$rjob_name"
    elif [ "$status" -ne 0 ]; then
        echo "[teacher-refresh-launch] ${name} failed with status=${status}, stopping ${rjob_name} if needed"
        stop_rjob_quietly "$rjob_name"
    fi
    return "$status"
}

echo "[teacher-refresh-launch] timestamp=${timestamp}"
echo "[teacher-refresh-launch] group=${group} charged_group=${charged_group}"
echo "[teacher-refresh-launch] cpu=${cpu} gpu=${gpu} memory=${memory}"
echo "[teacher-refresh-launch] per_job_timeout=${per_job_timeout}"
echo "[teacher-refresh-launch] test_jsonl=${test_jsonl}"
echo "[teacher-refresh-launch] out_root=${out_root}"

predict_one s80_zoom "$node_tag_0"
predict_one s80_gaussian "$node_tag_1"
predict_one metric_gaussian "$node_tag_2"
predict_one metric_zoom "$node_tag_3"

rjob_s80_zoom="${RJOB_S80_ZOOM:-eval-trs80-zoom-${timestamp}}"
rjob_s80_gaussian="${RJOB_S80_GAUSSIAN:-eval-trs80-gauss-${timestamp}}"
rjob_metric_gaussian="${RJOB_METRIC_GAUSSIAN:-eval-trmetric-gauss-${timestamp}}"
rjob_metric_zoom="${RJOB_METRIC_ZOOM:-eval-trmetric-zoom-${timestamp}}"

launch_one s80_zoom "$model_s80_zoom" "$node_tag_0" "$rjob_s80_zoom" &
pid_s80_zoom=$!
launch_one s80_gaussian "$model_s80_gaussian" "$node_tag_1" "$rjob_s80_gaussian" &
pid_s80_gaussian=$!
launch_one metric_gaussian "$model_metric_gaussian" "$node_tag_2" "$rjob_metric_gaussian" &
pid_metric_gaussian=$!
launch_one metric_zoom "$model_metric_zoom" "$node_tag_3" "$rjob_metric_zoom" &
pid_metric_zoom=$!

set +e
wait "$pid_s80_zoom"; status_s80_zoom=$?
wait "$pid_s80_gaussian"; status_s80_gaussian=$?
wait "$pid_metric_gaussian"; status_metric_gaussian=$?
wait "$pid_metric_zoom"; status_metric_zoom=$?
set -e

echo "[teacher-refresh-launch] status_s80_zoom=${status_s80_zoom}"
echo "[teacher-refresh-launch] status_s80_gaussian=${status_s80_gaussian}"
echo "[teacher-refresh-launch] status_metric_gaussian=${status_metric_gaussian}"
echo "[teacher-refresh-launch] status_metric_zoom=${status_metric_zoom}"

python3 - "$out_root" <<'PY'
import json
import os
import sys

out_root = sys.argv[1]
names = ("s80_zoom", "s80_gaussian", "metric_gaussian", "metric_zoom")
fields = ("mIoU", "IoU@0.5", "IoU@0.6", "IoU@0.7", "IoU@0.8", "IoU@0.9", "parse_rate", "valid_rate")

print("\n=== teacher-refresh eval summary ===")
for name in names:
    summary_path = os.path.join(out_root, name, "threshold_summary.json")
    if not os.path.isfile(summary_path):
        summary_path = os.path.join(out_root, name, "summary.json")
    if not os.path.isfile(summary_path):
        print(f"{name}: missing summary at {summary_path}")
        continue
    with open(summary_path) as f:
        m = json.load(f)
    parts = []
    for key in fields:
        if key in m:
            parts.append(f"{key}={m[key]:.4f}")
    print(f"{name}: " + ", ".join(parts) + f", summary={summary_path}")
PY

if [ "$status_s80_zoom" -ne 0 ] || [ "$status_s80_gaussian" -ne 0 ] || \
   [ "$status_metric_gaussian" -ne 0 ] || [ "$status_metric_zoom" -ne 0 ]; then
    exit 1
fi
