#!/usr/bin/env bash
# Evaluate the resumed gaussian no-refresh baseline checkpoint on rs_test.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
rjob_name="${RJOB_NAME:-eval-tr-gauss-resume870-final-${timestamp}}"
group="${RJOB_GROUP:-aos}"
charged_group="${RJOB_CHARGED_GROUP:-$group}"
node_tag="${RJOB_NODE_TAG:-node/gpu-a800-0031.host.platform.shaipower.com}"
cpu="${RJOB_CPU:-8}"
gpu="${RJOB_GPU:-1}"
memory="${RJOB_MEMORY:-80000}"
max_wait_duration="${RJOB_MAX_WAIT_DURATION:-20m0s}"
timeout_duration="${RJOB_TIMEOUT:-75m}"
mount_arg="--mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot"

model="${MODEL:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rsfull_grpo_sdpo_hint_gaussian_resume650_e1-20260524-142613/v0-20260524-143052/checkpoint-870}"
test_jsonl="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
eval_name="${EVAL_NAME:-tr_gauss_baseline_resume870}"
out_root="${OUT_ROOT:-/data/codes/gui_grounding/data/logs/eval/tr_latest_eval_${timestamp}}"
out_dir="${OUT_DIR:-${out_root}/${eval_name}}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs/eval}"
log_path="${LOG_PATH:-${log_root}/${rjob_name}.rjob.log}"
worker_script="${WORKER_SCRIPT:-${repo_root}/tools/evaluation/run_teacher_refresh_eval_worker.sh}"

mkdir -p "$out_dir" "$log_root"
chmod +x "$worker_script"

echo "[tr-gauss-eval] timestamp=${timestamp}"
echo "[tr-gauss-eval] rjob_name=${rjob_name}"
echo "[tr-gauss-eval] group=${group} charged_group=${charged_group}"
echo "[tr-gauss-eval] node_tag=${node_tag} cpu=${cpu} gpu=${gpu} memory=${memory}"
echo "[tr-gauss-eval] model=${model}"
echo "[tr-gauss-eval] test_jsonl=${test_jsonl}"
echo "[tr-gauss-eval] out_dir=${out_dir}"
echo "[tr-gauss-eval] log_path=${log_path}"

brainctl rjob launch \
    --cpu "$cpu" \
    --gpu "$gpu" \
    --memory "$memory" \
    --group "$group" \
    --charged-group="$charged_group" \
    --private-machine=group \
    "$mount_arg" \
    --positive-tags "$node_tag" \
    --predict-only

stop_rjob_quietly() {
    brainctl -n shai-core stop "rjob/${rjob_name}" >/dev/null 2>&1 || true
}

set +e
{
    timeout --kill-after=2m "$timeout_duration" \
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
            --set-env "EVAL_NAME=$eval_name" \
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
status=${PIPESTATUS[0]}
set -e

if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
    echo "[tr-gauss-eval] timed out, stopping ${rjob_name}"
    stop_rjob_quietly
elif [ "$status" -ne 0 ]; then
    echo "[tr-gauss-eval] failed with status=${status}, stopping ${rjob_name} if needed"
    stop_rjob_quietly
fi

summary_path="${out_dir}/threshold_summary.json"
if [ -f "$summary_path" ]; then
    python3 - "$summary_path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as f:
    metrics = json.load(f)
keys = ("mIoU", "IoU@0.5", "IoU@0.6", "IoU@0.7", "IoU@0.8", "IoU@0.9", "parse_rate", "valid_rate")
print("[tr-gauss-eval] summary=" + path)
print("[tr-gauss-eval] " + " ".join(f"{key}={metrics[key]:.4f}" for key in keys if key in metrics))
PY
else
    echo "[tr-gauss-eval] missing summary: ${summary_path}"
fi

exit "$status"
