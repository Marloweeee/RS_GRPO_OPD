#!/usr/bin/env bash
# Detached one-checkpoint eval for the resumed gaussian no-refresh baseline.
# The worker writes logs and summaries directly to the shared workspace.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
rjob_name="${RJOB_NAME:-eval-tr-gauss-resume870-detached-${timestamp}}"
group="${RJOB_GROUP:-aos}"
charged_group="${RJOB_CHARGED_GROUP:-$group}"
node_tag="${RJOB_NODE_TAG:-node/gpu-a800-0031.host.platform.shaipower.com}"
cpu="${RJOB_CPU:-8}"
gpu="${RJOB_GPU:-1}"
memory="${RJOB_MEMORY:-80000}"
max_wait_duration="${RJOB_MAX_WAIT_DURATION:-20m0s}"
mount_arg="--mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot"

model="${MODEL:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rsfull_grpo_sdpo_hint_gaussian_resume650_e1-20260524-142613/v0-20260524-143052/checkpoint-870}"
test_jsonl="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
eval_name="${EVAL_NAME:-tr_gauss_baseline_resume870}"
out_root="${OUT_ROOT:-/data/codes/gui_grounding/data/logs/eval/tr_latest_eval_${timestamp}}"
out_dir="${OUT_DIR:-${out_root}/${eval_name}}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs/eval}"
submit_log="${LOG_PATH:-${log_root}/${rjob_name}.submit.log}"
worker_log="${WORKER_LOG:-${out_dir}/worker.log}"
worker_script="${WORKER_SCRIPT:-${repo_root}/tools/evaluation/run_teacher_refresh_eval_worker.sh}"

mkdir -p "$out_dir" "$log_root"
chmod +x "$worker_script"

{
    echo "[tr-gauss-detached] timestamp=${timestamp}"
    echo "[tr-gauss-detached] rjob_name=${rjob_name}"
    echo "[tr-gauss-detached] group=${group} charged_group=${charged_group}"
    echo "[tr-gauss-detached] node_tag=${node_tag} cpu=${cpu} gpu=${gpu} memory=${memory}"
    echo "[tr-gauss-detached] model=${model}"
    echo "[tr-gauss-detached] test_jsonl=${test_jsonl}"
    echo "[tr-gauss-detached] out_dir=${out_dir}"
    echo "[tr-gauss-detached] submit_log=${submit_log}"
    echo "[tr-gauss-detached] worker_log=${worker_log}"

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
        --set-env "WORKER_LOG=$worker_log" \
        -- \
        bash -lc '
            set -euo pipefail
            mkdir -p "$OUT_DIR" "$(dirname "$WORKER_LOG")"
            {
                echo "[detached-worker] start=$(date "+%F %T %Z")"
                echo "[detached-worker] host=$(hostname)"
                set +e
                bash "$REPO_ROOT/tools/evaluation/run_teacher_refresh_eval_worker.sh"
                status=$?
                set -e
                echo "[detached-worker] status=${status}"
                echo "[detached-worker] end=$(date "+%F %T %Z")"
                exit "$status"
            } > "$WORKER_LOG" 2>&1
        '
} 2>&1 | tee "$submit_log"

echo "[tr-gauss-detached] submitted. Poll:"
echo "  brainctl get rjob ${rjob_name} -n shai-core"
echo "  tail -n 120 ${worker_log}"
echo "  cat ${out_dir}/threshold_summary.json"
