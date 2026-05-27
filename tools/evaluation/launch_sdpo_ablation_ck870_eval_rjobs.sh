#!/usr/bin/env bash
# Launch two parallel GPU rjobs to evaluate SDPO ablation checkpoint-870 runs.
set -euo pipefail

repo_root="${REPO_ROOT:-/data/codes/gui_grounding/GUI-SD-code-main}"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
group="${RJOB_GROUP:-aos}"
charged_group="${RJOB_CHARGED_GROUP:-aos}"
cpu="${RJOB_CPU:-24}"
gpu="${RJOB_GPU:-1}"
memory="${RJOB_MEMORY:-180000}"
max_wait_duration="${RJOB_MAX_WAIT_DURATION:-10m0s}"
mount_arg="--mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot"
worker_script="${WORKER_SCRIPT:-${repo_root}/tools/evaluation/run_sdpo_ablation_ck870_eval_worker.sh}"

jitter_node_tag="${JITTER_NODE_TAG:-node/gpu-a800-0023.host.platform.shaipower.com}"
zoom_node_tag="${ZOOM_NODE_TAG:-node/gpu-a800-0024.host.platform.shaipower.com}"

jitter_model="${JITTER_MODEL:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rsfull_grpo_sdpo_hint_jitter_box_e1-r3-05222116/v0-20260522-212941/checkpoint-870}"
zoom_model="${ZOOM_MODEL:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rsfull_grpo_sdpo_hint_zoom_in_e1-r3-05222116/v0-20260522-212947/checkpoint-870}"
test_jsonl="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
out_root="${OUT_ROOT:-/data/codes/gui_grounding/data/logs/sdpo_ablation_ck870_eval/${timestamp}}"

mkdir -p "$out_root" /data/codes/gui_grounding/data/logs
chmod +x "$worker_script"

predict_one() {
    local name="$1"
    local node_tag="$2"
    echo "[sdpo-ck870-launch] predict ${name}: node_tag=${node_tag}"
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

launch_one() {
    local name="$1"
    local model="$2"
    local node_tag="$3"
    local rjob_name="$4"
    local out_dir="${out_root}/${name}"
    local log_path="/data/codes/gui_grounding/data/logs/${rjob_name}.rjob.log"

    mkdir -p "$out_dir"
    echo "[sdpo-ck870-launch] launch ${name}: rjob=${rjob_name} node_tag=${node_tag}"
    echo "[sdpo-ck870-launch] ${name} model=${model}"
    echo "[sdpo-ck870-launch] ${name} out_dir=${out_dir}"
    echo "[sdpo-ck870-launch] ${name} log_path=${log_path}"

    {
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
            --set-env "NUM_ROLLOUTS=${NUM_ROLLOUTS:-1}" \
            --set-env "ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.0}" \
            -- \
            bash "$worker_script"
    } 2>&1 | tee "$log_path"
}

echo "[sdpo-ck870-launch] timestamp=${timestamp}"
echo "[sdpo-ck870-launch] group=${group} charged_group=${charged_group}"
echo "[sdpo-ck870-launch] cpu=${cpu} gpu=${gpu} memory=${memory}"
echo "[sdpo-ck870-launch] test_jsonl=${test_jsonl}"
echo "[sdpo-ck870-launch] out_root=${out_root}"

predict_one jitter_box "$jitter_node_tag"
predict_one zoom_in "$zoom_node_tag"

jitter_rjob="${JITTER_RJOB_NAME:-eval-jitter-ck870-${timestamp}}"
zoom_rjob="${ZOOM_RJOB_NAME:-eval-zoom-ck870-${timestamp}}"

launch_one jitter_box "$jitter_model" "$jitter_node_tag" "$jitter_rjob" &
jitter_pid=$!
launch_one zoom_in "$zoom_model" "$zoom_node_tag" "$zoom_rjob" &
zoom_pid=$!

set +e
wait "$jitter_pid"
jitter_status=$?
wait "$zoom_pid"
zoom_status=$?
set -e

echo "[sdpo-ck870-launch] jitter_status=${jitter_status}"
echo "[sdpo-ck870-launch] zoom_status=${zoom_status}"

python3 - "$out_root" <<'PY'
import json
import os
import sys

out_root = sys.argv[1]
rows = []
for name in ("jitter_box", "zoom_in"):
    summary_path = os.path.join(out_root, name, "summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            m = json.load(f)
        rows.append((name, summary_path, m))
    else:
        rows.append((name, summary_path, None))

print("\n=== SDPO ablation checkpoint-870 eval summary ===")
for name, summary_path, m in rows:
    if m is None:
        print(f"{name}: missing summary at {summary_path}")
        continue
    print(
        f"{name}: IoU@0.5={m['IoU@0.5']:.4f}, "
        f"mIoU={m['mIoU']:.4f}, IoU@0.7={m['IoU@0.7']:.4f}, "
        f"parse_rate={m['parse_rate']:.4f}, valid_rate={m['valid_rate']:.4f}, "
        f"summary={summary_path}"
    )
PY

if [ "$jitter_status" -ne 0 ] || [ "$zoom_status" -ne 0 ]; then
    exit 1
fi
