#!/usr/bin/env bash
# Evaluate no-hint checkpoint-870 and soft-window checkpoint-690 on the full
# remote-sensing test set, then report IoU thresholds from 0.5 to 0.9.
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
mount_arg="--mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot"

test_jsonl="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
out_root="${OUT_ROOT:-/data/codes/gui_grounding/data/logs/sdpo_threshold_eval/${timestamp}}"

no_hint_model="${NO_HINT_MODEL:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rsfull_grpo_sdpo_hint_no-hint_e1-05222159-aos/v0-20260522-220317/checkpoint-870}"
soft_window_model="${SOFT_WINDOW_MODEL:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rsfull_grpo_sdpo_hint_soft_window_e1-05222050/v0-20260522-210546/checkpoint-690}"

no_hint_node_tag="${NO_HINT_NODE_TAG:-node/gpu-a800-0022.host.platform.shaipower.com}"
soft_window_node_tag="${SOFT_WINDOW_NODE_TAG:-node/gpu-a800-0026.host.platform.shaipower.com}"

mkdir -p "$out_root" /data/codes/gui_grounding/data/logs

predict_one() {
    local name="$1"
    local node_tag="$2"
    echo "[threshold-eval-launch] predict ${name}: node_tag=${node_tag}"
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
    echo "[threshold-eval-launch] launch ${name}: rjob=${rjob_name} node_tag=${node_tag}"
    echo "[threshold-eval-launch] ${name} model=${model}"
    echo "[threshold-eval-launch] ${name} out_dir=${out_dir}"
    echo "[threshold-eval-launch] ${name} log_path=${log_path}"

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
            --set-env "MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}" \
            --set-env "NUM_ROLLOUTS=${NUM_ROLLOUTS:-1}" \
            --set-env "ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.0}" \
            --set-env "TOP_P=${TOP_P:-0.95}" \
            -- \
            bash -lc "$(cat <<'EOS'
set -euo pipefail

repo_root="${REPO_ROOT:-/data/codes/gui_grounding/GUI-SD-code-main}"
env_root="${GUI_SD_ENV_ROOT:-/data/codes/gui_grounding/conda_envs/GUI-SD}"
cd "$repo_root"

export PATH="${env_root}/bin:${PATH}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONFAULTHANDLER=1
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-10000}"
export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"

python_bin="${PYTHON_BIN:-${env_root}/bin/python}"
model="${MODEL:?MODEL must point to checkpoint dir}"
test_jsonl="${TEST_JSONL:?TEST_JSONL is required}"
out_dir="${OUT_DIR:?OUT_DIR is required}"
eval_name="${EVAL_NAME:-eval}"

cache_root="/tmp/guisd_threshold_eval_${eval_name}_$$"
export XDG_CACHE_HOME="${cache_root}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
export TRITON_CACHE_DIR="${cache_root}/triton"
export VLLM_CACHE_ROOT="${cache_root}/vllm"
mkdir -p "$out_dir" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

echo "[threshold-eval] eval_name=${eval_name}"
echo "[threshold-eval] host=$(hostname)"
echo "[threshold-eval] cwd=${PWD}"
echo "[threshold-eval] python=${python_bin}"
echo "[threshold-eval] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[threshold-eval] model=${model}"
echo "[threshold-eval] test_jsonl=${test_jsonl} ($(wc -l < "${test_jsonl}") samples)"
echo "[threshold-eval] out_dir=${out_dir}"
echo "[threshold-eval] cache_root=${cache_root}"
echo "[threshold-eval] eval_batch_size=${EVAL_BATCH_SIZE:-512}"

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is empty" >&2
    exit 1
fi
if [ ! -d "$model" ]; then
    echo "ERROR: checkpoint directory not found: ${model}" >&2
    ls -ld "$(dirname "$model")" 2>/dev/null || true
    exit 1
fi
if [ ! -f "$test_jsonl" ]; then
    echo "ERROR: test jsonl not found: ${test_jsonl}" >&2
    exit 1
fi

"$python_bin" -m py_compile tools/evaluation/eval_student.py
"$python_bin" tools/evaluation/eval_student.py \
    --model "$model" \
    --test_jsonl "$test_jsonl" \
    --out_dir "$out_dir" \
    --tp "${TP:-1}" \
    --max_model_len "${MAX_MODEL_LEN:-12000}" \
    --gpu_mem_util "${GPU_MEM_UTIL:-0.88}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-128}" \
    --seed "${SEED:-42}" \
    --num_rollouts "${NUM_ROLLOUTS:-1}" \
    --rollout_temperature "${ROLLOUT_TEMPERATURE:-0.0}" \
    --top_p "${TOP_P:-0.95}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-512}"

"$python_bin" - <<'PY'
import json
import os

out_dir = os.environ["OUT_DIR"]
summary_path = os.path.join(out_dir, "summary.json")
per_sample_path = os.path.join(out_dir, "per_sample.jsonl")

with open(summary_path) as f:
    metrics = json.load(f)

ious = []
with open(per_sample_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        ious.append(float(json.loads(line)["iou"]))

thresholds = {
    f"IoU@{t:.1f}": round(sum(i > t for i in ious) / max(1, len(ious)), 4)
    for t in (0.5, 0.6, 0.7, 0.8, 0.9)
}
metrics.update(thresholds)

threshold_summary_path = os.path.join(out_dir, "threshold_summary.json")
with open(threshold_summary_path, "w") as f:
    json.dump(metrics, f, indent=2)
with open(summary_path, "w") as f:
    json.dump(metrics, f, indent=2)

print("[threshold-eval] summary_path=" + summary_path)
print("[threshold-eval] threshold_summary_path=" + threshold_summary_path)
print("[threshold-eval] model=" + metrics["model"])
print("[threshold-eval] n=%s mIoU=%.4f parse_rate=%.4f valid_rate=%.4f" % (
    metrics["n"], metrics["mIoU"], metrics["parse_rate"], metrics["valid_rate"]))
print("[threshold-eval] " + " ".join(f"{k}={v:.4f}" for k, v in thresholds.items()))
PY
EOS
)"
    } 2>&1 | tee "$log_path"
}

echo "[threshold-eval-launch] timestamp=${timestamp}"
echo "[threshold-eval-launch] group=${group} charged_group=${charged_group}"
echo "[threshold-eval-launch] cpu=${cpu} gpu=${gpu} memory=${memory}"
echo "[threshold-eval-launch] test_jsonl=${test_jsonl}"
echo "[threshold-eval-launch] out_root=${out_root}"

predict_one no_hint "$no_hint_node_tag"
predict_one soft_window "$soft_window_node_tag"

no_hint_rjob="${NO_HINT_RJOB_NAME:-eval-nohint-threshold-${timestamp}}"
soft_window_rjob="${SOFT_WINDOW_RJOB_NAME:-eval-softwindow-threshold-${timestamp}}"

launch_one no_hint "$no_hint_model" "$no_hint_node_tag" "$no_hint_rjob" &
no_hint_pid=$!
launch_one soft_window "$soft_window_model" "$soft_window_node_tag" "$soft_window_rjob" &
soft_window_pid=$!

set +e
wait "$no_hint_pid"
no_hint_status=$?
wait "$soft_window_pid"
soft_window_status=$?
set -e

echo "[threshold-eval-launch] no_hint_status=${no_hint_status}"
echo "[threshold-eval-launch] soft_window_status=${soft_window_status}"

python3 - "$out_root" <<'PY'
import json
import os
import sys

out_root = sys.argv[1]
print("\n=== SDPO threshold eval summary ===")
for name in ("no_hint", "soft_window"):
    summary_path = os.path.join(out_root, name, "threshold_summary.json")
    if not os.path.isfile(summary_path):
        print(f"{name}: missing summary at {summary_path}")
        continue
    with open(summary_path) as f:
        m = json.load(f)
    fields = ["mIoU", "IoU@0.5", "IoU@0.6", "IoU@0.7", "IoU@0.8", "IoU@0.9", "parse_rate", "valid_rate"]
    print(name + ": " + ", ".join(f"{k}={m[k]:.4f}" for k in fields) + f", summary={summary_path}")
PY

if [ "$no_hint_status" -ne 0 ] || [ "$soft_window_status" -ne 0 ]; then
    exit 1
fi
