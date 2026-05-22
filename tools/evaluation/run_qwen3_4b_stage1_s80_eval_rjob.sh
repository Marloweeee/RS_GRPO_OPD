#!/usr/bin/env bash
# Evaluate the Qwen3-VL-4B stage-1 (80-step) checkpoint on the full rs test set.
#
# The job runs on one GPU and writes metrics to /mnt artifacts by default.
# Torch/Triton/vLLM caches are redirected to pod-local /tmp to avoid stale NFS
# compile-cache handles during vLLM startup.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

timestamp="$(date +%Y%m%d-%H%M%S)"
rjob_name="${RJOB_NAME:-qwen3-4b-stage1-s80-eval-${timestamp}}"
log_path="${RJOB_LOG_PATH:-/data/codes/gui_grounding/data/${rjob_name}.rjob.log}"

model="${MODEL:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-base_rsfull_grpo_sdpo_2stage_s80_e1-20260521-231115_stage1_s80/v0-20260521-231853/checkpoint-80}"
test_jsonl="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
out_dir="${OUT_DIR:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd/eval/gui-sd-qwen3-4b-stage1-s80-fulltest/checkpoint-80_greedy_${timestamp}}"

group="${RJOB_GROUP:-gui_agent}"
charged_group="${RJOB_CHARGED_GROUP:-$group}"
cpu="${RJOB_CPU:-16}"
gpu="${RJOB_GPU:-1}"
memory="${RJOB_MEMORY:-160000}"
positive_tags="${RJOB_POSITIVE_TAGS:-}"
mount_arg="--mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot"

tp="${TP:-1}"
gpu_mem_util="${GPU_MEM_UTIL:-0.88}"
eval_batch_size="${EVAL_BATCH_SIZE:-512}"
max_model_len="${MAX_MODEL_LEN:-12000}"
max_new_tokens="${MAX_NEW_TOKENS:-128}"

mkdir -p "$(dirname "$log_path")"

predict_cmd=(
    brainctl launch
    --cpu "$cpu"
    --gpu "$gpu"
    --memory "$memory"
    --group "$group"
    --charged-group="$charged_group"
    --private-machine=group
    "$mount_arg"
    --predict-only
)
if [ -n "$positive_tags" ]; then
    predict_cmd+=(--positive-tags "$positive_tags")
fi

launch_cmd=(
    brainctl launch
    --name "$rjob_name"
    --cpu "$cpu"
    --gpu "$gpu"
    --memory="$memory"
    --group "$group"
    --charged-group="$charged_group"
    --private-machine=group
    "$mount_arg"
    --set-env "MODEL=$model"
    --set-env "TEST_JSONL=$test_jsonl"
    --set-env "OUT_DIR=$out_dir"
    --set-env "TP=$tp"
    --set-env "GPU_MEM_UTIL=$gpu_mem_util"
    --set-env "EVAL_BATCH_SIZE=$eval_batch_size"
    --set-env "MAX_MODEL_LEN=$max_model_len"
    --set-env "MAX_NEW_TOKENS=$max_new_tokens"
    --set-env "RJOB_NAME=$rjob_name"
)
if [ -n "$positive_tags" ]; then
    launch_cmd+=(--positive-tags "$positive_tags")
fi

launch_cmd+=(
    --
    bash -lc
    "$(cat <<'EOS'
set -euo pipefail
cd /data/codes/gui_grounding/GUI-SD-code-main

export PATH=/data/codes/gui_grounding/conda_envs/GUI-SD/bin:${PATH}
export PYTHONPATH=/data/codes/gui_grounding/GUI-SD-code-main:${PYTHONPATH:-}
export PYTHON_BIN=/data/codes/gui_grounding/conda_envs/GUI-SD/bin/python
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-10000}

cache_root="/tmp/guisd_vllm_eval_cache_${RJOB_NAME:-eval}_$$"
export XDG_CACHE_HOME="${cache_root}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
export TRITON_CACHE_DIR="${cache_root}/triton"
export VLLM_CACHE_ROOT="${cache_root}/vllm"
mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

echo "[4b-stage1-eval] host=$(hostname)"
echo "[4b-stage1-eval] python=${PYTHON_BIN}"
echo "[4b-stage1-eval] model=${MODEL}"
echo "[4b-stage1-eval] test_jsonl=${TEST_JSONL} ($(wc -l < "${TEST_JSONL}") samples)"
echo "[4b-stage1-eval] out_dir=${OUT_DIR}"
echo "[4b-stage1-eval] cache_root=${cache_root}"
echo "[4b-stage1-eval] tp=${TP} gpu_mem_util=${GPU_MEM_UTIL} eval_batch_size=${EVAL_BATCH_SIZE}"

if [ ! -d "${MODEL}" ]; then
    echo "[4b-stage1-eval] ERROR: model checkpoint not found: ${MODEL}" >&2
    exit 1
fi
if [ ! -f "${MODEL}/config.json" ]; then
    echo "[4b-stage1-eval] ERROR: missing config.json under ${MODEL}" >&2
    exit 1
fi
if [ ! -f "${TEST_JSONL}" ]; then
    echo "[4b-stage1-eval] ERROR: test jsonl not found: ${TEST_JSONL}" >&2
    exit 1
fi

"${PYTHON_BIN}" -m py_compile tools/evaluation/eval_student.py
"${PYTHON_BIN}" tools/evaluation/eval_student.py \
    --model "${MODEL}" \
    --test_jsonl "${TEST_JSONL}" \
    --out_dir "${OUT_DIR}" \
    --tp "${TP}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --gpu_mem_util "${GPU_MEM_UTIL}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed 42

"${PYTHON_BIN}" - <<'PY'
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
        if line.strip():
            ious.append(float(json.loads(line)["iou"]))

thresholds = {f"IoU@{t:.1f}": round(sum(i > t for i in ious) / max(1, len(ious)), 4)
              for t in (0.5, 0.6, 0.7, 0.8, 0.9)}
metrics.update(thresholds)
threshold_summary_path = os.path.join(out_dir, "threshold_summary.json")
with open(threshold_summary_path, "w") as f:
    json.dump(metrics, f, indent=2)

print("[4b-stage1-eval] summary_path=" + summary_path)
print("[4b-stage1-eval] threshold_summary_path=" + threshold_summary_path)
print("[4b-stage1-eval] model=" + metrics["model"])
print("[4b-stage1-eval] n=%s mIoU=%.4f parse_rate=%.4f valid_rate=%.4f" % (
    metrics["n"], metrics["mIoU"], metrics["parse_rate"], metrics["valid_rate"]))
print("[4b-stage1-eval] " + " ".join(f"{k}={v:.4f}" for k, v in thresholds.items()))
PY
EOS
)"
)

{
    echo "[launch-4b-stage1-eval] predict: ${predict_cmd[*]}"
    "${predict_cmd[@]}"
    echo "[launch-4b-stage1-eval] rjob_name=${rjob_name}"
    echo "[launch-4b-stage1-eval] log_path=${log_path}"
    echo "[launch-4b-stage1-eval] group=${group}"
    echo "[launch-4b-stage1-eval] model=${model}"
    echo "[launch-4b-stage1-eval] test_jsonl=${test_jsonl}"
    echo "[launch-4b-stage1-eval] out_dir=${out_dir}"
    echo "[launch-4b-stage1-eval] command: ${launch_cmd[*]}"
    "${launch_cmd[@]}"
} 2>&1 | tee "$log_path"
