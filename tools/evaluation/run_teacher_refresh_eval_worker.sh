#!/usr/bin/env bash
# Evaluate one teacher-refresh checkpoint on the full RS test set.
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
test_jsonl="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
out_dir="${OUT_DIR:?OUT_DIR is required}"
eval_name="${EVAL_NAME:-$(basename "$(dirname "$(dirname "$model")")")_$(basename "$model")}"

cache_root="/tmp/guisd_teacher_refresh_eval_${eval_name}_$$"
export XDG_CACHE_HOME="${cache_root}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
export TRITON_CACHE_DIR="${cache_root}/triton"
export VLLM_CACHE_ROOT="${cache_root}/vllm"

mkdir -p "$out_dir" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

cleanup() {
    rm -rf "$cache_root" 2>/dev/null || true
}
trap cleanup EXIT

echo "[teacher-refresh-eval] eval_name=${eval_name}"
echo "[teacher-refresh-eval] host=$(hostname)"
echo "[teacher-refresh-eval] cwd=${PWD}"
echo "[teacher-refresh-eval] python=${python_bin}"
echo "[teacher-refresh-eval] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[teacher-refresh-eval] model=${model}"
echo "[teacher-refresh-eval] test_jsonl=${test_jsonl}"
echo "[teacher-refresh-eval] out_dir=${out_dir}"
echo "[teacher-refresh-eval] cache_root=${cache_root}"
echo "[teacher-refresh-eval] eval_batch_size=${EVAL_BATCH_SIZE:-512}"

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

echo "[teacher-refresh-eval] test_samples=$(wc -l < "$test_jsonl")"
find "$model" -maxdepth 1 -type f \( \
    -name 'config.json' -o \
    -name 'model.safetensors.index.json' -o \
    -name '*.safetensors' \
\) -printf '[teacher-refresh-eval] model_file=%f size=%s\n' | head -n 40

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
        if line:
            ious.append(float(json.loads(line)["iou"]))

thresholds = {
    f"IoU@{t:.1f}": round(sum(i > t for i in ious) / max(1, len(ious)), 4)
    for t in (0.5, 0.6, 0.7, 0.8, 0.9)
}
metrics.update(thresholds)

threshold_summary_path = os.path.join(out_dir, "threshold_summary.json")
with open(summary_path, "w") as f:
    json.dump(metrics, f, indent=2)
with open(threshold_summary_path, "w") as f:
    json.dump(metrics, f, indent=2)

print("[teacher-refresh-eval] summary_path=" + summary_path)
print("[teacher-refresh-eval] threshold_summary_path=" + threshold_summary_path)
print("[teacher-refresh-eval] n=%s mIoU=%.4f parse_rate=%.4f valid_rate=%.4f" % (
    metrics["n"], metrics["mIoU"], metrics["parse_rate"], metrics["valid_rate"]))
print("[teacher-refresh-eval] " + " ".join(f"{k}={v:.4f}" for k, v in thresholds.items()))
PY
