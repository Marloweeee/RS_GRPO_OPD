#!/usr/bin/env bash
# Evaluate one SDPO ablation checkpoint with eval_student.py inside a GPU rjob.
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

mkdir -p "$out_dir"

echo "[sdpo-ck870-eval] eval_name=${eval_name}"
echo "[sdpo-ck870-eval] host=$(hostname)"
echo "[sdpo-ck870-eval] cwd=${PWD}"
echo "[sdpo-ck870-eval] python=${python_bin}"
echo "[sdpo-ck870-eval] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[sdpo-ck870-eval] model=${model}"
echo "[sdpo-ck870-eval] test_jsonl=${test_jsonl}"
echo "[sdpo-ck870-eval] out_dir=${out_dir}"
echo "[sdpo-ck870-eval] eval_batch_size=${EVAL_BATCH_SIZE:-512}"

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

find "$model" -maxdepth 1 -type f \( \
    -name 'config.json' -o \
    -name 'model.safetensors.index.json' -o \
    -name '*.safetensors' \
\) -printf '[sdpo-ck870-eval] model_file=%f size=%s\n' | head -n 40

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

"$python_bin" - <<PY
import json
summary_path = "${out_dir}/summary.json"
with open(summary_path) as f:
    metrics = json.load(f)
print("[sdpo-ck870-eval] summary_path=" + summary_path)
print("[sdpo-ck870-eval] IoU@0.5=%.4f" % metrics["IoU@0.5"])
print("[sdpo-ck870-eval] mIoU=%.4f IoU@0.7=%.4f parse_rate=%.4f valid_rate=%.4f" % (
    metrics["mIoU"],
    metrics["IoU@0.7"],
    metrics["parse_rate"],
    metrics["valid_rate"],
))
PY
