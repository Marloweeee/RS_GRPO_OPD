#!/usr/bin/env bash
# Evaluate one checkpoint on one jsonl split inside a GPU rjob container.
set -euo pipefail

repo_root="${REPO_ROOT:-/data/codes/gui_grounding/GUI-SD-code-main}"
env_root="${GUI_SD_ENV_ROOT:-/data/codes/gui_grounding/conda_envs/GUI-SD}"
cd "$repo_root"

export PATH="${env_root}/bin:${PATH}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONFAULTHANDLER=1
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-10000}"
export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"

python_bin="${PYTHON_BIN:-${env_root}/bin/python}"
model="${MODEL:?MODEL is required}"
eval_jsonl="${EVAL_JSONL:?EVAL_JSONL is required}"
out_dir="${OUT_DIR:?OUT_DIR is required}"

mkdir -p "$out_dir"

echo "[single-eval] host=$(hostname)"
echo "[single-eval] cwd=$PWD"
echo "[single-eval] python=$python_bin"
echo "[single-eval] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[single-eval] model=$model"
echo "[single-eval] eval_jsonl=$eval_jsonl"
echo "[single-eval] out_dir=$out_dir"

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is empty" >&2
    exit 1
fi
if [ ! -d "$model" ]; then
    echo "ERROR: model checkpoint not found: $model" >&2
    exit 1
fi
if [ ! -f "$eval_jsonl" ]; then
    echo "ERROR: eval jsonl not found: $eval_jsonl" >&2
    exit 1
fi

"$python_bin" -m py_compile tools/evaluation/eval_student.py

cache_root="/tmp/guisd_single_eval_$(date +%s)_$$"
export XDG_CACHE_HOME="${cache_root}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
export TRITON_CACHE_DIR="${cache_root}/triton"
export VLLM_CACHE_ROOT="${cache_root}/vllm"
mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

set +e
"$python_bin" tools/evaluation/eval_student.py \
    --model "$model" \
    --test_jsonl "$eval_jsonl" \
    --out_dir "$out_dir" \
    --tp "${TP:-8}" \
    --max_model_len "${MAX_MODEL_LEN:-12000}" \
    --gpu_mem_util "${GPU_MEM_UTIL:-0.88}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-128}" \
    --seed "${SEED:-42}" \
    --num_rollouts "${NUM_ROLLOUTS:-1}" \
    --rollout_temperature "${ROLLOUT_TEMPERATURE:-0.0}" \
    --top_p "${TOP_P:-0.95}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-512}" 2>&1 | tee "${out_dir}/eval.log"
rc=${PIPESTATUS[0]}
set -e
rm -rf "$cache_root" 2>/dev/null || true
if [ "$rc" -ne 0 ]; then
    echo "{\"status\":\"failed\",\"rc\":$rc,\"model\":\"$model\",\"eval_jsonl\":\"$eval_jsonl\",\"updated_at\":\"$(date '+%F %T')\"}" > "${out_dir}/status.json"
    exit "$rc"
fi

"$python_bin" - "$out_dir" "$model" "$eval_jsonl" <<'PY'
import json
import os
import sys
from datetime import datetime

out_dir, model, eval_jsonl = sys.argv[1:4]
summary_path = os.path.join(out_dir, "summary.json")
per_sample_path = os.path.join(out_dir, "per_sample.jsonl")
with open(summary_path) as f:
    metrics = json.load(f)
ious = []
with open(per_sample_path) as f:
    for line in f:
        if line.strip():
            ious.append(float(json.loads(line)["iou"]))
for t in (0.5, 0.6, 0.7, 0.8, 0.9):
    metrics[f"IoU@{t:.1f}"] = round(sum(i > t for i in ious) / max(1, len(ious)), 4)
with open(summary_path, "w") as f:
    json.dump(metrics, f, indent=2)
threshold_path = os.path.join(out_dir, "threshold_summary.json")
with open(threshold_path, "w") as f:
    json.dump(metrics, f, indent=2)
status = {
    "status": "done",
    "model": model,
    "eval_jsonl": eval_jsonl,
    "summary_path": summary_path,
    "threshold_summary_path": threshold_path,
    "metrics": {
        key: metrics.get(key)
        for key in ("n", "mIoU", "IoU@0.5", "IoU@0.6", "IoU@0.7", "IoU@0.8", "IoU@0.9", "parse_rate", "valid_rate")
    },
    "updated_at": datetime.now().strftime("%F %T"),
}
with open(os.path.join(out_dir, "status.json"), "w") as f:
    json.dump(status, f, indent=2)
print("[single-eval] summary_path=" + summary_path)
print("[single-eval] metrics=" + json.dumps(status["metrics"], ensure_ascii=False))
PY

echo "[single-eval] completed"
