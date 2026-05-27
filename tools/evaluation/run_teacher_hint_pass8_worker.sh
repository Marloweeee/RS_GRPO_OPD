#!/usr/bin/env bash
# Run one teacher-hint pass@K evaluation inside a GPU rjob container.
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
hint_mode="${HINT_MODE:-gaussian}"

mkdir -p "$out_dir"

echo "[hint-pass8-worker] host=$(hostname)"
echo "[hint-pass8-worker] cwd=$PWD"
echo "[hint-pass8-worker] python=$python_bin"
echo "[hint-pass8-worker] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[hint-pass8-worker] model=$model"
echo "[hint-pass8-worker] eval_jsonl=$eval_jsonl"
echo "[hint-pass8-worker] out_dir=$out_dir"
echo "[hint-pass8-worker] hint_mode=$hint_mode"

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

"$python_bin" -m py_compile tools/evaluation/eval_teacher_hint_pass8.py

cache_root="/tmp/guisd_hint_pass8_eval_$(date +%s)_$$"
export XDG_CACHE_HOME="${cache_root}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
export TRITON_CACHE_DIR="${cache_root}/triton"
export VLLM_CACHE_ROOT="${cache_root}/vllm"
mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

set +e
"$python_bin" tools/evaluation/eval_teacher_hint_pass8.py \
    --model "$model" \
    --test_jsonl "$eval_jsonl" \
    --out_dir "$out_dir" \
    --hint_mode "$hint_mode" \
    --tp "${TP:-8}" \
    --num_rollouts "${NUM_ROLLOUTS:-8}" \
    --rollout_temperature "${ROLLOUT_TEMPERATURE:-0.7}" \
    --top_p "${TOP_P:-0.95}" \
    --max_model_len "${MAX_MODEL_LEN:-12000}" \
    --gpu_mem_util "${GPU_MEM_UTIL:-0.88}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-128}" \
    --seed "${SEED:-42}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-128}" \
    --opsd_gaussian_sigma_ratio "${OPSD_GAUSSIAN_SIGMA_RATIO:-1.5}" \
    --opsd_min_area_frac "${OPSD_MIN_AREA_FRAC:-0.1}" \
    --opsd_zoom_ratio "${OPSD_ZOOM_RATIO:-2.0}" \
    --opsd_jitter_ratio "${OPSD_JITTER_RATIO:-0.2}" \
    --opsd_hint_box_color "${OPSD_HINT_BOX_COLOR:-magenta}" 2>&1 | tee "${out_dir}/eval.log"
rc=${PIPESTATUS[0]}
set -e
rm -rf "$cache_root" 2>/dev/null || true

if [ "$rc" -ne 0 ]; then
    echo "{\"status\":\"failed\",\"rc\":$rc,\"model\":\"$model\",\"eval_jsonl\":\"$eval_jsonl\",\"hint_mode\":\"$hint_mode\",\"updated_at\":\"$(date '+%F %T')\"}" > "${out_dir}/status.json"
    exit "$rc"
fi

"$python_bin" - "$out_dir" "$model" "$eval_jsonl" "$hint_mode" <<'PY'
import json
import os
import sys
from datetime import datetime

out_dir, model, eval_jsonl, hint_mode = sys.argv[1:5]
summary_path = os.path.join(out_dir, "summary.json")
with open(summary_path) as f:
    metrics = json.load(f)
status = {
    "status": "done",
    "model": model,
    "eval_jsonl": eval_jsonl,
    "hint_mode": hint_mode,
    "summary_path": summary_path,
    "metrics": {
        key: metrics.get(key)
        for key in (
            "n", "hint_mode", "num_rollouts",
            "first_mIoU", "first_IoU@0.5", "first_IoU@0.6", "first_IoU@0.7", "first_IoU@0.8", "first_IoU@0.9",
            "best@8_mIoU", "pass@8@0.5", "pass@8@0.6", "pass@8@0.7", "pass@8@0.8", "pass@8@0.9",
            "first_parse_rate", "any_parse_rate", "first_valid_rate", "any_valid_rate",
        )
    },
    "updated_at": datetime.now().strftime("%F %T"),
}
with open(os.path.join(out_dir, "status.json"), "w") as f:
    json.dump(status, f, indent=2)
print("[hint-pass8-worker] summary_path=" + summary_path)
print("[hint-pass8-worker] metrics=" + json.dumps(status["metrics"], ensure_ascii=False))
PY

echo "[hint-pass8-worker] completed"
