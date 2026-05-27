#!/usr/bin/env bash
# Evaluate a shard of 4B checkpoints on rs_val.jsonl.
#
# The worker is restart-friendly: every checkpoint writes status.json before and
# after evaluation, and completed summary.json files are skipped on rerun.
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
manifest="${MANIFEST:?MANIFEST is required}"
shard_index="${SHARD_INDEX:?SHARD_INDEX is required}"
num_shards="${NUM_SHARDS:?NUM_SHARDS is required}"
eval_jsonl="${EVAL_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_val.jsonl}"
out_root="${OUT_ROOT:?OUT_ROOT is required}"

mkdir -p "$out_root"

echo "[val-ckpt-eval] host=$(hostname)"
echo "[val-ckpt-eval] cwd=$PWD"
echo "[val-ckpt-eval] python=$python_bin"
echo "[val-ckpt-eval] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[val-ckpt-eval] manifest=$manifest"
echo "[val-ckpt-eval] shard=${shard_index}/${num_shards}"
echo "[val-ckpt-eval] eval_jsonl=$eval_jsonl"
echo "[val-ckpt-eval] out_root=$out_root"
echo "[val-ckpt-eval] samples=$(wc -l < "$eval_jsonl")"

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is empty" >&2
    exit 1
fi
if [ ! -f "$manifest" ]; then
    echo "ERROR: manifest not found: $manifest" >&2
    exit 1
fi
if [ ! -f "$eval_jsonl" ]; then
    echo "ERROR: eval jsonl not found: $eval_jsonl" >&2
    exit 1
fi

"$python_bin" -m py_compile tools/evaluation/eval_student.py

"$python_bin" - "$manifest" "$shard_index" "$num_shards" "$out_root" "$eval_jsonl" <<'PY'
import json
import os
import sys

manifest, shard_index, num_shards, out_root, eval_jsonl = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
selected = []
with open(manifest) as f:
    for idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if idx % num_shards == shard_index:
            selected.append(item)

shard_path = os.path.join(out_root, f"shard_{shard_index:02d}_items.jsonl")
with open(shard_path, "w") as f:
    for item in selected:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"[val-ckpt-eval] selected={len(selected)} shard_items={shard_path}")
PY

shard_items="${out_root}/shard_$(printf '%02d' "$shard_index")_items.jsonl"
while IFS= read -r line; do
    [ -n "$line" ] || continue
    eval "$(LINE="$line" "$python_bin" - <<'PY'
import json
import os
import shlex

item = json.loads(os.environ["LINE"])
for key in ("id", "group", "name", "model", "out_dir"):
    print(f"{key.upper()}={shlex.quote(str(item[key]))}")
PY
)"
    mkdir -p "$OUT_DIR"
    status_json="${OUT_DIR}/status.json"
    summary_json="${OUT_DIR}/summary.json"
    threshold_json="${OUT_DIR}/threshold_summary.json"
    eval_log="${OUT_DIR}/eval.log"

    if [ -f "$summary_json" ] && [ -f "$threshold_json" ]; then
        echo "[val-ckpt-eval][$ID] skip completed: $summary_json"
        continue
    fi
    if [ ! -d "$MODEL" ]; then
        echo "{\"id\":\"$ID\",\"status\":\"failed\",\"reason\":\"checkpoint_not_found\",\"model\":\"$MODEL\",\"updated_at\":\"$(date '+%F %T')\"}" > "$status_json"
        echo "[val-ckpt-eval][$ID] checkpoint not found: $MODEL" >&2
        continue
    fi

    cache_root="/tmp/guisd_val_eval_${ID}_$$"
    export XDG_CACHE_HOME="${cache_root}/xdg"
    export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
    export TRITON_CACHE_DIR="${cache_root}/triton"
    export VLLM_CACHE_ROOT="${cache_root}/vllm"
    mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

    echo "{\"id\":\"$ID\",\"status\":\"running\",\"model\":\"$MODEL\",\"eval_jsonl\":\"$eval_jsonl\",\"started_at\":\"$(date '+%F %T')\"}" > "$status_json"
    echo "[val-ckpt-eval][$ID] start name=$NAME model=$MODEL"
    set +e
    "$python_bin" tools/evaluation/eval_student.py \
        --model "$MODEL" \
        --test_jsonl "$eval_jsonl" \
        --out_dir "$OUT_DIR" \
        --tp "${TP:-8}" \
        --max_model_len "${MAX_MODEL_LEN:-12000}" \
        --gpu_mem_util "${GPU_MEM_UTIL:-0.88}" \
        --max_new_tokens "${MAX_NEW_TOKENS:-128}" \
        --seed "${SEED:-42}" \
        --num_rollouts "${NUM_ROLLOUTS:-1}" \
        --rollout_temperature "${ROLLOUT_TEMPERATURE:-0.0}" \
        --top_p "${TOP_P:-0.95}" \
        --eval_batch_size "${EVAL_BATCH_SIZE:-512}" 2>&1 | tee "$eval_log"
    rc=${PIPESTATUS[0]}
    set -e

    rm -rf "$cache_root" 2>/dev/null || true
    if [ "$rc" -ne 0 ]; then
        echo "{\"id\":\"$ID\",\"status\":\"failed\",\"rc\":$rc,\"model\":\"$MODEL\",\"updated_at\":\"$(date '+%F %T')\"}" > "$status_json"
        echo "[val-ckpt-eval][$ID] failed rc=$rc" >&2
        continue
    fi

    "$python_bin" - "$OUT_DIR" <<'PY'
import json
import os
import sys

out_dir = sys.argv[1]
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
threshold_summary_path = os.path.join(out_dir, "threshold_summary.json")
with open(summary_path, "w") as f:
    json.dump(metrics, f, indent=2)
with open(threshold_summary_path, "w") as f:
    json.dump(metrics, f, indent=2)
print("[val-ckpt-eval] metrics " + json.dumps({
    "n": metrics["n"],
    "mIoU": metrics["mIoU"],
    "IoU@0.5": metrics["IoU@0.5"],
    "IoU@0.6": metrics["IoU@0.6"],
    "IoU@0.7": metrics["IoU@0.7"],
    "IoU@0.8": metrics["IoU@0.8"],
    "IoU@0.9": metrics["IoU@0.9"],
}, ensure_ascii=False))
PY
    "$python_bin" - "$status_json" "$summary_json" "$ID" "$NAME" "$GROUP" "$MODEL" <<'PY'
import json
import sys
from datetime import datetime

status_path, summary_path, item_id, name, group, model = sys.argv[1:7]
with open(summary_path) as f:
    metrics = json.load(f)
status = {
    "id": item_id,
    "name": name,
    "group": group,
    "model": model,
    "status": "done",
    "summary_path": summary_path,
    "metrics": {
        key: metrics.get(key)
        for key in ("n", "mIoU", "IoU@0.5", "IoU@0.6", "IoU@0.7", "IoU@0.8", "IoU@0.9", "parse_rate", "valid_rate")
    },
    "updated_at": datetime.now().strftime("%F %T"),
}
with open(status_path, "w") as f:
    json.dump(status, f, indent=2)
PY
    echo "[val-ckpt-eval][$ID] done summary=$summary_json"
done < "$shard_items"

echo "[val-ckpt-eval] shard ${shard_index}/${num_shards} completed"
