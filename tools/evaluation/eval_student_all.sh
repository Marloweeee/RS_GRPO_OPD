#!/bin/bash
# Loop over student ckpts, eval each on rs_test, then aggregate.
# Usage:
#   sh tools/evaluation/eval_student_all.sh <run_dir> [out_base] [test_jsonl]
#   - run_dir: contains checkpoint-*/ subdirs (e.g. .../gui-sd-4b_student_rs/v0-20260517-184421)
#   - out_base: results dir; defaults to /data/.../eval_<run_name> (run_name = parent of v*)
#   - test_jsonl: defaults to the 349-sample subset for direct comparison across runs

set -u

RUN_DIR=${1:-"/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-4b_student_rs/v0-20260517-184421"}
RUN_NAME=$(basename "$(dirname "$RUN_DIR")")
OUT_BASE=${2:-"/data/codes/gui_grounding/data/rs_sub_dir/eval_${RUN_NAME}"}
TEST_JSONL=${3:-"/data/codes/gui_grounding/data/rs_sub_dir/rs_test.jsonl"}
TEACHER_BASELINE_IOU50=0.579   # eval_teacher_hint v2 'none' 模式

mkdir -p "$OUT_BASE"

ckpts=$(ls -1d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
if [ -z "$ckpts" ]; then
    echo "[eval_all] no checkpoint-* in $RUN_DIR" >&2
    exit 1
fi

echo "[eval_all] run_dir = $RUN_DIR"
echo "[eval_all] out_base = $OUT_BASE"
echo "[eval_all] test_jsonl = $TEST_JSONL ($(wc -l < $TEST_JSONL) samples)"
echo "[eval_all] checkpoints:"
echo "$ckpts" | sed 's/^/  /'

for ckpt in $ckpts; do
    name=$(basename "$ckpt")
    out="$OUT_BASE/$name"
    if [ -f "$out/summary.json" ]; then
        echo "[eval_all] $name already done, skipping"
        continue
    fi
    echo ""
    echo "============================================================"
    echo "[eval_all] $name"
    echo "============================================================"
    IMAGE_MAX_TOKEN_NUM=10000 python tools/evaluation/eval_student.py \
        --model "$ckpt" --test_jsonl "$TEST_JSONL" --out_dir "$out" --tp 1
done

# 拼对比表
python3 - <<PY
import json, os
out_base = "$OUT_BASE"
run_name = "$RUN_NAME"
baseline_iou50 = $TEACHER_BASELINE_IOU50
rows = []
for name in sorted(os.listdir(out_base), key=lambda n: int(n.split('-')[-1]) if n.startswith('checkpoint-') else 0):
    p = os.path.join(out_base, name, 'summary.json')
    if not os.path.isfile(p):
        continue
    m = json.load(open(p))
    rows.append((name, m))

print()
print(f'=== {run_name} (vs teacher zero-shot baseline IoU@0.5=%.3f) ===' % baseline_iou50)
print()
print(f'{"ckpt":<18s} | {"mIoU":>7s} | {"IoU@0.5":>8s} | {"IoU@0.7":>8s} | {"parse":>6s} | {"mIoU|parsed":>11s} | {"Δ vs teacher":>13s}')
print('-' * 105)
for name, m in rows:
    delta = m['IoU@0.5'] - baseline_iou50
    print(f'{name:<18s} | {m["mIoU"]:>7.4f} | {m["IoU@0.5"]:>8.4f} | {m["IoU@0.7"]:>8.4f} | {m["parse_rate"]:>6.3f} | {m["mIoU_parsed"]:>11.4f} | {delta:>+13.4f}')
PY
