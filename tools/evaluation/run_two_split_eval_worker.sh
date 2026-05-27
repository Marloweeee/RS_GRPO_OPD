#!/usr/bin/env bash
# Run val and test evaluation for one checkpoint inside a GPU rjob container.
set -euo pipefail

repo_root="${REPO_ROOT:-/data/codes/gui_grounding/GUI-SD-code-main}"
cd "$repo_root"

model="${MODEL:?MODEL is required}"
val_jsonl="${VAL_JSONL:?VAL_JSONL is required}"
test_jsonl="${TEST_JSONL:?TEST_JSONL is required}"
val_out_dir="${VAL_OUT_DIR:?VAL_OUT_DIR is required}"
test_out_dir="${TEST_OUT_DIR:?TEST_OUT_DIR is required}"
single_worker="${SINGLE_WORKER:-${repo_root}/tools/evaluation/run_single_ckpt_eval_worker.sh}"

run_split() {
    local split="$1"
    local jsonl="$2"
    local out_dir="$3"
    mkdir -p "$out_dir"
    echo "[two-split-eval] split=${split} start=$(date '+%F %T')"
    MODEL="$model" EVAL_JSONL="$jsonl" OUT_DIR="$out_dir" bash "$single_worker"
    echo "[two-split-eval] split=${split} end=$(date '+%F %T')"
}

overall_status=0
if ! run_split val "$val_jsonl" "$val_out_dir"; then
    overall_status=1
fi
if ! run_split test "$test_jsonl" "$test_out_dir"; then
    overall_status=1
fi

echo "[two-split-eval] completed status=${overall_status}"
exit "$overall_status"
