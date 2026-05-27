#!/usr/bin/env bash
# Watch SFT->OPSD-only, evaluate its final checkpoint, then launch the next P0
# no-hint/no-mask GRPO+OPSD ablation on gui_agent.
set -euo pipefail

repo_root="${REPO_ROOT:-/data/codes/gui_grounding/GUI-SD-code-main}"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
training_log="${TRAINING_LOG:-/data/codes/gui_grounding/data/logs/training/q4b-sft-opsd-only-gauss-e1-20260527-082554.rjob.log}"
opsd_root="${OPSD_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rrsisd_sft_opsd_only_gaussian_e1-20260527-082554}"
watch_log="${WATCH_LOG:-/data/codes/gui_grounding/data/logs/training/watch_opsd_only_then_launch_p0_no_hint-${timestamp}.log}"
status_file="${STATUS_FILE:-/data/codes/gui_grounding/data/logs/training/watch_opsd_only_then_launch_p0_no_hint.status}"
eval_jsonl="${EVAL_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"

mkdir -p "$(dirname "$watch_log")" /data/codes/gui_grounding/data/logs/eval

log() {
    echo "[watch-opsd-p0][$(date '+%F %T')] $*" | tee -a "$watch_log"
}

latest_ckpt() {
    python3 - "$opsd_root" <<'PY'
import glob
import os
import sys

root = sys.argv[1]
items = []
for path in glob.glob(root + "/*/checkpoint-*"):
    try:
        step = int(os.path.basename(path).split("-")[-1])
    except ValueError:
        step = -1
    items.append((step, path))
if items:
    print(sorted(items)[-1][1])
PY
}

already_done() {
    [ -f "$status_file" ] && grep -q "^launched_no_hint=1$" "$status_file"
}

if already_done; then
    log "status file says no-hint has already been launched; exiting"
    exit 0
fi

log "started timestamp=${timestamp}"
log "training_log=${training_log}"
log "opsd_root=${opsd_root}"
log "eval_jsonl=${eval_jsonl}"

deadline=$(( $(date +%s) + ${MAX_WAIT_SEC:-7200} ))
while true; do
    if [ ! -f "$training_log" ]; then
        log "waiting for training log"
    else
        completed_count="$(grep -c "\\[4b-opd-only-2node-worker\\].*completed" "$training_log" || true)"
        ckpt="$(latest_ckpt || true)"
        log "completed_count=${completed_count} latest_ckpt=${ckpt:-none}"
        if [ "${completed_count}" -ge 2 ] && [ -n "${ckpt:-}" ] && [ -d "$ckpt" ]; then
            break
        fi
        if grep -qE "Traceback \\(most recent call last\\)|RuntimeError:|CUDA out of memory|OutOfMemoryError|NCCL.*error|\\bERROR\\b|Failed to|failed with" "$training_log"; then
            log "strict error detected in OPSD-only log; not launching follow-up automatically"
            exit 1
        fi
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
        log "timeout waiting for OPSD-only completion"
        exit 1
    fi
    sleep "${POLL_SEC:-180}"
done

eval_slug="q4b-sft-opsd-only-gaussian-${timestamp}-test"
eval_out="/data/codes/gui_grounding/data/logs/eval/${eval_slug}"
eval_rjob="eval-${eval_slug}"
log "launching OPSD-only test eval: model=${ckpt} out=${eval_out}"
RJOB_GROUP="${EVAL_RJOB_GROUP:-gui_agent}" \
RJOB_CHARGED_GROUP="${EVAL_RJOB_CHARGED_GROUP:-gui_agent}" \
RJOB_CPU="${EVAL_RJOB_CPU:-64}" \
RJOB_GPU="${EVAL_RJOB_GPU:-8}" \
RJOB_MEMORY="${EVAL_RJOB_MEMORY:-700000}" \
RJOB_NAME="$eval_rjob" \
MODEL="$ckpt" \
EVAL_JSONL="$eval_jsonl" \
OUT_DIR="$eval_out" \
LOG_ROOT="/data/codes/gui_grounding/data/logs/eval" \
bash "$repo_root/tools/evaluation/launch_single_ckpt_split_eval_detached_rjob.sh" | tee -a "$watch_log"

nohint_timestamp="${NOHINT_RUN_TIMESTAMP:-${timestamp}-after-opsd}"
log "launching P0 no-hint/no-mask ablation timestamp=${nohint_timestamp}"
RUN_TIMESTAMP="$nohint_timestamp" \
RJOB_GROUP="${NOHINT_RJOB_GROUP:-gui_agent}" \
RJOB_CHARGED_GROUP="${NOHINT_RJOB_CHARGED_GROUP:-gui_agent}" \
RJOB_CPU="${NOHINT_RJOB_CPU:-64}" \
RJOB_MEMORY="${NOHINT_RJOB_MEMORY:-700000}" \
MASTER_PORT="${NOHINT_MASTER_PORT:-30160}" \
VLLM_SERVER_PORT="${NOHINT_VLLM_SERVER_PORT:-8932}" \
OPSD_MASK_MODE=no_mask \
OPSD_HINT_MODE=none \
bash "$repo_root/tools/training/sft/launch_mrpd_sft_ablation_2node.sh" sft_grpo_opsd_no_hint | tee -a "$watch_log"

{
    echo "launched_no_hint=1"
    echo "timestamp=${timestamp}"
    echo "opsd_ckpt=${ckpt}"
    echo "opsd_eval_out=${eval_out}"
    echo "nohint_timestamp=${nohint_timestamp}"
} > "$status_file"

log "done"
