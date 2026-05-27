#!/usr/bin/env bash
# Wait until one of the currently running SDPO hint ablations leaves Running,
# then launch the no-hint/no-mask ablation once.
set -euo pipefail

repo_root="/data/codes/gui_grounding/GUI-SD-code-main"
launcher="${repo_root}/tools/training/launch_qwen3_4b_rsfull_grpo_sdpo_hint_ablation_2node_rjob.sh"
namespace="${RJOB_NAMESPACE:-shai-core}"
poll_sec="${POLL_SEC:-300}"
max_wait_sec="${MAX_WAIT_SEC:-21600}"
log_dir="${LOG_DIR:-/data/codes/gui_grounding/data/logs/tmp}"
mkdir -p "$log_dir"

watched_jobs=(
    q4b-sdpo-soft-window-2n-05222050
    q4b-sdpo-jitter-box-2n-05222116-r3
    q4b-sdpo-zoom-in-2n-05222116-r3
)

running_count() {
    local count=0
    local job status line
    for job in "${watched_jobs[@]}"; do
        line="$(brainctl -n "$namespace" get "rjob/${job}" 2>/dev/null | tail -n +2 | head -n 1 || true)"
        status="$(awk '{print $2}' <<<"$line")"
        if [ "$status" = "Running" ]; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

start_ts="$(date +%s)"
echo "[wait-no-hint] started_at=$(date '+%F %T %Z')"
echo "[wait-no-hint] watched_jobs=${watched_jobs[*]}"

while true; do
    current_count="$(running_count)"
    echo "[wait-no-hint] $(date '+%F %T %Z') running_hint_jobs=${current_count}/${#watched_jobs[@]}"
    if [ "$current_count" -lt "${#watched_jobs[@]}" ]; then
        break
    fi
    now="$(date +%s)"
    if [ $((now - start_ts)) -ge "$max_wait_sec" ]; then
        echo "[wait-no-hint] timeout after ${max_wait_sec}s; no launch performed" >&2
        exit 1
    fi
    sleep "$poll_sec"
done

timestamp="$(date +%m%d%H%M)-deferred"
rjob_log="/data/codes/gui_grounding/data/logs/q4b-sdpo-no-hint-2n-${timestamp}.rjob.log"
echo "[wait-no-hint] launching no_hint at $(date '+%F %T %Z')"
echo "[wait-no-hint] rjob_log=${rjob_log}"

cd "$repo_root"
OPSD_MASK_MODE=no_mask \
OPSD_HINT_MODE=none \
MODE_SLUG=no-hint \
RUN_TIMESTAMP="$timestamp" \
RJOB_GROUP="${RJOB_GROUP:-gui_agent}" \
SAVE_STEPS=10 \
SAVE_TOTAL_LIMIT=20 \
RJOB_REPLICA_CREATION_TIMEOUT_SEC="${RJOB_REPLICA_CREATION_TIMEOUT_SEC:-900}" \
RJOB_LOG_PATH="$rjob_log" \
bash "$launcher"
