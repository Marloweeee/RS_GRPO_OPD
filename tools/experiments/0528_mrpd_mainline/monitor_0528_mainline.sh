#!/usr/bin/env bash
# Lightweight monitor for 0528 MRPD mainline experiments.
set -euo pipefail

repo_root="${REPO_ROOT:-/data/codes/gui_grounding/GUI-SD-code-main}"
cd "$repo_root"

state_root="${STATE_ROOT:-/data/codes/gui_grounding/data/logs/0528_mainline}"
interval="${INTERVAL_SEC:-120}"
namespace="${RJOB_NAMESPACE:-shai-core}"
out_log="${MONITOR_LOG:-${state_root}/monitor_$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$state_root"

log() {
  echo "[$(date '+%F %T')][0528-monitor] $*" | tee -a "$out_log"
}

summarize_pass8() {
  local manifest="$1"
  local dir
  dir="$(dirname "$manifest")"
  python tools/experiments/0528_mrpd_mainline/aggregate_pass8_gap.py \
    --manifest "$manifest" \
    --out_dir "$dir" >> "$out_log" 2>&1 || true
}

parse_routes() {
  local log_path="$1"
  local out_dir="$2"
  local name="$3"
  if [ -f "$log_path" ]; then
    python tools/experiments/0528_mrpd_mainline/parse_route_stats.py \
      --log "$log_path" \
      --out_dir "$out_dir" \
      --name "$name" >> "$out_log" 2>&1 || true
  fi
}

while true; do
  log "tick state_root=${state_root}"

  for manifest in "$state_root"/pass8_*/manifest.json; do
    [ -f "$manifest" ] || continue
    done_count=0
    total_count=0
    eval_root="$(python - "$manifest" <<'PY'
import json, sys
m=json.load(open(sys.argv[1]))
print(m.get("eval_root",""))
PY
)"
    for status in "$eval_root"/*/status.json; do
      [ -f "$status" ] || continue
      total_count=$((total_count + 1))
      if grep -q '"status": "done"' "$status"; then
        done_count=$((done_count + 1))
      fi
    done
    log "pass8 manifest=${manifest} done_status=${done_count}/8"
    summarize_pass8 "$manifest"
  done

  for tsv in "$state_root"/distill_scope_*/rjobs.tsv; do
    [ -f "$tsv" ] || continue
    while IFS='|' read -r rjob scope group log_path tmux_name run_name; do
      [ -n "${rjob:-}" ] || continue
      status="$(brainctl -n "$namespace" get "rjob/${rjob}" 2>/dev/null | awk 'NR==2 {print $2}' || true)"
      if [ -n "${tmux_name:-}" ]; then
        tmux_status="$(tmux has-session -t "$tmux_name" 2>/dev/null && echo alive || echo missing)"
      else
        tmux_status="unknown"
      fi
      log "scope=${scope} rjob=${rjob} status=${status:-unknown} tmux=${tmux_name:-unknown}:${tmux_status}"
      if [ -f "$log_path" ]; then
        latest="$(grep -E "global_step/max_steps|Saving model checkpoint|last_model_checkpoint|Traceback|RuntimeError|ERROR|OutOfMemory|NCCL" "$log_path" 2>/dev/null | tail -n 3 | tr '\n' ' ' | cut -c1-900)"
        log "scope=${scope} latest=${latest}"
        parse_routes "$log_path" "$(dirname "$tsv")/route_${scope}" "$scope"
      fi
    done < "$tsv"
  done

  sleep "$interval"
done
