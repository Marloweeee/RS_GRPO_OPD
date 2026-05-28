#!/usr/bin/env bash
# Stop auxiliary monitors once all 0528 acceptance tasks are complete.
set -euo pipefail

state_root="${STATE_ROOT:-/data/codes/gui_grounding/data/logs/0528_mainline}"
status_json="${state_root}/0528_acceptance_status.json"
interval="${INTERVAL_SEC:-120}"
max_ticks="${MAX_TICKS:-360}"
namespace="${RJOB_NAMESPACE:-shai-core}"
cpu_monitor_rjob="${CPU_MONITOR_RJOB:-m28-cpu-mon-0528cpu3}"
watch_tmux="${WATCH_TMUX:-mrpd0528_gpu_watch}"
cpu_tmux="${CPU_TMUX:-mrpd0528_cpu_monitor}"
log_path="${LOG_PATH:-${state_root}/cleanup_monitors_$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$log_path")"

log() {
  echo "[$(date '+%F %T')][0528-cleanup] $*" | tee -a "$log_path"
}

acceptance_complete() {
  [ -f "$status_json" ] || return 1
  python - "$status_json" <<'PY'
import json
import sys
from pathlib import Path

try:
    data = json.loads(Path(sys.argv[1]).read_text())
except Exception:
    sys.exit(1)
tasks = data.get("tasks", {})
required = (
    "pass8_gap_reduction",
    "route_mechanism_analysis",
    "selective_distillation_ablation",
)
sys.exit(0 if all(tasks.get(name, {}).get("complete") for name in required) else 1)
PY
}

stop_rjob_if_active() {
  local rjob="$1"
  local status
  status="$(brainctl -n "$namespace" get "rjob/${rjob}" 2>/dev/null | awk 'NR==2 {print $2}' || true)"
  case "$status" in
    Running|Pending|Starting)
      log "stopping rjob/${rjob} status=${status}"
      brainctl -n "$namespace" stop "rjob/${rjob}" >> "$log_path" 2>&1 || true
      ;;
    "")
      log "rjob/${rjob} not found"
      ;;
    *)
      log "rjob/${rjob} status=${status}; no stop needed"
      ;;
  esac
}

kill_tmux_if_exists() {
  local session="$1"
  if tmux has-session -t "$session" 2>/dev/null; then
    log "killing tmux session ${session}"
    tmux kill-session -t "$session" >> "$log_path" 2>&1 || true
  else
    log "tmux session ${session} not found"
  fi
}

log "start status_json=${status_json} interval=${interval} max_ticks=${max_ticks}"
tick=0
while [ "$tick" -lt "$max_ticks" ]; do
  tick=$((tick + 1))
  if acceptance_complete; then
    log "acceptance complete; cleaning auxiliary monitors"
    stop_rjob_if_active "$cpu_monitor_rjob"
    kill_tmux_if_exists "$watch_tmux"
    kill_tmux_if_exists "$cpu_tmux"
    log "cleanup done"
    exit 0
  fi
  log "tick=${tick}; acceptance incomplete"
  sleep "$interval"
done

log "max_ticks reached without complete acceptance"
