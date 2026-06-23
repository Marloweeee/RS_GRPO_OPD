#!/usr/bin/env bash
# Start the three rollout-count ablations in a fresh tmux session.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
session="${TMUX_SESSION:-rollout_n_ablation_aos_${timestamp}}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs/rollout_n_ablation}"
mkdir -p "$log_root"

if tmux has-session -t "$session" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 2
fi

start_window() {
    local n="$1"
    local window_name="ngen${n}"
    local tmux_log="${log_root}/tmux_${window_name}_${timestamp}.log"
    local cmd="cd '${repo_root}' && RUN_TIMESTAMP='${timestamp}' LOG_ROOT='${log_root}' bash '${script_dir}/run_one_rollout_n_ablation_aos.sh' '${n}' 2>&1 | tee '${tmux_log}'"

    if [ "$n" = "2" ]; then
        tmux new-session -d -s "$session" -n "$window_name" "$cmd"
    else
        tmux new-window -t "$session" -n "$window_name" "$cmd"
    fi
}

start_window 2
start_window 4
start_window 16

echo "TMUX_SESSION=${session}"
echo "RUN_TIMESTAMP=${timestamp}"
echo "LOG_ROOT=${log_root}"
tmux list-windows -t "$session"
