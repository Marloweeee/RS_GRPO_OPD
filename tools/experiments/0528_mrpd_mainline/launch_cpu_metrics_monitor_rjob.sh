#!/usr/bin/env bash
# Launch the 0528 CPU metrics monitor in a CPU-only rjob.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%m%d%H%M)}"
rjob_name="${RJOB_NAME:-m28-cpu-mon-${timestamp}}"
group="${RJOB_GROUP:-gui_agent}"
charged_group="${RJOB_CHARGED_GROUP:-$group}"
cpu="${RJOB_CPU:-8}"
gpu="${RJOB_GPU:-0}"
memory="${RJOB_MEMORY:-32000}"
max_wait_duration="${RJOB_MAX_WAIT_DURATION:-12h0m0s}"
positive_tags="${RJOB_POSITIVE_TAGS:-feature/gpfs=yes}"
state_root="${STATE_ROOT:-/data/codes/gui_grounding/data/logs/0528_mainline}"
log_root="${LOG_ROOT:-${state_root}/cpu_monitor_${timestamp}}"
submit_log="${LOG_PATH:-${log_root}/${rjob_name}.submit.log}"
worker_log="${WORKER_LOG:-${log_root}/${rjob_name}.worker.log}"
worker_script="${WORKER_SCRIPT:-${repo_root}/tools/experiments/0528_mrpd_mainline/run_cpu_metrics_monitor_worker.sh}"
mount_arg="--mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot"

mkdir -p "$log_root" "$state_root"
chmod +x "$worker_script"

{
  echo "[0528-cpu-monitor-launch] timestamp=${timestamp}"
  echo "[0528-cpu-monitor-launch] rjob_name=${rjob_name}"
  echo "[0528-cpu-monitor-launch] group=${group} charged_group=${charged_group}"
  echo "[0528-cpu-monitor-launch] cpu=${cpu} gpu=${gpu} memory=${memory}"
  echo "[0528-cpu-monitor-launch] state_root=${state_root}"
  echo "[0528-cpu-monitor-launch] worker_log=${worker_log}"

  brainctl launch \
    --cpu "$cpu" \
    --gpu "$gpu" \
    --memory "$memory" \
    --group "$group" \
    --charged-group="$charged_group" \
    --private-machine=group \
    "$mount_arg" \
    --positive-tags "$positive_tags" \
    --predict-only

  brainctl launch \
    -d \
    --name "$rjob_name" \
    --replica-restart=never \
    --backoff-limit 1 \
    --max-wait-duration "$max_wait_duration" \
    --cpu "$cpu" \
    --gpu "$gpu" \
    --memory="$memory" \
    --group "$group" \
    --charged-group="$charged_group" \
    --private-machine=group \
    "$mount_arg" \
    --positive-tags "$positive_tags" \
    --set-env "REPO_ROOT=$repo_root" \
    --set-env "GUI_SD_ENV_ROOT=${GUI_SD_ENV_ROOT:-/data/codes/gui_grounding/conda_envs/GUI-SD}" \
    --set-env "STATE_ROOT=$state_root" \
    --set-env "INTERVAL_SEC=${INTERVAL_SEC:-120}" \
    --set-env "MAX_TICKS=${MAX_TICKS:-240}" \
    --set-env "WORKER_LOG=$worker_log" \
    -- \
    bash -lc 'bash "$REPO_ROOT/tools/experiments/0528_mrpd_mainline/run_cpu_metrics_monitor_worker.sh"'
} 2>&1 | tee "$submit_log"

echo "[0528-cpu-monitor-launch] submitted"
echo "[0528-cpu-monitor-launch] submit_log=${submit_log}"
echo "[0528-cpu-monitor-launch] worker_log=${worker_log}"
