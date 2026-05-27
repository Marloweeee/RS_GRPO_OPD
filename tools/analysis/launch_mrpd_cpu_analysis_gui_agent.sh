#!/usr/bin/env bash
# Launch CPU-only MRPD statistical analysis on gui_agent.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
rjob_name="${RJOB_NAME:-mrpd-cpu-analysis-${timestamp}}"
group="${RJOB_GROUP:-gui_agent}"
charged_group="${RJOB_CHARGED_GROUP:-$group}"
cpu="${RJOB_CPU:-16}"
gpu="${RJOB_GPU:-0}"
memory="${RJOB_MEMORY:-120000}"
positive_tags="${RJOB_POSITIVE_TAGS:-feature/gpfs=yes}"
max_wait_duration="${RJOB_MAX_WAIT_DURATION:-1h0m0s}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs}"
submit_log="${SUBMIT_LOG:-${log_root}/eval/${rjob_name}.submit.log}"
worker_log="${WORKER_LOG:-${log_root}/eval/${rjob_name}.worker.log}"
out_dir="${OUT_DIR:-${repo_root}/MRPD/cpu_analysis/${timestamp}}"
mount_arg="--mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot"

mkdir -p "$(dirname "$submit_log")" "$(dirname "$worker_log")" "$out_dir"
chmod +x "${repo_root}/tools/analysis/mrpd_cpu_analysis.py"

{
    echo "[mrpd-cpu-analysis] timestamp=${timestamp}"
    echo "[mrpd-cpu-analysis] rjob_name=${rjob_name}"
    echo "[mrpd-cpu-analysis] group=${group} cpu=${cpu} gpu=${gpu} memory=${memory}"
    echo "[mrpd-cpu-analysis] out_dir=${out_dir}"
    echo "[mrpd-cpu-analysis] worker_log=${worker_log}"

    brainctl rjob launch \
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
        --set-env "OUT_DIR=$out_dir" \
        --set-env "WORKER_LOG=$worker_log" \
        -- bash -lc '
            set -euo pipefail
            cd "$REPO_ROOT"
            mkdir -p "$(dirname "$WORKER_LOG")" "$OUT_DIR"
            {
                echo "[mrpd-cpu-analysis-worker] start=$(date "+%F %T %Z")"
                echo "[mrpd-cpu-analysis-worker] host=$(hostname)"
                python3 tools/analysis/mrpd_cpu_analysis.py --out-dir "$OUT_DIR" --bootstrap-iters 5000
                echo "[mrpd-cpu-analysis-worker] end=$(date "+%F %T %Z")"
            } > "$WORKER_LOG" 2>&1
        '
} 2>&1 | tee "$submit_log"

echo "[mrpd-cpu-analysis] submitted"
echo "[mrpd-cpu-analysis] worker_log=${worker_log}"
echo "[mrpd-cpu-analysis] report=${out_dir}/mrpd_cpu_analysis.html"
