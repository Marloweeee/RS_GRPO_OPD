#!/usr/bin/env bash
# Monitor current MRPD/SFT initialized training rjobs for up to two hours.
# Once a final checkpoint is available, stop the training rjob/tmux session and
# launch the matching rs_test.jsonl evaluation as a detached rjob.
set -u -o pipefail

repo_root="${REPO_ROOT:-/data/codes/gui_grounding/GUI-SD-code-main}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs}"
monitor_log="${MONITOR_LOG:-${log_root}/training/monitor_current_training_and_eval-$(date +%Y%m%d-%H%M%S).log}"
state_dir="${STATE_DIR:-${log_root}/training/monitor_current_training_and_eval.state}"
poll_sec="${POLL_SEC:-60}"
max_wait_sec="${MAX_WAIT_SEC:-7200}"
eval_jsonl="${EVAL_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"

mkdir -p "$(dirname "$monitor_log")" "$state_dir" "${log_root}/eval"

log() {
    echo "[monitor-current][$(date '+%F %T')] $*" | tee -a "$monitor_log"
}

rjob_status() {
    local rjob="$1"
    brainctl -n shai-core get "rjob/${rjob}" 2>/dev/null \
        | awk 'NR==2 {print $2}'
}

latest_ckpt_from_log() {
    local train_log="$1"
    [ -f "$train_log" ] || return 0
    grep -Eo '(latest_ckpt=|last_model_checkpoint: |Saving model checkpoint to )/[^[:space:]]+/checkpoint-[0-9]+' "$train_log" 2>/dev/null \
        | sed -E 's/^(latest_ckpt=|last_model_checkpoint: |Saving model checkpoint to )//' \
        | tail -1
}

terminal_ckpt_from_log() {
    local train_log="$1"
    [ -f "$train_log" ] || return 0
    grep -Eo '(latest_ckpt=|last_model_checkpoint: )/[^[:space:]]+/checkpoint-[0-9]+' "$train_log" 2>/dev/null \
        | sed -E 's/^(latest_ckpt=|last_model_checkpoint: )//' \
        | tail -1
}

latest_ckpt_from_root() {
    local run_name="$1"
    local ckpt_root="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"
    local root="${ckpt_root}/${run_name}"
    [ -d "$root" ] || return 0
    find "$root" -maxdepth 2 -type d -name 'checkpoint-*' -printf '%f %p\n' 2>/dev/null \
        | awk '{step=$1; sub(/^checkpoint-/, "", step); if (step ~ /^[0-9]+$/) print step " " $2}' \
        | sort -n \
        | tail -1 \
        | cut -d' ' -f2-
}

ckpt_step() {
    local ckpt="$1"
    basename "$ckpt" | sed -n 's/^checkpoint-\([0-9][0-9]*\)$/\1/p'
}

completed_count() {
    local train_log="$1"
    [ -f "$train_log" ] || {
        echo 0
        return
    }
    grep -E '\[4b-[^]]*worker\].*completed|run completed|two-stage run completed' "$train_log" 2>/dev/null \
        | wc -l \
        | tr -d ' '
}

stop_training_job() {
    local key="$1"
    local rjob="$2"
    local tmux_session="$3"
    local marker="${state_dir}/${key}.training_stopped"

    [ -f "$marker" ] && return 0

    local status
    status="$(rjob_status "$rjob" || true)"
    log "${key}: stopping training resources; rjob=${rjob} status=${status:-unknown} tmux=${tmux_session}"
    if [ "${status:-}" = "Running" ]; then
        brainctl -n shai-core stop "rjob/${rjob}" >>"$monitor_log" 2>&1 || \
            brainctl -n shai-core delete "rjob/${rjob}" --ignore-not-found >>"$monitor_log" 2>&1 || true
    fi
    if tmux has-session -t "$tmux_session" 2>/dev/null; then
        tmux kill-session -t "$tmux_session" >>"$monitor_log" 2>&1 || true
    fi
    date '+%F %T' > "$marker"
}

launch_eval_once() {
    local key="$1"
    local ckpt="$2"
    local eval_out="$3"
    local eval_rjob="$4"
    local eval_group="$5"
    local marker="${state_dir}/${key}.eval_launched"
    local summary="${eval_out}/summary.json"

    if [ -f "$summary" ]; then
        log "${key}: eval summary already exists: ${summary}"
        return 0
    fi
    if [ -f "$marker" ]; then
        log "${key}: eval already launched; waiting for ${summary}"
        return 0
    fi

    mkdir -p "$eval_out"
    log "${key}: launching eval rjob=${eval_rjob} group=${eval_group} ckpt=${ckpt} out=${eval_out}"
    (
        cd "$repo_root" || exit 1
        RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)-${key}" \
        RJOB_GROUP="$eval_group" \
        RJOB_CHARGED_GROUP="$eval_group" \
        RJOB_CPU="${EVAL_RJOB_CPU:-28}" \
        RJOB_GPU="${EVAL_RJOB_GPU:-8}" \
        RJOB_MEMORY="${EVAL_RJOB_MEMORY:-600000}" \
        RJOB_MAX_WAIT_DURATION="${EVAL_RJOB_MAX_WAIT_DURATION:-2h0m0s}" \
        RJOB_NAME="$eval_rjob" \
        MODEL="$ckpt" \
        EVAL_JSONL="$eval_jsonl" \
        OUT_DIR="$eval_out" \
        LOG_ROOT="${log_root}/eval" \
        LOG_PATH="${log_root}/eval/${eval_rjob}.submit.log" \
        WORKER_LOG="${log_root}/eval/${eval_rjob}.worker.log" \
        TP="${EVAL_TP:-8}" \
        MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-12000}" \
        GPU_MEM_UTIL="${EVAL_GPU_MEM_UTIL:-0.82}" \
        EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}" \
        MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}" \
        NUM_ROLLOUTS="${EVAL_NUM_ROLLOUTS:-1}" \
        ROLLOUT_TEMPERATURE="${EVAL_ROLLOUT_TEMPERATURE:-0.0}" \
        TOP_P="${EVAL_TOP_P:-0.95}" \
        bash "${repo_root}/tools/evaluation/launch_single_ckpt_split_eval_detached_rjob.sh"
    ) >>"$monitor_log" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        {
            echo "launched_at=$(date '+%F %T')"
            echo "ckpt=${ckpt}"
            echo "eval_out=${eval_out}"
            echo "eval_rjob=${eval_rjob}"
        } > "$marker"
        log "${key}: eval submitted"
    else
        log "${key}: eval submit failed rc=${rc}; will retry next poll"
    fi
}

maybe_stop_eval_if_done() {
    local key="$1"
    local eval_rjob="$2"
    local eval_out="$3"
    local marker="${state_dir}/${key}.eval_stopped"
    [ -f "$marker" ] && return 0
    [ -f "${eval_out}/summary.json" ] || return 0

    local status
    status="$(rjob_status "$eval_rjob" || true)"
    log "${key}: eval summary ready; eval_rjob=${eval_rjob} status=${status:-unknown}"
    if [ "${status:-}" = "Running" ]; then
        brainctl -n shai-core stop "rjob/${eval_rjob}" >>"$monitor_log" 2>&1 || true
    fi
    date '+%F %T' > "$marker"
}

process_job() {
    local key="$1"
    local rjob="$2"
    local tmux_session="$3"
    local train_log="$4"
    local run_name="$5"
    local eval_out="$6"
    local eval_rjob="$7"
    local eval_group="$8"

    maybe_stop_eval_if_done "$key" "$eval_rjob" "$eval_out"

    if [ -f "${eval_out}/summary.json" ]; then
        return 0
    fi

    local ckpt terminal_ckpt status done_count step
    ckpt="$(latest_ckpt_from_log "$train_log" || true)"
    terminal_ckpt="$(terminal_ckpt_from_log "$train_log" || true)"
    if [ -z "${ckpt:-}" ]; then
        ckpt="$(latest_ckpt_from_root "$run_name" || true)"
    fi
    status="$(rjob_status "$rjob" || true)"
    done_count="$(completed_count "$train_log")"
    step="$(ckpt_step "${ckpt:-}" || true)"

    log "${key}: status=${status:-unknown} completed=${done_count} step=${step:-none} terminal_ckpt=${terminal_ckpt:-none} ckpt=${ckpt:-none}"

    if [ -n "${ckpt:-}" ] && {
        [ "${status:-}" = "Succeeded" ] || [ "${done_count:-0}" -ge 2 ] || [ -n "${terminal_ckpt:-}" ];
    }; then
        stop_training_job "$key" "$rjob" "$tmux_session"
        launch_eval_once "$key" "$ckpt" "$eval_out" "$eval_rjob" "$eval_group"
    fi
}

jobs=(
  "no_hint|q4b-sftopsd-nohint-e1-20260527-093400-aos2n|mrpd_sft_nohint_p0_aos2n|${log_root}/training/q4b-sftopsd-nohint-e1-20260527-093400-aos2n.rjob.log|gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_metric_no_hint_e1-20260527-093400-aos2n|${log_root}/eval/q4b-sft-grpo-opsd-no-hint-20260527-093400-aos2n-test|eval-nohint-0527|aos"
  "soft_window|q4b-sftopsd-soft-window-e1-20260527-100136|mrpd_sft_opsd_soft_window_retry|${log_root}/training/q4b-sftopsd-soft-window-e1-20260527-100136.rjob.log|gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_metric_soft_window_e1-20260527-100136|${log_root}/eval/q4b-sft-grpo-opsd-soft-window-20260527-100136-test|eval-softwin-0527|gui_agent"
  "ema098|q4b-sftopsd-ema098-gauss-e1-20260527-113500-ema098|mrpd_sft_refresh_ema098_gauss|${log_root}/training/q4b-sftopsd-ema098-gauss-e1-20260527-113500-ema098.rjob.log|gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_ema098_gaussian_e1-20260527-113500-ema098|${log_root}/eval/gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_ema098_gaussian_e1-20260527-113500-ema098_test|eval-ema098-0527|gui_agent"
  "fixed_s80|q4b-sftopsd-fixed-s80-gauss-e1-20260527-113500-s80|mrpd_sft_refresh_fixed_s80_gauss|${log_root}/training/q4b-sftopsd-fixed-s80-gauss-e1-20260527-113500-s80.rjob.log|gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_fixed_s80_gaussian_e1-20260527-113500-s80|${log_root}/eval/gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_fixed_s80_gaussian_e1-20260527-113500-s80_test|eval-fixed80-0527|gui_agent"
  "no_refresh|q4b-noref-gauss-05271142|mrpd_sft_refresh_no_refresh_gauss|${log_root}/training/q4b-noref-gauss-05271142.rjob.log|gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_no_refresh_gaussian_e1-20260527-114200-norefresh|${log_root}/eval/gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_no_refresh_gaussian_e1-20260527-114200-norefresh_test|eval-noref-0527|gui_agent"
)

deadline=$(( $(date +%s) + max_wait_sec ))
log "started monitor_log=${monitor_log}"
log "state_dir=${state_dir} poll_sec=${poll_sec} max_wait_sec=${max_wait_sec}"

while [ "$(date +%s)" -le "$deadline" ]; do
    remaining=0
    for entry in "${jobs[@]}"; do
        IFS='|' read -r key rjob tmux_session train_log run_name eval_out eval_rjob eval_group <<< "$entry"
        process_job "$key" "$rjob" "$tmux_session" "$train_log" "$run_name" "$eval_out" "$eval_rjob" "$eval_group"
        if [ ! -f "${eval_out}/summary.json" ]; then
            remaining=$((remaining + 1))
        fi
    done
    if [ "$remaining" -eq 0 ]; then
        log "all eval summaries are ready; monitor exiting"
        exit 0
    fi
    log "remaining_without_summary=${remaining}; sleeping ${poll_sec}s"
    sleep "$poll_sec"
done

log "deadline reached; monitor exiting"
exit 0
