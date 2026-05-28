#!/usr/bin/env bash
# Local tmux watcher that submits remaining GPU eval jobs when resources free up.
set -euo pipefail

repo_root="${REPO_ROOT:-/data/codes/gui_grounding/GUI-SD-code-main}"
cd "$repo_root"

state_root="${STATE_ROOT:-/data/codes/gui_grounding/data/logs/0528_mainline}"
pass8_ts="${PASS8_TS:-0528p8}"
pass8_state="${state_root}/pass8_${pass8_ts}"
scope_eval_root="${SCOPE_EVAL_ROOT:-/data/codes/gui_grounding/data/logs/eval/0528_mrpd_mainline/scope_tests}"
scope_eval_state="${state_root}/scope_test_eval"
interval="${INTERVAL_SEC:-180}"
max_ticks="${MAX_TICKS:-240}"
watch_log="${WATCH_LOG:-${state_root}/watch_submit_${pass8_ts}_$(date +%Y%m%d-%H%M%S).log}"
namespace="${RJOB_NAMESPACE:-shai-core}"
mkdir -p "$state_root" "$pass8_state" "$scope_eval_root" "$scope_eval_state"

log() {
  echo "[$(date '+%F %T')][0528-watch] $*" | tee -a "$watch_log"
}

acceptance_complete() {
  local status_json="${state_root}/0528_acceptance_status.json"
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

predict_single_gpu_group() {
  local group="$1"
  brainctl launch \
    --cpu 28 \
    --gpu 8 \
    --memory 600000 \
    --group "$group" \
    --charged-group="$group" \
    --private-machine=group \
    --mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot \
    --positive-tags feature/gpfs=yes \
    --predict-only 2>/dev/null | grep -q "Node:"
}

submit_pass8_batch_if_possible() {
  local jobs_tsv="${pass8_state}/jobs.tsv"
  if [ -f "$jobs_tsv" ] && [ ! -s "$jobs_tsv" ]; then
    log "pass8 queue empty"
    return 0
  fi
  for group in aos gui_agent; do
    if predict_single_gpu_group "$group"; then
      log "submitting pass8 batch on group=${group}"
      RUN_TIMESTAMP="$pass8_ts" \
        RJOB_GROUP="$group" \
        RJOB_CHARGED_GROUP="$group" \
        MAX_JOBS="${PASS8_MAX_JOBS:-3}" \
        bash tools/experiments/0528_mrpd_mainline/launch_pass8_batch.sh >> "$watch_log" 2>&1 || true
      return 0
    fi
  done
  log "no single-node GPU slot for pass8 this tick"
}

scope_short_name() {
  case "$1" in
    failed_ambiguous) echo "famb" ;;
    failed) echo "failed" ;;
    all) echo "all" ;;
    *) echo "$1" | tr -cd 'A-Za-z0-9_-' | cut -c1-10 ;;
  esac
}

latest_checkpoint_for_run() {
  local run_name="$1"
  local log_path="$2"
  local run_root="/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/${run_name}"
  local ckpt=""
  if [ -d "$run_root" ]; then
    ckpt="$(find "$run_root" -mindepth 2 -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
  fi
  if [ -z "$ckpt" ] && [ -f "$log_path" ]; then
    ckpt="$(grep -oE '/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/[^[:space:],]+/v[^[:space:],]+/checkpoint-[0-9]+' "$log_path" 2>/dev/null | sort -V | tail -n 1 || true)"
  fi
  echo "$ckpt"
}

submit_scope_test_eval_if_ready() {
  local submitted_this_tick=0
  for tsv in "$state_root"/distill_scope_*/rjobs.tsv; do
    [ -f "$tsv" ] || continue
    while IFS='|' read -r rjob scope group log_path tmux_name run_name; do
      [ -n "${rjob:-}" ] || continue
      [ -n "${run_name:-}" ] || continue
      if [ ! -f "$log_path" ] || ! grep -q "run completed" "$log_path"; then
        continue
      fi

      local out_dir="${scope_eval_root}/${scope}_${rjob}_test"
      if [ -f "${out_dir}/summary.json" ]; then
        continue
      fi

      local short eval_rjob eval_status ckpt eval_group env_file eval_tmux
      short="$(scope_short_name "$scope")"
      eval_rjob="${rjob/#m28/e28}"
      if [ "$eval_rjob" = "$rjob" ]; then
        eval_rjob="e28-${short}-${pass8_ts}"
      fi
      eval_status="$(brainctl -n "$namespace" get "rjob/${eval_rjob}" 2>/dev/null | awk 'NR==2 {print $2}' || true)"
      if [ "$eval_status" = "Running" ] || [ "$eval_status" = "Pending" ] || [ "$eval_status" = "Starting" ]; then
        log "scope_eval already active scope=${scope} rjob=${eval_rjob} status=${eval_status}"
        continue
      fi
      if [ -n "$eval_status" ] && [ ! -f "${out_dir}/summary.json" ]; then
        eval_rjob="${eval_rjob}-r$(date +%H%M)"
      fi

      ckpt="$(latest_checkpoint_for_run "$run_name" "$log_path")"
      if [ -z "$ckpt" ]; then
        log "scope_eval wait ckpt scope=${scope} run_name=${run_name}"
        continue
      fi

      eval_group=""
      for candidate_group in aos gui_agent; do
        if predict_single_gpu_group "$candidate_group"; then
          eval_group="$candidate_group"
          break
        fi
      done
      if [ -z "$eval_group" ]; then
        log "scope_eval no GPU slot scope=${scope} ckpt=${ckpt}"
        continue
      fi

      env_file="${scope_eval_state}/${eval_rjob}.env"
      eval_tmux="mrpd0528_eval_${short}_${eval_rjob}"
      eval_tmux="${eval_tmux//[^A-Za-z0-9_-]/_}"
      if tmux has-session -t "$eval_tmux" 2>/dev/null; then
        log "scope_eval tmux already exists scope=${scope} tmux=${eval_tmux}"
        continue
      fi

      mkdir -p "$out_dir"
      cat > "$env_file" <<ENV
export RUN_TIMESTAMP="$pass8_ts"
export RJOB_NAME="$eval_rjob"
export RJOB_GROUP="$eval_group"
export RJOB_CHARGED_GROUP="$eval_group"
export RJOB_CPU="${SCOPE_EVAL_CPU:-28}"
export RJOB_GPU="${SCOPE_EVAL_GPU:-8}"
export RJOB_MEMORY="${SCOPE_EVAL_MEMORY:-600000}"
export RJOB_MAX_WAIT_DURATION="${SCOPE_EVAL_MAX_WAIT_DURATION:-2h0m0s}"
export MODEL="$ckpt"
export EVAL_JSONL="${SCOPE_EVAL_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
export OUT_DIR="$out_dir"
export TP="${SCOPE_EVAL_TP:-8}"
export EVAL_BATCH_SIZE="${SCOPE_EVAL_BATCH_SIZE:-512}"
export MAX_MODEL_LEN="${SCOPE_EVAL_MAX_MODEL_LEN:-12000}"
export GPU_MEM_UTIL="${SCOPE_EVAL_GPU_MEM_UTIL:-0.88}"
export NUM_ROLLOUTS="${SCOPE_EVAL_NUM_ROLLOUTS:-1}"
export ROLLOUT_TEMPERATURE="${SCOPE_EVAL_TEMPERATURE:-0.0}"
export TOP_P="${SCOPE_EVAL_TOP_P:-0.95}"
export LOG_PATH="${scope_eval_state}/${eval_rjob}.submit.log"
export WORKER_LOG="${scope_eval_state}/${eval_rjob}.worker.log"
ENV
      log "scope_eval launch scope=${scope} train_rjob=${rjob} eval_rjob=${eval_rjob} group=${eval_group} ckpt=${ckpt}"
      tmux new-session -d -s "$eval_tmux" "cd '$repo_root' && set -a && source '$env_file' && set +a && bash tools/evaluation/launch_single_ckpt_split_eval_detached_rjob.sh 2>&1 | tee '${scope_eval_state}/${eval_rjob}.tmux.log'"
      echo "${eval_rjob}|${scope}|${rjob}|${eval_group}|${ckpt}|${out_dir}|${scope_eval_state}/${eval_rjob}.worker.log|${eval_tmux}" >> "${scope_eval_state}/submitted.tsv"
      submitted_this_tick=1
      break
    done < "$tsv"
    [ "$submitted_this_tick" -eq 0 ] || break
  done
}

summarize_scope_jobs() {
  for tsv in "$state_root"/distill_scope_*/rjobs.tsv; do
    [ -f "$tsv" ] || continue
    while IFS='|' read -r rjob scope group log_path tmux_name run_name; do
      [ -n "${rjob:-}" ] || continue
      status="$(brainctl -n "$namespace" get "rjob/${rjob}" 2>/dev/null | awk 'NR==2 {print $2}' || true)"
      latest=""
      if [ -f "$log_path" ]; then
        latest="$( (grep -E "global_step/max_steps|Saving model checkpoint|last_model_checkpoint|run completed|Traceback|RuntimeError|ERROR|OutOfMemory|NCCL" "$log_path" 2>/dev/null || true) | tail -n 2 | tr '\n' ' ' | cut -c1-500 )"
      fi
      log "scope=${scope} rjob=${rjob} status=${status:-unknown} latest=${latest}"
    done < "$tsv"
  done
}

log "start state_root=${state_root} pass8_ts=${pass8_ts} interval=${interval}"

tick=0
while [ "$tick" -lt "$max_ticks" ]; do
  tick=$((tick + 1))
  log "tick=${tick}"
  summarize_scope_jobs
  submit_scope_test_eval_if_ready
  submit_pass8_batch_if_possible
  if acceptance_complete; then
    log "acceptance complete; watcher exiting to avoid idle monitoring"
    break
  fi
  sleep "$interval"
done

log "completed max_ticks=${max_ticks}"
