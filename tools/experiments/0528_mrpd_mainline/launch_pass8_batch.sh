#!/usr/bin/env bash
# Submit a bounded batch of pass@8 jobs, recording the remaining queue on disk.
set -euo pipefail

repo_root="${REPO_ROOT:-/data/codes/gui_grounding/GUI-SD-code-main}"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-0528p8}"
group="${RJOB_GROUP:-aos}"
charged_group="${RJOB_CHARGED_GROUP:-$group}"
max_jobs="${MAX_JOBS:-3}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs/0528_mainline}"
eval_root="${EVAL_ROOT:-/data/codes/gui_grounding/data/logs/eval/0528_mrpd_mainline/pass8_${timestamp}}"
state_dir="${STATE_DIR:-${log_root}/pass8_${timestamp}}"
mkdir -p "$eval_root" "$state_dir"

launcher="${repo_root}/tools/evaluation/launch_teacher_hint_pass8_detached_rjob.sh"
submit_log="${state_dir}/submit_batch.log"
jobs_tsv="${state_dir}/jobs.tsv"

SFT_CKPT="${SFT_CKPT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sft/gui-sd-qwen3-4b-rrsisd_sft_e1-20260526-220915/v0-20260526-220853/checkpoint-96}"
GRPO_CKPT="${GRPO_CKPT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rrsisd_sft_grpo_only_e1-20260527-082554/v0-20260527-083221/checkpoint-870}"
MRPD_METRIC_CKPT="${MRPD_METRIC_CKPT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_metric_gaussian_e1-20260526-225857/v0-20260526-230454/checkpoint-870}"
MRPD_EMA_CKPT="${MRPD_EMA_CKPT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_ema098_gaussian_e1-20260527-113500-ema098/v0-20260527-113657/checkpoint-870}"

cat > "${state_dir}/manifest.json" <<JSON
{
  "timestamp": "${timestamp}",
  "eval_root": "${eval_root}",
  "gap_baseline_model": "sft",
  "models": [
    {
      "id": "sft",
      "name": "SFT",
      "greedy_summary": "/data/codes/gui_grounding/data/logs/eval/qwen3-4b-rrsisd-sft-20260526-220915-checkpoint-96-test-greedy/summary.json",
      "pass8": [
        {"hint_mode": "none", "summary": "${eval_root}/sft_none/summary.json"},
        {"hint_mode": "gaussian", "summary": "${eval_root}/sft_gaussian/summary.json"}
      ]
    },
    {
      "id": "grpo_only",
      "name": "SFT -> GRPO-only",
      "greedy_summary": "/data/codes/gui_grounding/data/logs/eval/q4b-sft-grpo-only-20260527-082554-test/summary.json",
      "pass8": [
        {"hint_mode": "none", "summary": "${eval_root}/grpo_only_none/summary.json"},
        {"hint_mode": "gaussian", "summary": "${eval_root}/grpo_only_gaussian/summary.json"}
      ]
    },
    {
      "id": "mrpd_metric",
      "name": "SFT -> MRPD metric refresh",
      "greedy_summary": "/data/codes/gui_grounding/data/logs/eval/gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_metric_gaussian_e1-20260526-225857_test/summary.json",
      "pass8": [
        {"hint_mode": "none", "summary": "${eval_root}/mrpd_metric_none/summary.json"},
        {"hint_mode": "gaussian", "summary": "${eval_root}/mrpd_metric_gaussian/summary.json"}
      ]
    },
    {
      "id": "mrpd_ema098",
      "name": "SFT -> MRPD EMA0.98",
      "greedy_summary": "/data/codes/gui_grounding/data/logs/eval/gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_ema098_gaussian_e1-20260527-113500-ema098_test/summary.json",
      "pass8": [
        {"hint_mode": "none", "summary": "${eval_root}/mrpd_ema098_none/summary.json"},
        {"hint_mode": "gaussian", "summary": "${eval_root}/mrpd_ema098_gaussian/summary.json"}
      ]
    }
  ]
}
JSON

if [ ! -f "$jobs_tsv" ]; then
  cat > "$jobs_tsv" <<EOF
sft|none|${SFT_CKPT}|${eval_root}/sft_none
sft|gaussian|${SFT_CKPT}|${eval_root}/sft_gaussian
grpo_only|none|${GRPO_CKPT}|${eval_root}/grpo_only_none
grpo_only|gaussian|${GRPO_CKPT}|${eval_root}/grpo_only_gaussian
mrpd_metric|none|${MRPD_METRIC_CKPT}|${eval_root}/mrpd_metric_none
mrpd_metric|gaussian|${MRPD_METRIC_CKPT}|${eval_root}/mrpd_metric_gaussian
mrpd_ema098|none|${MRPD_EMA_CKPT}|${eval_root}/mrpd_ema098_none
mrpd_ema098|gaussian|${MRPD_EMA_CKPT}|${eval_root}/mrpd_ema098_gaussian
EOF
fi

: > "$submit_log"
echo "[pass8-batch] timestamp=${timestamp} group=${group} max_jobs=${max_jobs}" | tee -a "$submit_log"
submitted=0
remaining_tmp="${state_dir}/jobs.remaining.$$"
: > "$remaining_tmp"

while IFS='|' read -r model_id hint_mode model out_dir; do
  [ -n "${model_id:-}" ] || continue
  retry_suffix=""
  if [ -f "${out_dir}/summary.json" ]; then
    echo "[pass8-batch] skip done ${model_id}/${hint_mode}" | tee -a "$submit_log"
    continue
  fi
  if [ "$submitted" -ge "$max_jobs" ]; then
    echo "${model_id}|${hint_mode}|${model}|${out_dir}" >> "$remaining_tmp"
    continue
  fi
  rjob_name="p8-${model_id}-${hint_mode}-${timestamp}"
  rjob_name="${rjob_name//_/-}"
  if [ "${#rjob_name}" -ge 50 ]; then
    rjob_name="p8-${model_id:0:4}-${hint_mode:0:4}-${timestamp}"
    rjob_name="${rjob_name//_/-}"
  fi
  rjob_status="$(brainctl -n shai-core get "rjob/${rjob_name}" 2>/dev/null | awk 'NR==2 {print $2}' || true)"
  if [ -n "$rjob_status" ] && [ "$rjob_status" != "Running" ] && [ "$rjob_status" != "Pending" ] && [ "$rjob_status" != "Starting" ]; then
    retry_suffix="${RJOB_RETRY_SUFFIX:--r$(date +%H%M)}"
    rjob_name="${rjob_name}${retry_suffix}"
  fi
  tmux_name="mrpd0528_p8_${model_id}_${hint_mode}_${timestamp}"
  tmux_name="${tmux_name//[^A-Za-z0-9_-]/_}"
  if [ -n "${retry_suffix:-}" ]; then
    tmux_name="${tmux_name}${retry_suffix//[^A-Za-z0-9_-]/_}"
  fi
  env_file="${state_dir}/${rjob_name}.env"

  if [ "$rjob_status" = "Running" ] || [ "$rjob_status" = "Pending" ] || [ "$rjob_status" = "Starting" ] || tmux has-session -t "$tmux_name" 2>/dev/null; then
    echo "[pass8-batch] already submitted ${model_id}/${hint_mode} rjob=${rjob_name} tmux=${tmux_name}" | tee -a "$submit_log"
    echo "${rjob_name}|${model_id}|${hint_mode}|${group}|${out_dir}|${state_dir}/${rjob_name}.worker.log|${tmux_name}" >> "${state_dir}/submitted.tsv"
    continue
  fi

  echo "[pass8-batch] launch ${model_id}/${hint_mode} rjob=${rjob_name} tmux=${tmux_name}" | tee -a "$submit_log"
  cat > "$env_file" <<ENV
export RUN_TIMESTAMP="$timestamp"
export RJOB_NAME="$rjob_name"
export RJOB_GROUP="$group"
export RJOB_CHARGED_GROUP="$charged_group"
export RJOB_CPU="${RJOB_CPU:-28}"
export RJOB_GPU="${RJOB_GPU:-8}"
export RJOB_MEMORY="${RJOB_MEMORY:-600000}"
export RJOB_MAX_WAIT_DURATION="${RJOB_MAX_WAIT_DURATION:-3h0m0s}"
export MODEL="$model"
export EVAL_JSONL="${EVAL_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
export OUT_DIR="$out_dir"
export HINT_MODE="$hint_mode"
export TP="${TP:-8}"
export NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
export TOP_P="${TOP_P:-0.95}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.78}"
export LOG_PATH="${state_dir}/${rjob_name}.submit.log"
export WORKER_LOG="${state_dir}/${rjob_name}.worker.log"
ENV
  if tmux new-session -d -s "$tmux_name" "cd '$repo_root' && set -a && source '$env_file' && set +a && bash '$launcher' 2>&1 | tee '${state_dir}/${rjob_name}.tmux.log'"; then
    echo "${rjob_name}|${model_id}|${hint_mode}|${group}|${out_dir}|${state_dir}/${rjob_name}.worker.log|${tmux_name}" >> "${state_dir}/submitted.tsv"
    submitted=$((submitted + 1))
  else
    echo "[pass8-batch] tmux launch failed, keep remaining ${model_id}/${hint_mode}" | tee -a "$submit_log"
    echo "${model_id}|${hint_mode}|${model}|${out_dir}" >> "$remaining_tmp"
  fi
done < "$jobs_tsv"

mv "$remaining_tmp" "$jobs_tsv"

cat > "${state_dir}/batch_status.json" <<JSON
{
  "status": "submitted_batch",
  "timestamp": "${timestamp}",
  "submitted": ${submitted},
  "remaining_jobs_tsv": "${jobs_tsv}",
  "manifest": "${state_dir}/manifest.json",
  "updated_at": "$(date '+%F %T')"
}
JSON

echo "[pass8-batch] submitted=${submitted} remaining=$(wc -l < "$jobs_tsv" | tr -d ' ')" | tee -a "$submit_log"
