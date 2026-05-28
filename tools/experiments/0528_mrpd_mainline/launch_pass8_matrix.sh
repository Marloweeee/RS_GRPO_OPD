#!/usr/bin/env bash
# Launch the pass@8 gap-reduction evaluation matrix.
set -euo pipefail

repo_root="${REPO_ROOT:-/data/codes/gui_grounding/GUI-SD-code-main}"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
group="${RJOB_GROUP:-gui_agent}"
charged_group="${RJOB_CHARGED_GROUP:-$group}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs/0528_mainline}"
eval_root="${EVAL_ROOT:-/data/codes/gui_grounding/data/logs/eval/0528_mrpd_mainline/pass8_${timestamp}}"
state_dir="${STATE_DIR:-${log_root}/pass8_${timestamp}}"
mkdir -p "$eval_root" "$state_dir" "$log_root"

launcher="${repo_root}/tools/evaluation/launch_teacher_hint_pass8_detached_rjob.sh"

SFT_CKPT="${SFT_CKPT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sft/gui-sd-qwen3-4b-rrsisd_sft_e1-20260526-220915/v0-20260526-220853/checkpoint-96}"
GRPO_CKPT="${GRPO_CKPT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rrsisd_sft_grpo_only_e1-20260527-082554/v0-20260527-083221/checkpoint-870}"
MRPD_METRIC_CKPT="${MRPD_METRIC_CKPT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_metric_gaussian_e1-20260526-225857/v0-20260526-230454/checkpoint-870}"
MRPD_EMA_CKPT="${MRPD_EMA_CKPT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-qwen3-4b-rrsisd_sft_grpo_opsd_ema098_gaussian_e1-20260527-113500-ema098/v0-20260527-113657/checkpoint-870}"

declare -A MODELS=(
  [sft]="$SFT_CKPT"
  [grpo_only]="$GRPO_CKPT"
  [mrpd_metric]="$MRPD_METRIC_CKPT"
  [mrpd_ema098]="$MRPD_EMA_CKPT"
)

submit_log="${state_dir}/submit.log"
: > "$submit_log"

echo "[pass8-matrix] timestamp=$timestamp" | tee -a "$submit_log"
echo "[pass8-matrix] group=$group charged_group=$charged_group" | tee -a "$submit_log"
echo "[pass8-matrix] eval_root=$eval_root" | tee -a "$submit_log"

for model_id in sft grpo_only mrpd_metric mrpd_ema098; do
  for hint_mode in none gaussian; do
    out_dir="${eval_root}/${model_id}_${hint_mode}"
    rjob_name="p8-${model_id}-${hint_mode}-${timestamp}"
    echo "[pass8-matrix] launch model=${model_id} hint=${hint_mode} rjob=${rjob_name}" | tee -a "$submit_log"
    RUN_TIMESTAMP="$timestamp" \
      RJOB_NAME="$rjob_name" \
      RJOB_GROUP="$group" \
      RJOB_CHARGED_GROUP="$charged_group" \
      RJOB_CPU="${RJOB_CPU:-28}" \
      RJOB_GPU="${RJOB_GPU:-8}" \
      RJOB_MEMORY="${RJOB_MEMORY:-600000}" \
      RJOB_MAX_WAIT_DURATION="${RJOB_MAX_WAIT_DURATION:-2h0m0s}" \
      MODEL="${MODELS[$model_id]}" \
      EVAL_JSONL="${EVAL_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}" \
      OUT_DIR="$out_dir" \
      HINT_MODE="$hint_mode" \
      TP="${TP:-8}" \
      NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}" \
      ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}" \
      TOP_P="${TOP_P:-0.95}" \
      EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}" \
      LOG_PATH="${state_dir}/${rjob_name}.submit.log" \
      WORKER_LOG="${state_dir}/${rjob_name}.worker.log" \
      bash "$launcher" >> "$submit_log" 2>&1 || {
        echo "[pass8-matrix] launch failed model=${model_id} hint=${hint_mode}" | tee -a "$submit_log"
      }
  done
done

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

echo "[pass8-matrix] manifest=${state_dir}/manifest.json" | tee -a "$submit_log"
