#!/usr/bin/env bash
# Launch failed-only / failed+ambiguous / all-samples MRPD scope ablations.
#
# Each training launcher must stay in the foreground of its own tmux session:
# the underlying rjob launcher traps HUP and stops the remote rjob when its
# parent shell disappears.  Do not background multiple launchers from one tmux.
set -euo pipefail

repo_root="${REPO_ROOT:-/data/codes/gui_grounding/GUI-SD-code-main}"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%m%d%H%M)}"
group="${RJOB_GROUP:-gui_agent}"
charged_group="${RJOB_CHARGED_GROUP:-$group}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs/0528_mainline}"
state_dir="${STATE_DIR:-${log_root}/distill_scope_${timestamp}}"
requested_scopes=",${SCOPES:-failed,famb,all},"
mkdir -p "$state_dir" /data/codes/gui_grounding/data/logs/training

launcher="${repo_root}/tools/training/launch_qwen3_4b_rsfull_grpo_sdpo_2node_experiment_rjob.sh"
worker="${repo_root}/tools/training/run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh"
source_ckpt="${SOURCE_CKPT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sft/gui-sd-qwen3-4b-rrsisd_sft_e1-20260526-220915/v0-20260526-220853/checkpoint-96}"

submit_log="${state_dir}/submit.log"
: > "$submit_log"

echo "[scope-ablation] timestamp=$timestamp" | tee -a "$submit_log"
echo "[scope-ablation] group=$group charged_group=$charged_group" | tee -a "$submit_log"
echo "[scope-ablation] source_ckpt=$source_ckpt" | tee -a "$submit_log"
echo "[scope-ablation] scopes=${requested_scopes#,}" | tee -a "$submit_log"

want_scope() {
  case "$requested_scopes" in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

launch_one() {
  local scope="$1"
  local short="$2"
  local local_group="${3:-$group}"
  local local_charged="${4:-$charged_group}"
  local run_ts="${timestamp}${short}"
  local run_name="gui-sd-qwen3-4b-rrsisd_sft_mrpd_scope_${short}_gaussian_e1-${run_ts}"
  local rjob_name="m28-${short}-${run_ts}"
  local tmux_name="mrpd0528_${short}_${run_ts}"
  local log_path="/data/codes/gui_grounding/data/logs/training/${rjob_name}.rjob.log"
  local env_file="${state_dir}/${short}.env"
  local tmux_log="${state_dir}/${short}.tmux.log"

  if [ "${#rjob_name}" -ge 50 ]; then
    echo "ERROR: generated rjob_name too long: ${rjob_name}" | tee -a "$submit_log"
    exit 1
  fi
  if tmux has-session -t "$tmux_name" 2>/dev/null; then
    echo "[scope-ablation] skip existing tmux=${tmux_name}" | tee -a "$submit_log"
    return 0
  fi

  cat > "$env_file" <<ENV
export RUN_TIMESTAMP="$run_ts"
export EXPERIMENT_SLUG="mrpd-scope-${short}"
export RUN_NAME="$run_name"
export RJOB_NAME="$rjob_name"
export RJOB_GROUP="$local_group"
export RJOB_CHARGED_GROUP="$local_charged"
export RJOB_CPU="${RJOB_CPU:-64}"
export RJOB_GPU="${RJOB_GPU:-8}"
export RJOB_MEMORY="${RJOB_MEMORY:-760000}"
export RJOB_REPLICA="${RJOB_REPLICA:-2}"
export RJOB_MAX_WAIT_DURATION="${RJOB_MAX_WAIT_DURATION:-18h0m0s}"
export WORKER_SCRIPT="$worker"
export RJOB_LOG_PATH="$log_path"
export BASE_MODEL_PATH="$source_ckpt"
export MODEL_PATH="$source_ckpt"
export TEACHER_PATH="$source_ckpt"
export ROLLOUT_MODEL_PATH="$source_ckpt"
export CKPT_ROOT="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd}"
export TRAIN_JSONL="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}"
export TEST_JSONL="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-7}"
export MASTER_PORT="${MASTER_PORT:-30120}"
export VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8892}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export SAVE_STEPS="${SAVE_STEPS:-10}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export LR="${LR:-1e-6}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.01}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
export MAX_LENGTH="${MAX_LENGTH:-20000}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-64}"
export SDPO_LAMBDA="${SDPO_LAMBDA:-0.25}"
export SDPO_DISTILL_SCOPE="$scope"
export SDPO_TARGET="${SDPO_TARGET:-rollout}"
export SDPO_TEACHER_REFRESH_MODE="${SDPO_TEACHER_REFRESH_MODE:-metric}"
export SDPO_TEACHER_REFRESH_STEP="${SDPO_TEACHER_REFRESH_STEP:--1}"
export SDPO_TEACHER_REFRESH_WARMUP="${SDPO_TEACHER_REFRESH_WARMUP:-80}"
export SDPO_TEACHER_REFRESH_WINDOW="${SDPO_TEACHER_REFRESH_WINDOW:-50}"
export SDPO_TEACHER_REFRESH_CHECK_INTERVAL="${SDPO_TEACHER_REFRESH_CHECK_INTERVAL:-10}"
export SDPO_TEACHER_REFRESH_MAX_REFRESHES="${SDPO_TEACHER_REFRESH_MAX_REFRESHES:-1}"
export OPSD_MASK_MODE="${OPSD_MASK_MODE:-gaussian}"
export OPSD_HINT_MODE="${OPSD_HINT_MODE:-hint}"
export OPSD_TOKEN_WEIGHT_MODE="${OPSD_TOKEN_WEIGHT_MODE:-uniform-entropy}"
export OPSD_NON_DIGIT_WEIGHT="${OPSD_NON_DIGIT_WEIGHT:-0.05}"
export OPSD_MAX_DIGIT_LEN="${OPSD_MAX_DIGIT_LEN:-3}"
export OPSD_EMA_DECAY="${OPSD_EMA_DECAY:-0.0}"
export GRPO_BETA="${GRPO_BETA:-0.04}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-50}"
export VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.82}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-20000}"
export SKIP_EVAL=true
export STAGE2_SKIP_EVAL=true
export COMPLETION_GREP_PATTERN="${COMPLETION_GREP_PATTERN:-\\[4b-grpo-sdpo-2node-worker\\].*run completed}"
export COMPLETION_GRACE_SEC="${COMPLETION_GRACE_SEC:-60}"
ENV

  echo "[scope-ablation] launch scope=${scope} rjob=${rjob_name} tmux=${tmux_name} group=${local_group}" | tee -a "$submit_log"
  tmux new-session -d -s "$tmux_name" "cd '$repo_root' && set -a && source '$env_file' && set +a && bash '$launcher' 2>&1 | tee '$tmux_log'"
  echo "${rjob_name}|${scope}|${local_group}|${log_path}|${tmux_name}|${run_name}" >> "${state_dir}/rjobs.tsv"
}

[ -f "${state_dir}/rjobs.tsv" ] || : > "${state_dir}/rjobs.tsv"

if want_scope failed; then
  launch_one failed failed "$group" "$charged_group"
fi
if want_scope famb; then
  launch_one failed_ambiguous famb "$group" "$charged_group"
fi
if want_scope all; then
  launch_one all all "${RJOB_GROUP_3:-$group}" "${RJOB_CHARGED_GROUP_3:-$charged_group}"
fi

cat > "${state_dir}/status.json" <<JSON
{
  "status": "submitted",
  "timestamp": "${timestamp}",
  "state_dir": "${state_dir}",
  "rjobs_tsv": "${state_dir}/rjobs.tsv"
}
JSON

echo "[scope-ablation] state_dir=${state_dir}" | tee -a "$submit_log"
