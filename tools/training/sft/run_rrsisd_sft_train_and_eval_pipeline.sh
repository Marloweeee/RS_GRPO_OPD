#!/usr/bin/env bash
# Run the RRSIS-D SFT experiment and submit follow-up evaluations.
#
# Sequence:
#   1. Two-node SFT on rs_full/rs_train.jsonl.
#   2. Greedy val+test evaluation for the final SFT checkpoint.
#   3. No-hint test best/pass@8 evaluation.
#   4. Gaussian and zoom_in teacher-hint test pass@8 evaluations.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
run_name="${RUN_NAME:-gui-sd-qwen3-4b-rrsisd_sft_e1-${timestamp}}"
ckpt_root="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sft}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs}"
eval_root="${EVAL_LOG_ROOT:-${log_root}/eval}"
train_log_root="${TRAIN_LOG_ROOT:-${log_root}/training}"

mkdir -p "$eval_root" "$train_log_root"

export RUN_TIMESTAMP="$timestamp"
export RUN_NAME="$run_name"
export RJOB_NAME="${RJOB_NAME:-q4b-rrsisd-sft-2n-${timestamp}}"
export RJOB_LOG_PATH="${RJOB_LOG_PATH:-${train_log_root}/${RJOB_NAME}.rjob.log}"
export CKPT_ROOT="$ckpt_root"

echo "[sft-pipeline] timestamp=${timestamp}"
echo "[sft-pipeline] run_name=${run_name}"
echo "[sft-pipeline] ckpt_root=${ckpt_root}"
echo "[sft-pipeline] training_log=${RJOB_LOG_PATH}"
echo "[sft-pipeline] start SFT training"

bash "${repo_root}/tools/training/sft/launch_qwen3_4b_rrsisd_sft_2node_aos.sh"

echo "[sft-pipeline] training launcher completed; locating latest checkpoint"
latest_ckpt="$(
    find "${ckpt_root}/${run_name}" -path '*/checkpoint-*' -type d -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -1 \
        | cut -d' ' -f2- || true
)"
if [ -z "$latest_ckpt" ] && [ -f "${RJOB_LOG_PATH:-}" ]; then
    latest_ckpt="$(
        grep -E 'latest_ckpt=|last_model_checkpoint: ' "$RJOB_LOG_PATH" 2>/dev/null \
            | sed -E 's/^.*latest_ckpt=//; s/^.*last_model_checkpoint: //' \
            | grep -E '/checkpoint-[0-9]+$' \
            | tail -1 || true
    )"
fi
if [ -z "$latest_ckpt" ]; then
    echo "ERROR: no checkpoint found under ${ckpt_root}/${run_name}" >&2
    echo "ERROR: also failed to parse checkpoint from training log: ${RJOB_LOG_PATH:-unset}" >&2
    exit 1
fi

ckpt_name="$(basename "$latest_ckpt")"
eval_slug="qwen3-4b-rrsisd-sft-${timestamp}-${ckpt_name}"
echo "[sft-pipeline] latest_ckpt=${latest_ckpt}"

echo "[sft-pipeline] submit greedy val+test eval"
MODEL="$latest_ckpt" \
RJOB_NAME="q4b-sft-greedy-vt-${timestamp}" \
RJOB_GROUP="${EVAL_RJOB_GROUP:-aos}" \
RJOB_CHARGED_GROUP="${EVAL_RJOB_CHARGED_GROUP:-${EVAL_RJOB_GROUP:-aos}}" \
RJOB_CPU="${EVAL_RJOB_CPU:-28}" \
RJOB_GPU="${EVAL_RJOB_GPU:-8}" \
RJOB_MEMORY="${EVAL_RJOB_MEMORY:-600000}" \
VAL_JSONL="${VAL_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_val.jsonl}" \
TEST_JSONL="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}" \
VAL_OUT_DIR="${eval_root}/${eval_slug}-val-greedy" \
TEST_OUT_DIR="${eval_root}/${eval_slug}-test-greedy" \
LOG_PATH="${eval_root}/q4b-sft-greedy-vt-${timestamp}.submit.log" \
WORKER_LOG="${eval_root}/q4b-sft-greedy-vt-${timestamp}.worker.log" \
NUM_ROLLOUTS=1 \
ROLLOUT_TEMPERATURE=0.0 \
EVAL_BATCH_SIZE="${GREEDY_EVAL_BATCH_SIZE:-512}" \
bash "${repo_root}/tools/evaluation/launch_q4b_e2_val_test_eval_detached_rjob.sh"

echo "[sft-pipeline] submit no-hint test best/pass@8 eval"
MODEL="$latest_ckpt" \
RJOB_NAME="q4b-sft-nohint-pass8-${timestamp}" \
RJOB_GROUP="${EVAL_RJOB_GROUP:-aos}" \
RJOB_CHARGED_GROUP="${EVAL_RJOB_CHARGED_GROUP:-${EVAL_RJOB_GROUP:-aos}}" \
RJOB_CPU="${EVAL_RJOB_CPU:-28}" \
RJOB_GPU="${EVAL_RJOB_GPU:-8}" \
RJOB_MEMORY="${EVAL_RJOB_MEMORY:-600000}" \
EVAL_JSONL="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}" \
OUT_DIR="${eval_root}/${eval_slug}-test-nohint-pass8" \
LOG_PATH="${eval_root}/q4b-sft-nohint-pass8-${timestamp}.submit.log" \
WORKER_LOG="${eval_root}/q4b-sft-nohint-pass8-${timestamp}.worker.log" \
NUM_ROLLOUTS=8 \
ROLLOUT_TEMPERATURE="${PASS8_TEMPERATURE:-0.7}" \
TOP_P="${PASS8_TOP_P:-0.95}" \
EVAL_BATCH_SIZE="${PASS8_EVAL_BATCH_SIZE:-256}" \
bash "${repo_root}/tools/evaluation/launch_single_ckpt_split_eval_detached_rjob.sh"

for hint_mode in gaussian zoom_in; do
    echo "[sft-pipeline] submit teacher-hint pass@8 eval: ${hint_mode}"
    MODEL="$latest_ckpt" \
    RJOB_NAME="q4b-sft-hint-${hint_mode}-pass8-${timestamp}" \
    RJOB_GROUP="${EVAL_RJOB_GROUP:-aos}" \
    RJOB_CHARGED_GROUP="${EVAL_RJOB_CHARGED_GROUP:-${EVAL_RJOB_GROUP:-aos}}" \
    RJOB_CPU="${EVAL_RJOB_CPU:-28}" \
    RJOB_GPU="${EVAL_RJOB_GPU:-8}" \
    RJOB_MEMORY="${EVAL_RJOB_MEMORY:-600000}" \
    EVAL_JSONL="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}" \
    OUT_DIR="${eval_root}/${eval_slug}-test-hint-${hint_mode}-pass8" \
    HINT_MODE="$hint_mode" \
    LOG_PATH="${eval_root}/q4b-sft-hint-${hint_mode}-pass8-${timestamp}.submit.log" \
    WORKER_LOG="${eval_root}/q4b-sft-hint-${hint_mode}-pass8-${timestamp}.worker.log" \
    NUM_ROLLOUTS=8 \
    ROLLOUT_TEMPERATURE="${PASS8_TEMPERATURE:-0.7}" \
    TOP_P="${PASS8_TOP_P:-0.95}" \
    EVAL_BATCH_SIZE="${HINT_PASS8_EVAL_BATCH_SIZE:-128}" \
    bash "${repo_root}/tools/evaluation/launch_teacher_hint_pass8_detached_rjob.sh"
done

cat <<EOF
[sft-pipeline] submitted all stages
[sft-pipeline] latest_ckpt=${latest_ckpt}
[sft-pipeline] greedy_val_summary=${eval_root}/${eval_slug}-val-greedy/threshold_summary.json
[sft-pipeline] greedy_test_summary=${eval_root}/${eval_slug}-test-greedy/threshold_summary.json
[sft-pipeline] nohint_pass8_summary=${eval_root}/${eval_slug}-test-nohint-pass8/threshold_summary.json
[sft-pipeline] gaussian_hint_pass8_summary=${eval_root}/${eval_slug}-test-hint-gaussian-pass8/summary.json
[sft-pipeline] zoom_in_hint_pass8_summary=${eval_root}/${eval_slug}-test-hint-zoom_in-pass8/summary.json
EOF
