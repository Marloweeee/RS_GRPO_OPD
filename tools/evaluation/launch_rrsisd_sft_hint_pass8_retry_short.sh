#!/usr/bin/env bash
set -euo pipefail

cd /data/codes/gui_grounding/GUI-SD-code-main

latest_ckpt="/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sft/gui-sd-qwen3-4b-rrsisd_sft_e1-20260526-220915/v0-20260526-220853/checkpoint-96"
eval_root="/data/codes/gui_grounding/data/logs/eval"
eval_slug="qwen3-4b-rrsisd-sft-20260526-220915-checkpoint-96"

export RJOB_GROUP=gui_agent
export RJOB_CHARGED_GROUP=gui_agent
export RJOB_CPU=64
export RJOB_GPU=8
export RJOB_MEMORY=800000
export RJOB_POSITIVE_TAGS=feature/gpfs=yes
export GPU_MEM_UTIL=0.82
export EVAL_BATCH_SIZE=32
export NUM_ROLLOUTS=8
export ROLLOUT_TEMPERATURE=0.7
export TOP_P=0.95
export OPSD_HINT_BOX_COLOR=magenta

submit_hint() {
    local hint_mode="$1"
    local rjob_name="$2"
    echo "[sft-hint-pass8-short] submit ${hint_mode} as ${rjob_name} at $(date +%F_%T)"
    MODEL="$latest_ckpt" \
        RJOB_NAME="$rjob_name" \
        EVAL_JSONL=/data/codes/gui_grounding/data/rs_full/rs_test.jsonl \
        OUT_DIR="${eval_root}/${eval_slug}-test-hint-${hint_mode}-pass8-b32" \
        HINT_MODE="$hint_mode" \
        LOG_PATH="${eval_root}/${rjob_name}.submit.log" \
        WORKER_LOG="${eval_root}/${rjob_name}.worker.log" \
        bash tools/evaluation/launch_teacher_hint_pass8_detached_rjob.sh
    echo "[sft-hint-pass8-short] submitted ${hint_mode}; summary=${eval_root}/${eval_slug}-test-hint-${hint_mode}-pass8-b32/summary.json"
}

submit_hint gaussian sftg-p8-05262248
submit_hint zoom_in sftz-p8-05262248

echo "[sft-hint-pass8-short] all submitted at $(date +%F_%T)"
