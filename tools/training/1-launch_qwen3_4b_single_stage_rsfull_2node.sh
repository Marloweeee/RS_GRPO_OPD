#!/usr/bin/env bash
# 启动 Qwen3-VL-4B 全量遥感数据的双机单阶段 GRPO+SDPO 训练。
#
# 默认使用 gui_agent 集群、2 个 replica、每个 replica 8 张卡：
#   - GPU 0-6: 分布式训练
#   - GPU 7: 本地 vLLM rollout server
#
# 如需切换到 aos：
#   RJOB_GROUP=aos RJOB_CHARGED_GROUP=aos bash tools/training/1-launch_qwen3_4b_single_stage_rsfull_2node.sh
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

timestamp="$(date +%Y%m%d-%H%M%S)"

export RUN_NAME="${RUN_NAME:-gui-sd-qwen3-4b-base_rsfull_grpo_sdpo_single_stage_e1-${timestamp}}"
export RJOB_NAME="${RJOB_NAME:-qwen3-4b-rsfull-single-stage-2n-${timestamp}}"

export RJOB_GROUP="${RJOB_GROUP:-gui_agent}"
export RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-${RJOB_GROUP}}"
export RJOB_CPU="${RJOB_CPU:-64}"
export RJOB_GPU="${RJOB_GPU:-8}"
export RJOB_MEMORY="${RJOB_MEMORY:-800000}"

export BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-4B-Instruct}"
export TRAIN_JSONL="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}"
export TEST_JSONL="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
export CKPT_ROOT="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd}"

export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export LR="${LR:-2e-6}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export SAVE_STEPS="${SAVE_STEPS:-50}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"

export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero2}"
export TEACHER_DEEPSPEED_CONFIG="${TEACHER_DEEPSPEED_CONFIG:-zero3}"
export OFFLOAD_TEACHER_MODEL="${OFFLOAD_TEACHER_MODEL:-false}"
export VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.82}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"

exec bash "${script_dir}/launch_qwen3_4b_base_rsfull_grpo_sdpo_2node_rjob.sh"
