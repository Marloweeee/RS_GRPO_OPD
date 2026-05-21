#!/usr/bin/env bash
set -euo pipefail

cd /data/codes/gui_grounding/GUI-SD-code-main

PYTHON_BIN=${PYTHON_BIN:-/data/codes/gui_grounding/conda_envs/GUI-SD/bin/python}
SWIFT_BIN=${SWIFT_BIN:-/data/codes/gui_grounding/conda_envs/GUI-SD/bin/swift}
run_name=${RUN_NAME:-gui-sd-4b_rs_grpo_sdpo}
ckpt_root=${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}
artifact_root=${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd}
model_path=${MODEL_PATH:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-4b_student_rs_full_legacy/v0-20260517-230632/checkpoint-80}
teacher_path=${TEACHER_PATH:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd/gui-sd-4b_student_rs_full_legacy/v0-20260517-230632/checkpoint-80}
train_jsonl=${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_sub_dir/rs_train.jsonl}
test_jsonl=${TEST_JSONL:-/data/codes/gui_grounding/data/rs_sub_dir/rs_test.jsonl}
eval_out=${EVAL_OUT:-${artifact_root}/eval/${run_name}}
mask_dir=${OPSD_MASK_DIR:-${artifact_root}/train_cache/${run_name}}
save_only_model=${SAVE_ONLY_MODEL:-true}

mkdir -p "${ckpt_root}/${run_name}" "${mask_dir}" "${eval_out}"

rollout_host=${VLLM_SERVER_HOST:-127.0.0.1}
rollout_port=${VLLM_SERVER_PORT:-8192}
rollout_cuda=${ROLLOUT_CUDA_VISIBLE_DEVICES:-7}
train_cuda=${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}
per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-8}
dataloader_num_workers=${DATALOADER_NUM_WORKERS:-4}
dataset_num_proc=${DATASET_NUM_PROC:-4}
max_length=${MAX_LENGTH:-20000}
max_completion_length=${MAX_COMPLETION_LENGTH:-64}
vllm_gpu_memory_util=${VLLM_GPU_MEMORY_UTIL:-${ROLLOUT_GPU_MEM_UTIL:-0.78}}
vllm_max_model_len=${VLLM_MAX_MODEL_LEN:-${ROLLOUT_MAX_MODEL_LEN:-20000}}

export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-10000}
export PYTHONPATH=/data/codes/gui_grounding/GUI-SD-code-main:${PYTHONPATH:-}
export PATH=/data/codes/gui_grounding/conda_envs/GUI-SD/bin:${PATH}
export PYTHONFAULTHANDLER=1

rollout_ready() {
    python - "$rollout_host" "$rollout_port" <<'PY'
import sys
import urllib.request

host, port = sys.argv[1], sys.argv[2]
url = f"http://{host}:{port}/get_engine_type/"
try:
    req = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=2) as resp:
        sys.exit(0 if 200 <= resp.status < 300 else 1)
except Exception:
    sys.exit(1)
PY
}

cleanup_rollout() {
    pkill -f "swift rollout" 2>/dev/null || true
    pkill -f vllm 2>/dev/null || true
}
trap cleanup_rollout EXIT

echo "[grpo-sdpo] cwd=$PWD"
echo "[grpo-sdpo] run_name=${run_name}"
echo "[grpo-sdpo] model=${model_path}"
echo "[grpo-sdpo] teacher=${teacher_path}"
echo "[grpo-sdpo] train=${train_jsonl}"
echo "[grpo-sdpo] test=${test_jsonl}"
echo "[grpo-sdpo] ckpt_root=${ckpt_root}"
echo "[grpo-sdpo] artifact_root=${artifact_root}"
echo "[grpo-sdpo] mask_dir=${mask_dir}"
echo "[grpo-sdpo] eval_out=${eval_out}"
echo "[grpo-sdpo] python=${PYTHON_BIN}"
echo "[grpo-sdpo] swift=${SWIFT_BIN}"
echo "[grpo-sdpo] train_cuda=${train_cuda}"
echo "[grpo-sdpo] rollout_cuda=${rollout_cuda}"
echo "[grpo-sdpo] per_device_train_batch_size=${per_device_train_batch_size}"
echo "[grpo-sdpo] gradient_accumulation_steps=${gradient_accumulation_steps}"
echo "[grpo-sdpo] max_length=${max_length}"
echo "[grpo-sdpo] max_completion_length=${max_completion_length}"
echo "[grpo-sdpo] vllm_gpu_memory_util=${vllm_gpu_memory_util}"
echo "[grpo-sdpo] vllm_max_model_len=${vllm_max_model_len}"

"${PYTHON_BIN}" -m py_compile \
    swift/rlhf_trainers/grpo_trainer.py \
    swift/rlhf_trainers/args_mixin.py \
    swift/cus_rewards/bbox_iou_reward.py \
    swift/rewards/orm.py

if rollout_ready; then
    echo "[grpo-sdpo] reuse healthy rollout server ${rollout_host}:${rollout_port}"
else
    CUDA_VISIBLE_DEVICES="${rollout_cuda}" \
    "${SWIFT_BIN}" rollout \
        --model_type qwen3_vl \
        --model "${model_path}" \
        --vllm_gpu_memory_utilization "${vllm_gpu_memory_util}" \
        --vllm_max_model_len "${vllm_max_model_len}" \
        --vllm_data_parallel_size 1 \
        --host "$rollout_host" \
        --port "$rollout_port" &

    echo "[grpo-sdpo] waiting for rollout server ${rollout_host}:${rollout_port}"
    for _ in $(seq 1 90); do
        if rollout_ready; then
            break
        fi
        sleep 5
    done
    if ! rollout_ready; then
        echo "[grpo-sdpo] ERROR: rollout server is not ready" >&2
        exit 1
    fi
fi

nnodes=${NNODES:-1}
nproc_per_node=${NPROC_PER_NODE:-7}
node_rank=${NODE_RANK:-${RANK:-0}}
master_addr=${MASTER_ADDR:-127.0.0.1}
master_port=${MASTER_PORT:-29500}

resume_checkpoint=${RESUME_FROM_CHECKPOINT:-}
if [ -z "${resume_checkpoint}" ] && [ "${AUTO_RESUME:-true}" = "true" ]; then
    resume_checkpoint=$(
        find "${ckpt_root}/${run_name}" -path '*/checkpoint-*' -type d -printf '%f %p\n' 2>/dev/null \
            | awk -v max_steps="${MAX_STEPS:-1}" '
                {
                    step = $1
                    sub(/^checkpoint-/, "", step)
                    if (step ~ /^[0-9]+$/ && step < max_steps) {
                        print step " " $2
                    }
                }' \
            | sort -n \
            | tail -1 \
            | cut -d' ' -f2- || true
    )
fi

resume_args=()
if [ -n "${resume_checkpoint}" ]; then
    if [ ! -d "${resume_checkpoint}" ]; then
        echo "[grpo-sdpo] ERROR: resume checkpoint not found: ${resume_checkpoint}" >&2
        exit 1
    fi
    resume_only_model=${RESUME_ONLY_MODEL:-${save_only_model}}
    resume_args=(--resume_from_checkpoint "${resume_checkpoint}" --resume_only_model "${resume_only_model}")
    echo "[grpo-sdpo] resume_from_checkpoint=${resume_checkpoint}"
    echo "[grpo-sdpo] resume_only_model=${resume_only_model}"
else
    echo "[grpo-sdpo] resume_from_checkpoint=<none>"
fi

NNODES=$nnodes \
NODE_RANK=$node_rank \
MASTER_ADDR=$master_addr \
MASTER_PORT=$master_port \
NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES="${train_cuda}" \
"${SWIFT_BIN}" rlhf \
    "${resume_args[@]}" \
    --rlhf_type grpo \
    --use_sdpo true \
    --sdpo_lambda "${SDPO_LAMBDA:-0.25}" \
    --sdpo_tau_good "${SDPO_TAU_GOOD:-0.5}" \
    --sdpo_tau_fail "${SDPO_TAU_FAIL:-0.3}" \
    --sdpo_delta "${SDPO_DELTA:-0.5}" \
    --sdpo_only_failed true \
    --sdpo_target "${SDPO_TARGET:-rollout}" \
    --opsd_mask_dir "${mask_dir}" \
    --opsd_token_weight_mode "${OPSD_TOKEN_WEIGHT_MODE:-uniform-entropy}" \
    --opsd_non_digit_weight "${OPSD_NON_DIGIT_WEIGHT:-0.05}" \
    --opsd_max_digit_len "${OPSD_MAX_DIGIT_LEN:-3}" \
    --opsd_mask_mode "${OPSD_MASK_MODE:-gaussian}" \
    --opsd_hint_mode "${OPSD_HINT_MODE:-hint}" \
    --opsd_hint_box_color "${OPSD_HINT_BOX_COLOR:-magenta}" \
    --opsd_jitter_ratio "${OPSD_JITTER_RATIO:-0.2}" \
    --model "${model_path}" \
    --model_type qwen3_vl \
    --ref_model "${model_path}" \
    --ref_model_type qwen3_vl \
    --teacher_model "${teacher_path}" \
    --teacher_model_type qwen3_vl \
    --train_type full \
    --dataset "${train_jsonl}" \
    --reward_funcs bbox-geometry \
    --num_generations 8 \
    --num_iterations 1 \
    --temperature "${ROLLOUT_TEMPERATURE:-0.7}" \
    --top_p "${TOP_P:-0.95}" \
    --top_k "${TOP_K:-50}" \
    --beta "${GRPO_BETA:-0.04}" \
    --torch_dtype bfloat16 \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --max_steps "${MAX_STEPS:-1}" \
    --per_device_train_batch_size "${per_device_train_batch_size}" \
    --learning_rate "${LR:-1e-6}" \
    --gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --save_steps "${SAVE_STEPS:-1}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
    --logging_steps 1 \
    --max_length "${max_length}" \
    --max_completion_length "${max_completion_length}" \
    --output_dir "${ckpt_root}/${run_name}" \
    --warmup_ratio 0 \
    --save_only_model "${save_only_model}" \
    --log_completions true \
    --dataloader_num_workers "${dataloader_num_workers}" \
    --dataset_num_proc "${dataset_num_proc}" \
    --deepspeed "${DEEPSPEED_CONFIG:-zero2}" \
    --teacher_deepspeed "${TEACHER_DEEPSPEED_CONFIG:-zero3}" \
    --offload_teacher_model "${OFFLOAD_TEACHER_MODEL:-false}" \
    --attn_impl flash_attn \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host "$rollout_host" \
    --vllm_server_port "$rollout_port" \
    --vllm_server_timeout 300 \
    --vllm_max_model_len "${vllm_max_model_len}" \
    --vllm_gpu_memory_utilization "${vllm_gpu_memory_util}"

latest_v=$(find "${ckpt_root}/${run_name}" -maxdepth 1 -type d -name 'v*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)
if [ -z "${latest_v}" ]; then
    echo "[grpo-sdpo] ERROR: no v* run directory found under ${ckpt_root}/${run_name}" >&2
    exit 1
fi

latest_ckpt=$(find "${latest_v}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)
if [ -z "${latest_ckpt}" ]; then
    echo "[grpo-sdpo] ERROR: no checkpoint found under ${ckpt_root}/${run_name}" >&2
    exit 1
fi

echo "[grpo-sdpo] latest_ckpt=${latest_ckpt}"

cleanup_rollout
sleep 5

"${PYTHON_BIN}" tools/evaluation/eval_student.py \
    --model "${latest_ckpt}" \
    --test_jsonl "${test_jsonl}" \
    --out_dir "${eval_out}/$(basename "${latest_ckpt}")" \
    --tp "${EVAL_TP:-1}" \
    --max_model_len "${EVAL_MAX_MODEL_LEN:-12000}" \
    --gpu_mem_util "${EVAL_GPU_MEM_UTIL:-0.85}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-0}" \
    --max_new_tokens 128 \
    --seed 42

"${PYTHON_BIN}" - <<PY
import json
import os

summary_path = os.path.join("${eval_out}", "$(basename "${latest_ckpt}")", "summary.json")
with open(summary_path) as f:
    metrics = json.load(f)
print("[grpo-sdpo] summary_path=" + summary_path)
print("[grpo-sdpo] IoU@0.5=%.4f" % metrics["IoU@0.5"])
print("[grpo-sdpo] mIoU=%.4f IoU@0.7=%.4f parse_rate=%.4f valid_rate=%.4f" % (
    metrics["mIoU"], metrics["IoU@0.7"], metrics["parse_rate"], metrics["valid_rate"]))
PY
