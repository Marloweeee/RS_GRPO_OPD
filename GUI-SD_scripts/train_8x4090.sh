#!/usr/bin/env bash
# Single-machine 8x RTX 4090 training entry for RS-GRPO-OPD.
#
# Default layout:
#   - GPU 0-6: distributed GRPO+SDPO training
#   - GPU 7: local vLLM rollout server
#   - checkpoints/logs/artifacts: ${RUN_ROOT}, outside the git repository by default
#
# Minimal usage:
#   MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct \
#   DATA_ROOT=/path/to/data \
#   bash GUI-SD_scripts/train_8x4090.sh
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

if [ -n "${GUI_SD_ENV_ROOT:-}" ]; then
    export PATH="${GUI_SD_ENV_ROOT}/bin:${PATH}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
SWIFT_BIN="${SWIFT_BIN:-swift}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[8x4090] ERROR: python not found. Activate the conda environment first." >&2
    exit 1
fi

if ! command -v "${SWIFT_BIN}" >/dev/null 2>&1; then
    echo "[8x4090] ERROR: swift CLI not found. Install requirements and activate the environment first." >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
    if [ "${gpu_count}" -lt 8 ] && [ "${ALLOW_FEWER_GPUS:-false}" != "true" ]; then
        echo "[8x4090] ERROR: detected ${gpu_count} GPU(s), but this script expects 8 GPUs." >&2
        echo "[8x4090] Set ALLOW_FEWER_GPUS=true only if you also adjust TRAIN_CUDA_VISIBLE_DEVICES/NPROC_PER_NODE." >&2
        exit 1
    fi
else
    echo "[8x4090] WARNING: nvidia-smi not found; skip GPU count check." >&2
fi

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONFAULTHANDLER=1
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-10000}"

data_root="${DATA_ROOT:-${repo_root}/data}"
if [ ! -d "${data_root}" ] && [ -d "${repo_root}/guisd_data" ]; then
    data_root="${repo_root}/guisd_data"
fi

run_name="${RUN_NAME:-rs-grpo-opd-qwen3vl4b-8x4090}"
run_root="${RUN_ROOT:-${HOME}/rs_grpo_opd_runs}"
ckpt_root="${CKPT_ROOT:-${run_root}/checkpoints}"
artifact_root="${ARTIFACT_ROOT:-${run_root}/artifacts}"
log_dir="${LOG_DIR:-${run_root}/logs}"
cache_root="${CACHE_ROOT:-/tmp/rs_grpo_opd_${USER:-user}_${run_name}}"

model_path="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
teacher_path="${TEACHER_PATH:-${model_path}}"
ref_model_path="${REF_MODEL_PATH:-${model_path}}"
train_jsonl="${TRAIN_JSONL:-${data_root}/rs_full/rs_train.jsonl}"
test_jsonl="${TEST_JSONL:-${data_root}/rs_full/rs_test.jsonl}"
mask_dir="${OPSD_MASK_DIR:-${artifact_root}/train_cache/${run_name}}"
eval_out="${EVAL_OUT:-${artifact_root}/eval/${run_name}}"

mkdir -p "${ckpt_root}/${run_name}" "${mask_dir}" "${eval_out}" "${log_dir}" "${cache_root}"

log_file="${LOG_FILE:-${log_dir}/${run_name}_$(date +%Y%m%d-%H%M%S).log}"
rollout_log="${ROLLOUT_LOG:-${log_dir}/${run_name}_rollout_$(date +%Y%m%d-%H%M%S).log}"
exec > >(tee -a "${log_file}") 2>&1

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${cache_root}/xdg}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${cache_root}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${cache_root}/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${cache_root}/vllm}"
mkdir -p "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}"

if [ ! -f "${train_jsonl}" ]; then
    echo "[8x4090] ERROR: train jsonl not found: ${train_jsonl}" >&2
    echo "[8x4090] Set DATA_ROOT or TRAIN_JSONL to your converted RS dataset." >&2
    exit 1
fi

if [[ "${model_path}" == /* || "${model_path}" == .* ]] && [ ! -e "${model_path}" ]; then
    echo "[8x4090] ERROR: MODEL_PATH not found: ${model_path}" >&2
    exit 1
fi

train_cuda="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
rollout_cuda="${ROLLOUT_CUDA_VISIBLE_DEVICES:-7}"
eval_cuda="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
rollout_host="${VLLM_SERVER_HOST:-127.0.0.1}"
rollout_port="${VLLM_SERVER_PORT:-8192}"
rollout_pid=""
started_rollout=false

cleanup_rollout() {
    if [ "${started_rollout}" = "true" ] && [ -n "${rollout_pid}" ]; then
        if kill -0 "${rollout_pid}" 2>/dev/null; then
            echo "[8x4090] stopping rollout server pid=${rollout_pid}"
            kill "${rollout_pid}" 2>/dev/null || true
            pkill -P "${rollout_pid}" 2>/dev/null || true
            wait "${rollout_pid}" 2>/dev/null || true
        fi
    fi
}
trap cleanup_rollout EXIT

rollout_ready() {
    "${PYTHON_BIN}" - "${rollout_host}" "${rollout_port}" <<'PY'
import sys
import urllib.request

host, port = sys.argv[1], sys.argv[2]
for path, method in (("/get_engine_type/", "POST"), ("/health/", "GET")):
    url = f"http://{host}:{port}{path}"
    try:
        data = b"" if method == "POST" else None
        req = urllib.request.Request(url, data=data, method=method)
        with urllib.request.urlopen(req, timeout=3) as resp:
            if 200 <= resp.status < 300:
                raise SystemExit(0)
    except Exception:
        pass
raise SystemExit(1)
PY
}

wait_for_rollout() {
    local timeout="${ROLLOUT_READY_TIMEOUT_SEC:-900}"
    local interval="${ROLLOUT_READY_INTERVAL_SEC:-5}"
    local attempts=$((timeout / interval))
    if [ "${attempts}" -lt 1 ]; then
        attempts=1
    fi
    for _ in $(seq 1 "${attempts}"); do
        if rollout_ready; then
            return 0
        fi
        if [ -n "${rollout_pid}" ] && ! kill -0 "${rollout_pid}" 2>/dev/null; then
            echo "[8x4090] ERROR: rollout server exited before ready. Last rollout log:" >&2
            tail -n 80 "${rollout_log}" >&2 || true
            return 1
        fi
        sleep "${interval}"
    done
    echo "[8x4090] ERROR: rollout server not ready after ${timeout}s. Last rollout log:" >&2
    tail -n 80 "${rollout_log}" >&2 || true
    return 1
}

find_latest_checkpoint() {
    local latest_v latest_ckpt
    latest_v="$(find "${ckpt_root}/${run_name}" -maxdepth 1 -type d -name 'v*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
    if [ -z "${latest_v}" ]; then
        return 1
    fi
    latest_ckpt="$(find "${latest_v}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
    if [ -z "${latest_ckpt}" ]; then
        return 1
    fi
    printf '%s\n' "${latest_ckpt}"
}

echo "[8x4090] repo_root=${repo_root}"
echo "[8x4090] python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "[8x4090] swift=$(command -v "${SWIFT_BIN}")"
echo "[8x4090] run_name=${run_name}"
echo "[8x4090] model_path=${model_path}"
echo "[8x4090] teacher_path=${teacher_path}"
echo "[8x4090] ref_model_path=${ref_model_path}"
echo "[8x4090] train_jsonl=${train_jsonl} ($(wc -l < "${train_jsonl}") samples)"
if [ -f "${test_jsonl}" ]; then
    echo "[8x4090] test_jsonl=${test_jsonl} ($(wc -l < "${test_jsonl}") samples)"
else
    echo "[8x4090] test_jsonl=${test_jsonl} (not found; final eval will be skipped)"
fi
echo "[8x4090] ckpt_root=${ckpt_root}"
echo "[8x4090] artifact_root=${artifact_root}"
echo "[8x4090] log_file=${log_file}"
echo "[8x4090] rollout_log=${rollout_log}"
echo "[8x4090] cache_root=${cache_root}"
echo "[8x4090] train_cuda=${train_cuda}"
echo "[8x4090] rollout_cuda=${rollout_cuda}"

"${PYTHON_BIN}" -m py_compile \
    swift/rlhf_trainers/grpo_trainer.py \
    swift/rlhf_trainers/args_mixin.py \
    swift/cus_rewards/bbox_iou_reward.py \
    swift/rewards/orm.py \
    tools/evaluation/eval_student.py

if rollout_ready; then
    if [ "${REUSE_ROLLOUT_SERVER:-false}" = "true" ]; then
        echo "[8x4090] reuse existing rollout server ${rollout_host}:${rollout_port}"
    else
        echo "[8x4090] ERROR: ${rollout_host}:${rollout_port} already has a rollout/vLLM server." >&2
        echo "[8x4090] Stop it, set VLLM_SERVER_PORT to another port, or set REUSE_ROLLOUT_SERVER=true." >&2
        exit 1
    fi
else
    echo "[8x4090] starting rollout server on GPU ${rollout_cuda}; log=${rollout_log}"
    CUDA_VISIBLE_DEVICES="${rollout_cuda}" \
    "${SWIFT_BIN}" rollout \
        --model_type qwen3_vl \
        --model "${model_path}" \
        --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTIL:-0.70}" \
        --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-12000}" \
        --vllm_data_parallel_size 1 \
        --host "${rollout_host}" \
        --port "${rollout_port}" \
        > "${rollout_log}" 2>&1 &
    rollout_pid=$!
    started_rollout=true
    wait_for_rollout
fi

nnodes=1
nproc_per_node="${NPROC_PER_NODE:-7}"
master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-29500}"

resume_args=()
resume_checkpoint="${RESUME_FROM_CHECKPOINT:-}"
if [ -z "${resume_checkpoint}" ] && [ "${AUTO_RESUME:-false}" = "true" ]; then
    resume_checkpoint="$(find_latest_checkpoint || true)"
fi
if [ -n "${resume_checkpoint}" ]; then
    if [ ! -d "${resume_checkpoint}" ]; then
        echo "[8x4090] ERROR: resume checkpoint not found: ${resume_checkpoint}" >&2
        exit 1
    fi
    resume_args=(--resume_from_checkpoint "${resume_checkpoint}" --resume_only_model "${RESUME_ONLY_MODEL:-true}")
    echo "[8x4090] resume_from_checkpoint=${resume_checkpoint}"
else
    echo "[8x4090] resume_from_checkpoint=<none>"
fi

echo "[8x4090] start training"
NNODES="${nnodes}" \
NODE_RANK=0 \
MASTER_ADDR="${master_addr}" \
MASTER_PORT="${master_port}" \
NPROC_PER_NODE="${nproc_per_node}" \
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
    --ref_model "${ref_model_path}" \
    --ref_model_type qwen3_vl \
    --teacher_model "${teacher_path}" \
    --teacher_model_type qwen3_vl \
    --train_type full \
    --dataset "${train_jsonl}" \
    --reward_funcs bbox-geometry \
    --num_generations "${NUM_GENERATIONS:-8}" \
    --num_iterations "${NUM_ITERATIONS:-1}" \
    --generation_batch_size "${GENERATION_BATCH_SIZE:-56}" \
    --temperature "${ROLLOUT_TEMPERATURE:-0.7}" \
    --top_p "${TOP_P:-0.95}" \
    --top_k "${TOP_K:-50}" \
    --beta "${GRPO_BETA:-0.04}" \
    --torch_dtype bfloat16 \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --max_steps "${MAX_STEPS:--1}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
    --learning_rate "${LR:-1e-6}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
    --save_steps "${SAVE_STEPS:-250}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-4}" \
    --logging_steps "${LOGGING_STEPS:-1}" \
    --max_length "${MAX_LENGTH:-12000}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH:-64}" \
    --output_dir "${ckpt_root}/${run_name}" \
    --warmup_ratio "${WARMUP_RATIO:-0.03}" \
    --save_only_model "${SAVE_ONLY_MODEL:-true}" \
    --log_completions "${LOG_COMPLETIONS:-true}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
    --dataset_num_proc "${DATASET_NUM_PROC:-4}" \
    --deepspeed "${DEEPSPEED_CONFIG:-zero3}" \
    --teacher_deepspeed "${TEACHER_DEEPSPEED_CONFIG:-zero3}" \
    --offload_teacher_model "${OFFLOAD_TEACHER_MODEL:-false}" \
    --attn_impl "${ATTN_IMPL:-flash_attn}" \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host "${rollout_host}" \
    --vllm_server_port "${rollout_port}" \
    --vllm_server_timeout "${VLLM_SERVER_TIMEOUT:-900}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-12000}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTIL:-0.70}"

latest_ckpt="$(find_latest_checkpoint || true)"
if [ -z "${latest_ckpt}" ]; then
    echo "[8x4090] ERROR: no checkpoint found under ${ckpt_root}/${run_name}" >&2
    exit 1
fi
echo "[8x4090] latest_ckpt=${latest_ckpt}"

cleanup_rollout
started_rollout=false
sleep 5

if [ "${RUN_EVAL_AFTER_TRAIN:-true}" = "true" ]; then
    if [ ! -f "${test_jsonl}" ]; then
        echo "[8x4090] skip eval: test jsonl not found: ${test_jsonl}"
    else
        eval_dir="${eval_out}/$(basename "${latest_ckpt}")"
        echo "[8x4090] start eval on GPU ${eval_cuda}; out=${eval_dir}"
        CUDA_VISIBLE_DEVICES="${eval_cuda}" \
        "${PYTHON_BIN}" tools/evaluation/eval_student.py \
            --model "${latest_ckpt}" \
            --test_jsonl "${test_jsonl}" \
            --out_dir "${eval_dir}" \
            --tp "${EVAL_TP:-1}" \
            --max_model_len "${EVAL_MAX_MODEL_LEN:-12000}" \
            --gpu_mem_util "${EVAL_GPU_MEM_UTIL:-0.75}" \
            --eval_batch_size "${EVAL_BATCH_SIZE:-128}" \
            --max_new_tokens "${EVAL_MAX_NEW_TOKENS:-128}" \
            --seed "${EVAL_SEED:-42}"

        "${PYTHON_BIN}" - <<PY
import json
import os

summary_path = os.path.join("${eval_dir}", "summary.json")
with open(summary_path) as f:
    metrics = json.load(f)
print("[8x4090] summary_path=" + summary_path)
print("[8x4090] IoU@0.5=%.4f" % metrics["IoU@0.5"])
print("[8x4090] mIoU=%.4f IoU@0.7=%.4f parse_rate=%.4f valid_rate=%.4f" % (
    metrics["mIoU"], metrics["IoU@0.7"], metrics["parse_rate"], metrics["valid_rate"]))
PY
    fi
fi

echo "[8x4090] done"
