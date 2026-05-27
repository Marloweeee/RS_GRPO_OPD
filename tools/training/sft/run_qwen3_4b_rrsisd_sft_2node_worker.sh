#!/usr/bin/env bash
# Two-node worker for Qwen3-VL-4B RRSIS-D SFT.
#
# This worker trains a plain supervised baseline on rs_full/rs_train.jsonl.
# It intentionally does not start a rollout server and does not run evaluation;
# evaluate checkpoints separately with tools/evaluation/eval_student.py.
set -euo pipefail

repo_root="${REPO_ROOT:-/data/codes/gui_grounding/GUI-SD-code-main}"
env_root="${GUI_SD_ENV_ROOT:-/data/codes/gui_grounding/conda_envs/GUI-SD}"
cd "$repo_root"

export PATH="${env_root}/bin:${PATH}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONFAULTHANDLER=1
export PYTHON_BIN="${PYTHON_BIN:-${env_root}/bin/python}"
export SWIFT_BIN="${SWIFT_BIN:-${env_root}/bin/swift}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-10000}"

cache_root="${GUI_SD_CACHE_ROOT:-/tmp/guisd_sft_cache_${RUN_NAME:-qwen3_4b_sft}_$(hostname)}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${cache_root}/xdg}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${cache_root}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${cache_root}/triton}"
mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" 2>/dev/null || true

run_name="${RUN_NAME:-gui-sd-qwen3-4b-rrsisd_sft_e1}"
ckpt_root="${CKPT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sft}"
artifact_root="${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sft}"
model_path="${MODEL_PATH:-${BASE_MODEL_PATH:-/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-4B-Instruct}}"
train_jsonl="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}"
save_only_model="${SAVE_ONLY_MODEL:-true}"
tuner_type="${TUNER_TYPE:-full}"

export RUN_NAME="$run_name"
export CKPT_ROOT="$ckpt_root"
export CHECKPOINT_ROOT="$ckpt_root"
export ARTIFACT_ROOT="$artifact_root"
export MODEL_PATH="$model_path"
export TRAIN_JSONL="$train_jsonl"
export TUNER_TYPE="$tuner_type"

export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-29900}"
export RENDEZVOUS_ID="${RENDEZVOUS_ID:-${run_name}}"
export RENDEZVOUS_ROOT="${RENDEZVOUS_ROOT:-${artifact_root}/multinode_rendezvous}"
export RENDEZVOUS_TIMEOUT_SEC="${RENDEZVOUS_TIMEOUT_SEC:-1200}"
export CLEANUP_ON_REMOTE_FAILURE="${CLEANUP_ON_REMOTE_FAILURE:-true}"

export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export SAVE_STEPS="${SAVE_STEPS:-50}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-6}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export LR="${LR:-${LEARNING_RATE:-1e-5}}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export MAX_LENGTH="${MAX_LENGTH:-20000}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero2}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"

rendezvous_dir="${RENDEZVOUS_ROOT}/${RENDEZVOUS_ID}"
worker_log_dir="${MULTINODE_WORKER_LOG_DIR:-${artifact_root}/logs/sft_worker_logs}"
mkdir -p "$worker_log_dir" "${ckpt_root}/${run_name}" "$rendezvous_dir" 2>/dev/null || true
worker_log="${worker_log_dir}/${run_name}_rank-${NODE_RANK:-unknown}_$(hostname)_$(date +%Y%m%d-%H%M%S).log"
touch "$worker_log" 2>/dev/null || true
if [ -w "$worker_log" ]; then
    exec > >(tee -a "$worker_log") 2>&1
fi

log() {
    echo "[4b-sft-worker][$(date '+%F %T')][host=$(hostname)][rank=${NODE_RANK:-?}] $*"
}

infer_rank() {
    if [ -n "${NODE_RANK:-}" ]; then
        printf '%s\n' "$NODE_RANK"
        return 0
    fi
    for key in REPLICA_INDEX REPLICA_RANK RANK_INDEX WORKER_INDEX POD_INDEX INDEX; do
        local value="${!key:-}"
        if [[ "$value" =~ ^[0-9]+$ ]]; then
            printf '%s\n' "$value"
            return 0
        fi
    done
    "$PYTHON_BIN" - "$NNODES" "$(hostname)" <<'PY'
import hashlib
import re
import sys

nnodes = int(sys.argv[1])
host = sys.argv[2]
tokens = re.split(r'[-_.]', host)
for token in reversed(tokens):
    if token.isdigit() and int(token) < nnodes:
        print(int(token))
        raise SystemExit(0)

suffix = tokens[-1]
for idx in range(nnodes):
    if hashlib.md5(str(idx).encode()).hexdigest().startswith(suffix):
        print(idx)
        raise SystemExit(0)

raise SystemExit(f"cannot infer NODE_RANK from hostname={host}; set NODE_RANK explicitly")
PY
}

get_primary_ip() {
    local ip
    ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [ -z "$ip" ]; then
        ip="$(getent hosts "$(hostname)" | awk '{print $1; exit}')"
    fi
    if [ -z "$ip" ]; then
        echo "ERROR: cannot determine primary IP for host $(hostname)" >&2
        exit 1
    fi
    printf '%s\n' "$ip"
}

is_model_checkpoint_dir() {
    local path="$1"
    [ -n "$path" ] || return 1
    [ -d "$path" ] || return 1
    [ -f "${path}/config.json" ] || return 1
    [ -f "${path}/model.safetensors.index.json" ] || [ -f "${path}/pytorch_model.bin" ] || [ -n "$(find "$path" -maxdepth 1 -name '*.safetensors' -print -quit 2>/dev/null)" ]
}

write_node_file() {
    local status="$1"
    local tmp="${rendezvous_dir}/node_${NODE_RANK}.env.tmp.$$"
    local final="${rendezvous_dir}/node_${NODE_RANK}.env"
    {
        echo "NODE_RANK=${NODE_RANK}"
        echo "STATUS=${status}"
        echo "HOSTNAME=$(hostname)"
        echo "POD_IP=${pod_ip}"
        echo "MASTER_ADDR_CANDIDATE=${pod_ip}"
        echo "UPDATED_AT=$(date '+%F %T')"
    } > "$tmp"
    mv "$tmp" "$final"
    sync "$final" 2>/dev/null || sync 2>/dev/null || true
}

mark_phase() {
    local phase="$1"
    local tmp="${rendezvous_dir}/node_${NODE_RANK}.${phase}.tmp.$$"
    local final="${rendezvous_dir}/node_${NODE_RANK}.${phase}"
    {
        echo "NODE_RANK=${NODE_RANK}"
        echo "PHASE=${phase}"
        echo "HOSTNAME=$(hostname)"
        echo "UPDATED_AT=$(date '+%F %T')"
    } > "$tmp"
    mv "$tmp" "$final"
    sync "$final" 2>/dev/null || sync 2>/dev/null || true
}

count_cluster_phase() {
    local phase="$1"
    local idx ready file
    ready=0
    for idx in $(seq 0 $((NNODES - 1))); do
        file="${rendezvous_dir}/node_${idx}.${phase}"
        if [ -f "$file" ]; then
            ready=$((ready + 1))
        fi
    done
    printf '%s\n' "$ready"
}

wait_for_cluster_phase() {
    local phase="$1"
    local start now ready
    start="$(date +%s)"
    while true; do
        ready="$(count_cluster_phase "$phase")"
        if [ "$ready" -eq "$NNODES" ]; then
            return 0
        fi
        now="$(date +%s)"
        if [ $((now - start)) -gt "$RENDEZVOUS_TIMEOUT_SEC" ]; then
            echo "ERROR: rendezvous timed out; phase=${phase} ready=${ready}/${NNODES}" >&2
            ls -la "$rendezvous_dir" >&2 || true
            return 1
        fi
        log "waiting for cluster phase '${phase}': ${ready}/${NNODES}"
        sleep 10
    done
}

wait_for_terminal_phase() {
    local done_phase="$1"
    local failed_phase="$2"
    local start now done_count failed_count
    start="$(date +%s)"
    while true; do
        failed_count="$(count_cluster_phase "$failed_phase")"
        if [ "$failed_count" -gt 0 ]; then
            log "detected ${failed_phase}: ${failed_count}/${NNODES}"
            return 1
        fi
        done_count="$(count_cluster_phase "$done_phase")"
        if [ "$done_count" -eq "$NNODES" ]; then
            return 0
        fi
        now="$(date +%s)"
        if [ $((now - start)) -gt "$RENDEZVOUS_TIMEOUT_SEC" ]; then
            echo "ERROR: terminal rendezvous timed out; ${done_phase}=${done_count}/${NNODES} ${failed_phase}=${failed_count}/${NNODES}" >&2
            ls -la "$rendezvous_dir" >&2 || true
            return 1
        fi
        log "waiting for terminal phase: ${done_phase}=${done_count}/${NNODES} ${failed_phase}=${failed_count}/${NNODES}"
        sleep 10
    done
}

watch_remote_failure() {
    local train_pid="$1"
    local failed_count
    while kill -0 "$train_pid" 2>/dev/null; do
        failed_count="$(count_cluster_phase train_failed)"
        if [ "$failed_count" -gt 0 ]; then
            log "remote train_failed marker detected; terminating local training process group pid=${train_pid}"
            kill -TERM "-${train_pid}" 2>/dev/null || kill "$train_pid" 2>/dev/null || true
            sleep 20
            if kill -0 "$train_pid" 2>/dev/null; then
                kill -KILL "-${train_pid}" 2>/dev/null || kill -9 "$train_pid" 2>/dev/null || true
            fi
            return 0
        fi
        sleep "${TRAIN_FAILURE_POLL_SEC:-10}"
    done
}

find_latest_checkpoint() {
    local latest_v latest_ckpt
    latest_v="$(find "${ckpt_root}/${run_name}" -maxdepth 1 -type d -name 'v*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
    if [ -z "$latest_v" ]; then
        return 1
    fi
    latest_ckpt="$(find "$latest_v" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
    if [ -z "$latest_ckpt" ]; then
        return 1
    fi
    printf '%s\n' "$latest_ckpt"
}

export NODE_RANK="$(infer_rank)"
if [ "$NODE_RANK" -ge "$NNODES" ]; then
    echo "ERROR: inferred NODE_RANK=${NODE_RANK}, but NNODES=${NNODES}" >&2
    exit 1
fi

pod_ip="$(get_primary_ip)"

log "cwd=$PWD"
log "python=${PYTHON_BIN}"
log "swift=${SWIFT_BIN}"
log "run_name=${run_name}"
log "ckpt_root=${ckpt_root}"
log "artifact_root=${artifact_root}"
log "model_path=${model_path}"
log "train_jsonl=${train_jsonl}"
log "nnodes=${NNODES} nproc_per_node=${NPROC_PER_NODE} node_rank=${NODE_RANK}"
log "pod_ip=${pod_ip}"
log "train_cuda=${TRAIN_CUDA_VISIBLE_DEVICES}"
log "per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} grad_acc=${GRADIENT_ACCUMULATION_STEPS} lr=${LR}"
log "epochs=${NUM_TRAIN_EPOCHS} max_steps=${MAX_STEPS} save_steps=${SAVE_STEPS} save_total_limit=${SAVE_TOTAL_LIMIT}"
log "max_length=${MAX_LENGTH} deepspeed=${DEEPSPEED_CONFIG} tuner_type=${TUNER_TYPE}"
log "cache_root=${cache_root}"

if ! is_model_checkpoint_dir "$model_path"; then
    echo "ERROR: MODEL_PATH is not a usable model checkpoint: ${model_path}" >&2
    find "$model_path" -maxdepth 1 -type f -printf '%s %p\n' 2>/dev/null | sort -n | tail -n 40 >&2 || true
    exit 1
fi
if [ ! -f "$train_jsonl" ]; then
    echo "ERROR: train jsonl not found: ${train_jsonl}" >&2
    exit 1
fi

"$PYTHON_BIN" -m py_compile swift/cli/sft.py swift/trainers/arguments.py

cuda_probe_interval="${CUDA_READY_INTERVAL_SEC:-10}"
cuda_probe_timeout="${CUDA_READY_TIMEOUT_SEC:-180}"
cuda_probe_start="$(date +%s)"
while true; do
    if "$PYTHON_BIN" - <<'PY'
import os
import torch

prefix = "[4b-sft-worker]"
print(f"{prefix} torch={torch.__version__}")
print(f"{prefix} cuda_available={torch.cuda.is_available()}")
print(f"{prefix} cuda_device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
    raise SystemExit("CUDA is not ready or fewer than 8 GPUs are visible")
dataset = os.environ["TRAIN_JSONL"]
if not os.path.isfile(dataset):
    raise SystemExit(f"dataset not found: {dataset}")
print(f"{prefix} dataset={dataset}")
PY
    then
        break
    fi
    cuda_probe_now="$(date +%s)"
    if [ $((cuda_probe_now - cuda_probe_start)) -gt "$cuda_probe_timeout" ]; then
        echo "ERROR: CUDA did not become ready within ${cuda_probe_timeout}s." >&2
        exit 1
    fi
    log "CUDA not ready yet; retrying in ${cuda_probe_interval}s"
    sleep "$cuda_probe_interval"
done

write_node_file "ready"
mark_phase "ready"
wait_for_cluster_phase "ready"
MASTER_ADDR="$(grep -E '^POD_IP=' "${rendezvous_dir}/node_0.env" | tail -n 1 | cut -d= -f2-)"
export MASTER_ADDR
log "master_addr=${MASTER_ADDR}:${MASTER_PORT}"

resume_args=()
if [ "${ALLOW_RESUME:-false}" = "true" ]; then
    resume_checkpoint="${RESUME_FROM_CHECKPOINT:-}"
    if [ -z "$resume_checkpoint" ] && [ "${AUTO_RESUME:-true}" = "true" ]; then
        resume_checkpoint="$(
            find "${ckpt_root}/${run_name}" -path '*/checkpoint-*' -type d -printf '%f %p\n' 2>/dev/null \
                | awk -v max_steps="${MAX_STEPS}" '
                    {
                        step = $1
                        sub(/^checkpoint-/, "", step)
                        if (step ~ /^[0-9]+$/ && (max_steps == -1 || step < max_steps)) {
                            print step " " $2
                        }
                    }' \
                | sort -n \
                | tail -1 \
                | cut -d' ' -f2- || true
        )"
    fi
    if [ -n "$resume_checkpoint" ]; then
        if [ ! -d "$resume_checkpoint" ]; then
            echo "ERROR: resume checkpoint not found: ${resume_checkpoint}" >&2
            exit 1
        fi
        resume_only_model="${RESUME_ONLY_MODEL:-${save_only_model}}"
        resume_args=(--resume_from_checkpoint "$resume_checkpoint" --resume_only_model "$resume_only_model")
        log "resume_from_checkpoint=${resume_checkpoint}; resume_only_model=${resume_only_model}"
    fi
else
    export AUTO_RESUME=false
    log "resume disabled; starting from base model"
fi

set +e
setsid env \
    NNODES="$NNODES" \
    NODE_RANK="$NODE_RANK" \
    MASTER_ADDR="$MASTER_ADDR" \
    MASTER_PORT="$MASTER_PORT" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    IMAGE_MAX_TOKEN_NUM="$IMAGE_MAX_TOKEN_NUM" \
    PYTHONFAULTHANDLER=1 \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" \
    "$SWIFT_BIN" sft \
    "${resume_args[@]}" \
    --model "$model_path" \
    --model_type qwen3_vl \
    --tuner_type "$TUNER_TYPE" \
    --train_type "$TUNER_TYPE" \
    --dataset "$train_jsonl" \
    --torch_dtype bfloat16 \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --learning_rate "$LR" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --logging_steps 1 \
    --max_length "$MAX_LENGTH" \
    --output_dir "${ckpt_root}/${run_name}" \
    --warmup_ratio "$WARMUP_RATIO" \
    --save_only_model "$save_only_model" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --dataset_num_proc "$DATASET_NUM_PROC" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --attn_impl flash_attn &
train_pid=$!
watch_remote_failure "$train_pid" &
watcher_pid=$!
wait "$train_pid"
train_status=$?
kill "$watcher_pid" 2>/dev/null || true
wait "$watcher_pid" 2>/dev/null || true
set -e

if [ "$train_status" -ne 0 ]; then
    log "training failed with status=${train_status}"
    mark_phase "train_failed"
    exit "$train_status"
fi

mark_phase "train_done"
if ! wait_for_terminal_phase train_done train_failed; then
    mark_phase "train_failed"
    exit 1
fi

if [ "$NODE_RANK" = "0" ]; then
    latest_ckpt="$(find_latest_checkpoint)" || {
        echo "ERROR: no checkpoint found under ${ckpt_root}/${run_name}" >&2
        mark_phase "train_failed"
        exit 1
    }
    log "latest_ckpt=${latest_ckpt}"
fi

log "run completed"
