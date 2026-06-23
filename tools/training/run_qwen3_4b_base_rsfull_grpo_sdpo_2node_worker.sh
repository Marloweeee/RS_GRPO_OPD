#!/usr/bin/env bash
# Two-node worker for Qwen3-VL-4B full rs_full GRPO+SDPO training.
#
# Each replica uses 7 GPUs for training and 1 GPU for a local rollout server.
# Replicas exchange NODE_RANK, MASTER_ADDR, and rollout server addresses through
# a shared rendezvous directory, then run one distributed swift rlhf job.
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

cache_root="${GUI_SD_CACHE_ROOT:-/tmp/guisd_vllm_train_cache_${RUN_NAME:-qwen3_4b}_$(hostname)}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${cache_root}/xdg}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${cache_root}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${cache_root}/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${cache_root}/vllm}"
mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" 2>/dev/null || true

run_name="${RUN_NAME:-gui-sd-qwen3-4b-base_rsfull_grpo_sdpo_2node_bsz4_gacc2_lr2e6_e1}"
ckpt_root="${CKPT_ROOT:-${CHECKPOINT_ROOT:-/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd}}"
artifact_root="${ARTIFACT_ROOT:-/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd}"
base_model_path="${BASE_MODEL_PATH:-/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-4B-Instruct}"
model_path="${MODEL_PATH:-${base_model_path}}"
ref_model_path="${REF_MODEL_PATH:-${model_path}}"
teacher_path="${TEACHER_PATH:-${TEACHER_MODEL_PATH:-${base_model_path}}}"
rollout_model_path="${ROLLOUT_MODEL_PATH:-${model_path}}"
train_jsonl="${TRAIN_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_train.jsonl}"
test_jsonl="${TEST_JSONL:-/data/codes/gui_grounding/data/rs_full/rs_test.jsonl}"
mask_dir="${OPSD_MASK_DIR:-${artifact_root}/train_cache/${run_name}}"
eval_out="${EVAL_OUT:-${artifact_root}/eval/${run_name}}"
save_only_model="${SAVE_ONLY_MODEL:-true}"
tuner_type="${TUNER_TYPE:-full}"

export RUN_NAME="$run_name"
export CKPT_ROOT="$ckpt_root"
export CHECKPOINT_ROOT="$ckpt_root"
export ARTIFACT_ROOT="$artifact_root"
export MODEL_PATH="$model_path"
export REF_MODEL_PATH="$ref_model_path"
export TEACHER_PATH="$teacher_path"
export ROLLOUT_MODEL_PATH="$rollout_model_path"
export TRAIN_JSONL="$train_jsonl"
export TEST_JSONL="$test_jsonl"
export OPSD_MASK_DIR="$mask_dir"
export EVAL_OUT="$eval_out"
export TUNER_TYPE="$tuner_type"

export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
export TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-7}"
export VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8292}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export RENDEZVOUS_ID="${RENDEZVOUS_ID:-${run_name}}"
export RENDEZVOUS_ROOT="${RENDEZVOUS_ROOT:-${artifact_root}/multinode_rendezvous}"
export RENDEZVOUS_TIMEOUT_SEC="${RENDEZVOUS_TIMEOUT_SEC:-1200}"
export ROLLOUT_READY_TIMEOUT_SEC="${ROLLOUT_READY_TIMEOUT_SEC:-900}"
export ROLLOUT_READY_INTERVAL_SEC="${ROLLOUT_READY_INTERVAL_SEC:-5}"
export CLEANUP_ROLLOUT_ON_EXIT="${CLEANUP_ROLLOUT_ON_EXIT:-true}"

export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export SAVE_STEPS="${SAVE_STEPS:-50}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export LR="${LR:-${LEARNING_RATE:-2e-6}}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
export MAX_LENGTH="${MAX_LENGTH:-20000}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-64}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero2}"
export TEACHER_DEEPSPEED_CONFIG="${TEACHER_DEEPSPEED_CONFIG:-zero3}"
export OFFLOAD_TEACHER_MODEL="${OFFLOAD_TEACHER_MODEL:-false}"
export VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-${ROLLOUT_GPU_MEM_UTIL:-0.82}}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-20000}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"

export SDPO_LAMBDA="${SDPO_LAMBDA:-0.25}"
export SDPO_TAU_GOOD="${SDPO_TAU_GOOD:-0.5}"
export SDPO_TAU_FAIL="${SDPO_TAU_FAIL:-0.3}"
export SDPO_DELTA="${SDPO_DELTA:-0.5}"
export SDPO_TARGET="${SDPO_TARGET:-rollout}"
export SDPO_HINT_SOURCE="${SDPO_HINT_SOURCE:-gt}"
export SDPO_SIBLING_SELECT_METRIC="${SDPO_SIBLING_SELECT_METRIC:-reward}"
export SDPO_SIBLING_FALLBACK="${SDPO_SIBLING_FALLBACK:-gt}"
export SDPO_TEACHER_REFRESH_STEP="${SDPO_TEACHER_REFRESH_STEP:-${TEACHER_REFRESH_STEP:--1}}"
export OPSD_TOKEN_WEIGHT_MODE="${OPSD_TOKEN_WEIGHT_MODE:-uniform-entropy}"
export OPSD_NON_DIGIT_WEIGHT="${OPSD_NON_DIGIT_WEIGHT:-0.05}"
export OPSD_MAX_DIGIT_LEN="${OPSD_MAX_DIGIT_LEN:-3}"
export OPSD_EMA_DECAY="${OPSD_EMA_DECAY:-0.0}"
export OPSD_MASK_MODE="${OPSD_MASK_MODE:-gaussian}"
export OPSD_ZOOM_RATIO="${OPSD_ZOOM_RATIO:-2.0}"
export OPSD_MIN_AREA_FRAC="${OPSD_MIN_AREA_FRAC:-0.1}"
export OPSD_GAUSSIAN_SIGMA_RATIO="${OPSD_GAUSSIAN_SIGMA_RATIO:-1.5}"
export OPSD_HINT_MODE="${OPSD_HINT_MODE:-hint}"
export OPSD_HINT_BOX_COLOR="${OPSD_HINT_BOX_COLOR:-magenta}"
export OPSD_JITTER_RATIO="${OPSD_JITTER_RATIO:-0.2}"
export GRPO_BETA="${GRPO_BETA:-0.04}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-50}"

export EVAL_TP="${EVAL_TP:-1}"
export EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
export EVAL_GPU_MEM_UTIL="${EVAL_GPU_MEM_UTIL:-0.88}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
export EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-12000}"

rendezvous_dir="${RENDEZVOUS_ROOT}/${RENDEZVOUS_ID}"
rollout_pid=""
rollout_hosts=()
rollout_ports=()

worker_log_dir="${MULTINODE_WORKER_LOG_DIR:-${artifact_root}/logs/multinode_worker_logs}"
mkdir -p "$worker_log_dir" "$mask_dir" "$eval_out" "${ckpt_root}/${run_name}" 2>/dev/null || true
worker_log="${worker_log_dir}/${run_name}_rank-${NODE_RANK:-unknown}_$(hostname)_$(date +%Y%m%d-%H%M%S).log"
touch "$worker_log" 2>/dev/null || true
if [ -w "$worker_log" ]; then
    exec > >(tee -a "$worker_log") 2>&1
fi

log() {
    echo "[4b-2node-worker][$(date '+%F %T')][host=$(hostname)][rank=${NODE_RANK:-?}] $*"
}

cleanup_rollout() {
    if [ "${CLEANUP_ROLLOUT_ON_EXIT:-false}" = "true" ] && [ -n "$rollout_pid" ]; then
        if kill -0 "$rollout_pid" 2>/dev/null; then
            log "stopping rollout server pid=${rollout_pid}"
            kill "$rollout_pid" 2>/dev/null || true
            wait "$rollout_pid" 2>/dev/null || true
        fi
    fi
}
trap cleanup_rollout EXIT

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

rollout_ready() {
    local host="$1"
    local port="$2"
    "$PYTHON_BIN" - "$host" "$port" <<'PY'
import sys
import urllib.request

host, port = sys.argv[1], sys.argv[2]
for path in ("/get_engine_type/", "/health/"):
    url = f"http://{host}:{port}{path}"
    try:
        if path == "/get_engine_type/":
            req = urllib.request.Request(url, data=b"", method="POST")
        else:
            req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if 200 <= resp.status < 300:
                raise SystemExit(0)
    except Exception:
        pass
raise SystemExit(1)
PY
}

wait_for_local_rollout() {
    local attempts=$((ROLLOUT_READY_TIMEOUT_SEC / ROLLOUT_READY_INTERVAL_SEC))
    if [ "$attempts" -lt 1 ]; then
        attempts=1
    fi
    for _ in $(seq 1 "$attempts"); do
        if rollout_ready 127.0.0.1 "$VLLM_SERVER_PORT"; then
            return 0
        fi
        if ! kill -0 "$rollout_pid" 2>/dev/null; then
            wait "$rollout_pid" 2>/dev/null || true
            echo "ERROR: rollout server exited before becoming ready." >&2
            return 1
        fi
        sleep "$ROLLOUT_READY_INTERVAL_SEC"
    done
    echo "ERROR: rollout server did not become ready within ${ROLLOUT_READY_TIMEOUT_SEC}s." >&2
    return 1
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
        echo "ROLLOUT_HOST=${advertise_host}"
        echo "ROLLOUT_PORT=${VLLM_SERVER_PORT}"
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

wait_for_train_terminal_phase() {
    local start now done_count failed_count
    start="$(date +%s)"
    while true; do
        failed_count="$(count_cluster_phase train_failed)"
        if [ "$failed_count" -gt 0 ]; then
            log "detected train_failed marker: ${failed_count}/${NNODES}"
            return 1
        fi
        done_count="$(count_cluster_phase train_done)"
        if [ "$done_count" -eq "$NNODES" ]; then
            return 0
        fi
        now="$(date +%s)"
        if [ $((now - start)) -gt "$RENDEZVOUS_TIMEOUT_SEC" ]; then
            echo "ERROR: training terminal rendezvous timed out; train_done=${done_count}/${NNODES} train_failed=${failed_count}/${NNODES}" >&2
            ls -la "$rendezvous_dir" >&2 || true
            return 1
        fi
        log "waiting for training terminal phase: train_done=${done_count}/${NNODES} train_failed=${failed_count}/${NNODES}"
        sleep 10
    done
}

watch_remote_train_failure() {
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

wait_for_eval_terminal_phase() {
    local start now done_count failed_count
    start="$(date +%s)"
    while true; do
        failed_count="$(count_cluster_phase eval_failed)"
        if [ "$failed_count" -gt 0 ]; then
            log "detected eval_failed marker: ${failed_count}/${NNODES}"
            return 1
        fi
        done_count="$(count_cluster_phase eval_done)"
        if [ "$done_count" -eq "$NNODES" ]; then
            return 0
        fi
        now="$(date +%s)"
        if [ $((now - start)) -gt "$RENDEZVOUS_TIMEOUT_SEC" ]; then
            echo "ERROR: eval terminal rendezvous timed out; eval_done=${done_count}/${NNODES} eval_failed=${failed_count}/${NNODES}" >&2
            ls -la "$rendezvous_dir" >&2 || true
            return 1
        fi
        log "waiting for eval terminal phase: eval_done=${done_count}/${NNODES} eval_failed=${failed_count}/${NNODES}"
        sleep 10
    done
}

wait_for_cluster_phase() {
    local phase="$1"
    local start now ready idx file
    start="$(date +%s)"
    while true; do
        ready=0
        for idx in $(seq 0 $((NNODES - 1))); do
            file="${rendezvous_dir}/node_${idx}.${phase}"
            if [ -f "$file" ]; then
                ready=$((ready + 1))
            fi
        done
        if [ "$ready" -eq "$NNODES" ]; then
            return 0
        fi
        now="$(date +%s)"
        if [ $((now - start)) -gt "$RENDEZVOUS_TIMEOUT_SEC" ]; then
            echo "ERROR: rendezvous timed out after ${RENDEZVOUS_TIMEOUT_SEC}s; phase=${phase} ready=${ready}/${NNODES}" >&2
            ls -la "$rendezvous_dir" >&2 || true
            return 1
        fi
        log "waiting for cluster phase '${phase}': ${ready}/${NNODES}"
        sleep 10
    done
}

build_cluster_env() {
    local idx file host port
    rollout_hosts=()
    rollout_ports=()
    for idx in $(seq 0 $((NNODES - 1))); do
        file="${rendezvous_dir}/node_${idx}.env"
        host="$(grep -E '^ROLLOUT_HOST=' "$file" | tail -n 1 | cut -d= -f2-)"
        port="$(grep -E '^ROLLOUT_PORT=' "$file" | tail -n 1 | cut -d= -f2-)"
        rollout_hosts+=("$host")
        rollout_ports+=("$port")
    done
    MASTER_ADDR="$(grep -E '^POD_IP=' "${rendezvous_dir}/node_0.env" | tail -n 1 | cut -d= -f2-)"
    VLLM_SERVER_HOSTS="${rollout_hosts[*]}"
    VLLM_SERVER_PORTS="${rollout_ports[*]}"
    export MASTER_ADDR VLLM_SERVER_HOSTS VLLM_SERVER_PORTS
}

check_remote_rollouts() {
    local idx start now
    for idx in "${!rollout_hosts[@]}"; do
        start="$(date +%s)"
        while true; do
            log "checking rollout server ${rollout_hosts[$idx]}:${rollout_ports[$idx]}"
            if rollout_ready "${rollout_hosts[$idx]}" "${rollout_ports[$idx]}"; then
                break
            fi
            now="$(date +%s)"
            if [ $((now - start)) -gt "$ROLLOUT_READY_TIMEOUT_SEC" ]; then
                echo "ERROR: remote rollout server ${rollout_hosts[$idx]}:${rollout_ports[$idx]} did not become reachable within ${ROLLOUT_READY_TIMEOUT_SEC}s." >&2
                return 1
            fi
            sleep "$ROLLOUT_READY_INTERVAL_SEC"
        done
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

split_words() {
    local raw="$1"
    raw="${raw//,/ }"
    # shellcheck disable=SC2206
    SPLIT_WORDS_RESULT=($raw)
}

export NODE_RANK="$(infer_rank)"
if [ "$NODE_RANK" -ge "$NNODES" ]; then
    echo "ERROR: inferred NODE_RANK=${NODE_RANK}, but NNODES=${NNODES}" >&2
    exit 1
fi

pod_ip="$(get_primary_ip)"
advertise_host="${MULTINODE_ADVERTISE_HOST:-$pod_ip}"

log "cwd=$PWD"
log "python=${PYTHON_BIN}"
log "swift=${SWIFT_BIN}"
log "run_name=${run_name}"
log "ckpt_root=${ckpt_root}"
log "artifact_root=${artifact_root}"
log "model_path=${model_path}"
log "teacher_path=${teacher_path}"
log "rollout_model_path=${rollout_model_path}"
log "rendezvous_dir=${rendezvous_dir}"
log "nnodes=${NNODES} nproc_per_node=${NPROC_PER_NODE} node_rank=${NODE_RANK}"
log "pod_ip=${pod_ip} advertise_host=${advertise_host}"
log "train_cuda=${TRAIN_CUDA_VISIBLE_DEVICES} rollout_cuda=${ROLLOUT_CUDA_VISIBLE_DEVICES}"
log "per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} grad_acc=${GRADIENT_ACCUMULATION_STEPS} lr=${LR}"
log "sdpo_lambda=${SDPO_LAMBDA} sdpo_grpo_failed_weight=${SDPO_GRPO_FAILED_WEIGHT:-1.0} sdpo_target=${SDPO_TARGET} sdpo_hint_source=${SDPO_HINT_SOURCE} sdpo_sibling_select_metric=${SDPO_SIBLING_SELECT_METRIC} sdpo_sibling_fallback=${SDPO_SIBLING_FALLBACK} sdpo_teacher_refresh_mode=${SDPO_TEACHER_REFRESH_MODE:-fixed} sdpo_teacher_refresh_step=${SDPO_TEACHER_REFRESH_STEP} grpo_beta=${GRPO_BETA} num_generations=${NUM_GENERATIONS}"
log "sdpo_teacher_refresh_metric warmup=${SDPO_TEACHER_REFRESH_WARMUP:-80} window=${SDPO_TEACHER_REFRESH_WINDOW:-50} check_interval=${SDPO_TEACHER_REFRESH_CHECK_INTERVAL:-10} min_iou_improve=${SDPO_TEACHER_REFRESH_MIN_IOU_IMPROVE:-0.01} max_failed=${SDPO_TEACHER_REFRESH_MAX_FAILED:-0.55} min_sdpo_loss=${SDPO_TEACHER_REFRESH_MIN_SDPO_LOSS:-0.02} max_kl=${SDPO_TEACHER_REFRESH_MAX_KL:-0.30} max_refreshes=${SDPO_TEACHER_REFRESH_MAX_REFRESHES:-1} cooldown_steps=${SDPO_TEACHER_REFRESH_COOLDOWN_STEPS:-0}"
log "sdpo_teacher_refresh_metric_ewma short_window=${SDPO_TEACHER_REFRESH_SHORT_WINDOW:-20} long_window=${SDPO_TEACHER_REFRESH_LONG_WINDOW:-80} ewma_alpha=${SDPO_TEACHER_REFRESH_EWMA_ALPHA:-0.10} consecutive_checks=${SDPO_TEACHER_REFRESH_CONSECUTIVE_CHECKS:-2} min_short_long_iou_gain=${SDPO_TEACHER_REFRESH_MIN_SHORT_LONG_IOU_GAIN:-0.006} min_ewma_iou_gain=${SDPO_TEACHER_REFRESH_MIN_EWMA_IOU_GAIN:-0.008} max_iou05_drop=${SDPO_TEACHER_REFRESH_MAX_IOU05_DROP:-0.010}"
log "opsd_mask_mode=${OPSD_MASK_MODE} opsd_hint_mode=${OPSD_HINT_MODE} opsd_ema_decay=${OPSD_EMA_DECAY} opsd_gaussian_sigma_ratio=${OPSD_GAUSSIAN_SIGMA_RATIO} opsd_jitter_ratio=${OPSD_JITTER_RATIO}"
log "vllm_gpu_memory_util=${VLLM_GPU_MEMORY_UTIL} vllm_max_model_len=${VLLM_MAX_MODEL_LEN} vllm_max_num_seqs=${VLLM_MAX_NUM_SEQS:-unset} vllm_enforce_eager=${VLLM_ENFORCE_EAGER:-unset}"
log "save_steps=${SAVE_STEPS} save_total_limit=${SAVE_TOTAL_LIMIT} save_only_model=${save_only_model}"
log "tuner_type=${TUNER_TYPE}"
log "cache_root=${cache_root}"
log "torchinductor_cache=${TORCHINDUCTOR_CACHE_DIR}"
log "triton_cache=${TRITON_CACHE_DIR}"

if [ "${MULTINODE_WORKER_PROBE_ONLY:-false}" = "true" ]; then
    log "MULTINODE_WORKER_PROBE_ONLY=true; exiting before CUDA/model checks."
    exit 0
fi

mkdir -p "$rendezvous_dir"

if ! is_model_checkpoint_dir "$model_path"; then
    echo "ERROR: MODEL_PATH is not a usable model checkpoint: ${model_path}" >&2
    find "$model_path" -maxdepth 1 -type f -printf '%s %p\n' 2>/dev/null | sort -n | tail -n 40 >&2 || true
    exit 1
fi
if ! is_model_checkpoint_dir "$teacher_path"; then
    echo "ERROR: TEACHER_PATH is not a usable model checkpoint: ${teacher_path}" >&2
    exit 1
fi
if ! is_model_checkpoint_dir "$rollout_model_path"; then
    echo "ERROR: ROLLOUT_MODEL_PATH is not a usable model checkpoint: ${rollout_model_path}" >&2
    exit 1
fi
if [ ! -f "$train_jsonl" ]; then
    echo "ERROR: train jsonl not found: ${train_jsonl}" >&2
    exit 1
fi
if [ ! -f "$test_jsonl" ]; then
    echo "ERROR: test jsonl not found: ${test_jsonl}" >&2
    exit 1
fi

cuda_probe_interval="${CUDA_READY_INTERVAL_SEC:-10}"
cuda_probe_timeout="${CUDA_READY_TIMEOUT_SEC:-180}"
cuda_probe_start="$(date +%s)"
while true; do
    if "$PYTHON_BIN" - <<'PY'
import os
import torch

prefix = "[4b-2node-worker]"
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

"$PYTHON_BIN" -m py_compile \
    swift/rlhf_trainers/grpo_trainer.py \
    swift/rlhf_trainers/args_mixin.py \
    swift/cus_rewards/bbox_iou_reward.py \
    swift/rewards/orm.py

if rollout_ready 127.0.0.1 "$VLLM_SERVER_PORT"; then
    log "found existing local rollout server at 127.0.0.1:${VLLM_SERVER_PORT}"
else
    rollout_extra_args=()
    if [ -n "${VLLM_MAX_NUM_SEQS:-}" ]; then
        rollout_extra_args+=(--vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS")
    fi
    if [ -n "${VLLM_ENFORCE_EAGER:-}" ]; then
        rollout_extra_args+=(--vllm_enforce_eager "$VLLM_ENFORCE_EAGER")
    fi
    log "starting local rollout server on GPU ${ROLLOUT_CUDA_VISIBLE_DEVICES}, port ${VLLM_SERVER_PORT}, model ${rollout_model_path}"
    CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES}" \
    IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM}" \
    "$SWIFT_BIN" rollout \
        --model_type qwen3_vl \
        --model "$rollout_model_path" \
        --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTIL" \
        --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
        "${rollout_extra_args[@]}" \
        --vllm_data_parallel_size 1 \
        --host 0.0.0.0 \
        --port "$VLLM_SERVER_PORT" &
    rollout_pid=$!
    wait_for_local_rollout
fi

write_node_file "ready"
mark_phase "ready"
wait_for_cluster_phase "ready"
build_cluster_env
log "master_addr=${MASTER_ADDR}:${MASTER_PORT}"
log "all_rollout_hosts=${VLLM_SERVER_HOSTS}"
log "all_rollout_ports=${VLLM_SERVER_PORTS}"
check_remote_rollouts
mark_phase "train_ready"
wait_for_cluster_phase "train_ready"

split_words "$VLLM_SERVER_HOSTS"
rollout_hosts=("${SPLIT_WORDS_RESULT[@]}")
split_words "$VLLM_SERVER_PORTS"
rollout_ports=("${SPLIT_WORDS_RESULT[@]}")
if [ "${#rollout_hosts[@]}" -ne "${#rollout_ports[@]}" ]; then
    echo "ERROR: rollout hosts/ports length mismatch: hosts=${#rollout_hosts[@]} ports=${#rollout_ports[@]}" >&2
    exit 1
fi

generation_batch_size=$((NNODES * NPROC_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if [ $((generation_batch_size % NUM_GENERATIONS)) -ne 0 ]; then
    echo "ERROR: generation_batch_size=${generation_batch_size} must be divisible by NUM_GENERATIONS=${NUM_GENERATIONS}" >&2
    exit 1
fi
log "GRPO generation_batch_size=${generation_batch_size}; num_generations=${NUM_GENERATIONS}"

rlhf_vllm_extra_args=()
if [ -n "${VLLM_MAX_NUM_SEQS:-}" ]; then
    rlhf_vllm_extra_args+=(--vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS")
fi
if [ -n "${VLLM_ENFORCE_EAGER:-}" ]; then
    rlhf_vllm_extra_args+=(--vllm_enforce_eager "$VLLM_ENFORCE_EAGER")
fi

resume_args=()
if [ "${ALLOW_RESUME:-false}" = "true" ]; then
    resume_checkpoint="${RESUME_FROM_CHECKPOINT:-}"
    if [ -z "$resume_checkpoint" ] && [ "${AUTO_RESUME:-true}" = "true" ]; then
        resume_checkpoint=$(
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
        )
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
    "$SWIFT_BIN" rlhf \
    "${resume_args[@]}" \
    --rlhf_type grpo \
    --use_sdpo true \
    --sdpo_lambda "$SDPO_LAMBDA" \
    --sdpo_tau_good "$SDPO_TAU_GOOD" \
    --sdpo_tau_fail "$SDPO_TAU_FAIL" \
    --sdpo_delta "$SDPO_DELTA" \
    --sdpo_only_failed true \
    --sdpo_distill_scope "${SDPO_DISTILL_SCOPE:-failed}" \
    --sdpo_grpo_failed_weight "${SDPO_GRPO_FAILED_WEIGHT:-1.0}" \
    --sdpo_target "$SDPO_TARGET" \
    --sdpo_hint_source "$SDPO_HINT_SOURCE" \
    --sdpo_sibling_select_metric "$SDPO_SIBLING_SELECT_METRIC" \
    --sdpo_sibling_fallback "$SDPO_SIBLING_FALLBACK" \
    --sdpo_teacher_refresh_mode "${SDPO_TEACHER_REFRESH_MODE:-fixed}" \
    --sdpo_teacher_refresh_step "$SDPO_TEACHER_REFRESH_STEP" \
    --sdpo_teacher_refresh_warmup "${SDPO_TEACHER_REFRESH_WARMUP:-80}" \
    --sdpo_teacher_refresh_window "${SDPO_TEACHER_REFRESH_WINDOW:-50}" \
    --sdpo_teacher_refresh_check_interval "${SDPO_TEACHER_REFRESH_CHECK_INTERVAL:-10}" \
    --sdpo_teacher_refresh_min_iou_improve "${SDPO_TEACHER_REFRESH_MIN_IOU_IMPROVE:-0.01}" \
    --sdpo_teacher_refresh_max_failed "${SDPO_TEACHER_REFRESH_MAX_FAILED:-0.55}" \
    --sdpo_teacher_refresh_min_sdpo_loss "${SDPO_TEACHER_REFRESH_MIN_SDPO_LOSS:-0.02}" \
    --sdpo_teacher_refresh_max_kl "${SDPO_TEACHER_REFRESH_MAX_KL:-0.30}" \
    --sdpo_teacher_refresh_short_window "${SDPO_TEACHER_REFRESH_SHORT_WINDOW:-20}" \
    --sdpo_teacher_refresh_long_window "${SDPO_TEACHER_REFRESH_LONG_WINDOW:-80}" \
    --sdpo_teacher_refresh_ewma_alpha "${SDPO_TEACHER_REFRESH_EWMA_ALPHA:-0.10}" \
    --sdpo_teacher_refresh_consecutive_checks "${SDPO_TEACHER_REFRESH_CONSECUTIVE_CHECKS:-2}" \
    --sdpo_teacher_refresh_min_short_long_iou_gain "${SDPO_TEACHER_REFRESH_MIN_SHORT_LONG_IOU_GAIN:-0.006}" \
    --sdpo_teacher_refresh_min_ewma_iou_gain "${SDPO_TEACHER_REFRESH_MIN_EWMA_IOU_GAIN:-0.008}" \
    --sdpo_teacher_refresh_max_iou05_drop "${SDPO_TEACHER_REFRESH_MAX_IOU05_DROP:-0.010}" \
    --sdpo_teacher_refresh_max_refreshes "${SDPO_TEACHER_REFRESH_MAX_REFRESHES:-1}" \
    --sdpo_teacher_refresh_cooldown_steps "${SDPO_TEACHER_REFRESH_COOLDOWN_STEPS:-0}" \
    --opsd_mask_dir "$mask_dir" \
    --opsd_token_weight_mode "$OPSD_TOKEN_WEIGHT_MODE" \
    --opsd_non_digit_weight "$OPSD_NON_DIGIT_WEIGHT" \
    --opsd_max_digit_len "$OPSD_MAX_DIGIT_LEN" \
    --opsd_ema_decay "$OPSD_EMA_DECAY" \
    --opsd_mask_mode "$OPSD_MASK_MODE" \
    --opsd_zoom_ratio "$OPSD_ZOOM_RATIO" \
    --opsd_min_area_frac "$OPSD_MIN_AREA_FRAC" \
    --opsd_gaussian_sigma_ratio "$OPSD_GAUSSIAN_SIGMA_RATIO" \
    --opsd_hint_mode "$OPSD_HINT_MODE" \
    --opsd_hint_box_color "$OPSD_HINT_BOX_COLOR" \
    --opsd_jitter_ratio "$OPSD_JITTER_RATIO" \
    --model "$model_path" \
    --model_type qwen3_vl \
    --ref_model "$ref_model_path" \
    --ref_model_type qwen3_vl \
    --teacher_model "$teacher_path" \
    --teacher_model_type qwen3_vl \
    --tuner_type "$TUNER_TYPE" \
    --train_type "$TUNER_TYPE" \
    --dataset "$train_jsonl" \
    --reward_funcs bbox-geometry \
    --num_generations "$NUM_GENERATIONS" \
    --num_iterations "$NUM_ITERATIONS" \
    --temperature "$ROLLOUT_TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --beta "$GRPO_BETA" \
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
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --output_dir "${ckpt_root}/${run_name}" \
    --warmup_ratio "$WARMUP_RATIO" \
    --save_only_model "$save_only_model" \
    --log_completions true \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --dataset_num_proc "$DATASET_NUM_PROC" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --teacher_deepspeed "$TEACHER_DEEPSPEED_CONFIG" \
    --offload_teacher_model "$OFFLOAD_TEACHER_MODEL" \
    --attn_impl flash_attn \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host "${rollout_hosts[@]}" \
    --vllm_server_port "${rollout_ports[@]}" \
    --vllm_server_timeout "$ROLLOUT_READY_TIMEOUT_SEC" \
    --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTIL" \
    "${rlhf_vllm_extra_args[@]}" &
train_pid=$!
watch_remote_train_failure "$train_pid" &
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
if ! wait_for_train_terminal_phase; then
    mark_phase "train_failed"
    exit 1
fi
cleanup_rollout
rollout_pid=""

if [ "$NODE_RANK" = "0" ]; then
    latest_ckpt="$(find_latest_checkpoint)" || {
        echo "ERROR: no checkpoint found under ${ckpt_root}/${run_name}" >&2
        mark_phase "eval_failed"
        exit 1
    }
    log "latest_ckpt=${latest_ckpt}"
    if [ "${SKIP_EVAL:-false}" = "true" ]; then
        log "SKIP_EVAL=true; skip final greedy evaluation for ${latest_ckpt}"
    else
        set +e
        CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
        IMAGE_MAX_TOKEN_NUM="$IMAGE_MAX_TOKEN_NUM" \
        "$PYTHON_BIN" tools/evaluation/eval_student.py \
            --model "$latest_ckpt" \
            --test_jsonl "$test_jsonl" \
            --out_dir "${eval_out}/$(basename "$latest_ckpt")" \
            --tp "$EVAL_TP" \
            --max_model_len "$EVAL_MAX_MODEL_LEN" \
            --gpu_mem_util "$EVAL_GPU_MEM_UTIL" \
            --eval_batch_size "$EVAL_BATCH_SIZE" \
            --max_new_tokens 128 \
            --seed 42
        eval_status=$?

        if [ "$eval_status" -eq 0 ]; then
            "$PYTHON_BIN" - <<PY
import json
import os

summary_path = os.path.join("${eval_out}", "$(basename "${latest_ckpt}")", "summary.json")
with open(summary_path) as f:
    metrics = json.load(f)
print("[4b-2node-worker] summary_path=" + summary_path)
print("[4b-2node-worker] IoU@0.5=%.4f" % metrics["IoU@0.5"])
print("[4b-2node-worker] mIoU=%.4f IoU@0.7=%.4f parse_rate=%.4f valid_rate=%.4f" % (
    metrics["mIoU"], metrics["IoU@0.7"], metrics["parse_rate"], metrics["valid_rate"]))
PY
            eval_status=$?
        fi
        set -e
        if [ "$eval_status" -ne 0 ]; then
            log "final greedy evaluation failed with status=${eval_status}"
            mark_phase "eval_failed"
            exit "$eval_status"
        fi
    fi
fi

mark_phase "eval_done"
if ! wait_for_eval_terminal_phase; then
    exit 1
fi
log "completed"
