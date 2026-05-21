#!/bin/bash
# Full RRSIS-D 1-epoch ablation: train with given mask_mode → eval all ckpts → exit container.
#
# Usage:
#   sh tools/training/train_eval_rs_ablation.sh <mask_mode> <run_suffix>
# Example:
#   sh tools/training/train_eval_rs_ablation.sh gaussian   rs_full_legacy
#   sh tools/training/train_eval_rs_ablation.sh jitter_box rs_full_jitter

set -u
mask_mode="${1:?usage: $0 <mask_mode> <run_suffix>}"
run_suffix="${2:?usage: $0 <mask_mode> <run_suffix>}"

# cd 到 GUI-SD-code-main，保证后续相对路径(tools/, train_cache/)正确
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
echo "[ablation] cwd = $(pwd)"

run_name="gui-sd-4b_student_${run_suffix}"
ckpt_root="/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd"
mkdir -p "${ckpt_root}/${run_name}"

rollout_host=${VLLM_SERVER_HOST:-127.0.0.1}
rollout_port=${VLLM_SERVER_PORT:-8192}

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

if rollout_ready; then
    echo "Found existing healthy rollout server at ${rollout_host}:${rollout_port}; skip starting a duplicate."
else
    CUDA_VISIBLE_DEVICES=7 \
    IMAGE_MAX_TOKEN_NUM=10000 \
    swift rollout \
        --model_type "qwen3_vl" \
        --model "/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-8B-Instruct" \
        --vllm_gpu_memory_utilization 0.8 \
        --vllm_max_model_len 20000 \
        --vllm_data_parallel_size 1 \
        --host "$rollout_host" \
        --port "$rollout_port" &

    echo "Waiting for rollout server at ${rollout_host}:${rollout_port} ..."
    for _ in $(seq 1 60); do
        if rollout_ready; then break; fi
        sleep 5
    done
    if ! rollout_ready; then
        echo "ERROR: rollout server ${rollout_host}:${rollout_port} is not ready; aborting." >&2
        exit 1
    fi
fi

nnodes=${NNODES:-1}
nproc_per_node=${NPROC_PER_NODE:-7}
node_rank=${NODE_RANK:-${RANK:-0}}
master_addr=${MASTER_ADDR:-127.0.0.1}
master_port=${MASTER_PORT:-29500}

# ============================================
# Training (full RRSIS-D, 1 epoch)
# ============================================
echo "============================================"
echo "[ablation] mask_mode=${mask_mode} run_name=${run_name}"
echo "============================================"

NNODES=$nnodes \
NODE_RANK=$node_rank \
MASTER_ADDR=$master_addr \
MASTER_PORT=$master_port \
NPROC_PER_NODE=$nproc_per_node \
IMAGE_MAX_TOKEN_NUM=10000 \
PYTHONFAULTHANDLER=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
swift rlhf \
    --rlhf_type gkd \
    --use_opsd true \
    --opsd_mask_dir "./train_cache/${run_name}" \
    --model "/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-8B-Instruct" \
    --model_type "qwen3_vl" \
    --teacher_model "/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-8B-Instruct" \
    --teacher_model_type "qwen3_vl" \
    --train_type full \
    --dataset "/data/codes/gui_grounding/data/rs_full/rs_train.jsonl" \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 2.5e-6 \
    --gradient_accumulation_steps 16 \
    --save_steps 20 \
    --save_total_limit 100 \
    --logging_steps 1 \
    --max_length 20000 \
    --max_completion_length 64 \
    --output_dir "${ckpt_root}/${run_name}" \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --log_completions true \
    --dataloader_num_workers 32 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --teacher_deepspeed zero3 \
    --attn_impl flash_attn \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host "$rollout_host" \
    --vllm_server_port "$rollout_port" \
    --opsd_token_weight_mode "uniform-entropy" \
    --opsd_non_digit_weight 0.05 \
    --opsd_max_digit_len 3 \
    --opsd_mask_mode "${mask_mode}" \
    --opsd_hint_mode "hint" \
    --opsd_hint_box_color "magenta" \
    --opsd_jitter_ratio 0.2 \
    --opsd_monitor_teacher true \
    --opsd_ema_decay 0.95

train_rc=$?
echo "============================================"
echo "[ablation] train finished with rc=${train_rc}"
echo "============================================"

# ============================================
# Cleanup: kill rollout so eval has clean GPU mem
# ============================================
pkill -9 -f "swift rollout" 2>/dev/null || true
pkill -9 -f vllm 2>/dev/null || true
sleep 8

# ============================================
# Eval all ckpts (on 349-sample subset test for direct comparison)
# ============================================
if [ "$train_rc" -eq 0 ]; then
    # 找最新的 v* 子目录
    latest_v=$(ls -1dt "${ckpt_root}/${run_name}"/v* 2>/dev/null | head -1)
    if [ -n "$latest_v" ]; then
        echo "[ablation] eval ${latest_v}"
        sh tools/evaluation/eval_student_all.sh "$latest_v" \
            "/data/codes/gui_grounding/data/rs_sub_dir/eval_${run_name}"
    else
        echo "[ablation] no v* dir under ${ckpt_root}/${run_name}, skipping eval" >&2
    fi
else
    echo "[ablation] train failed, skipping eval" >&2
fi

echo "[ablation] all done for ${run_name}"
