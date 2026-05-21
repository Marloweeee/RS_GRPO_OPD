#!/bin/bash
# OPSD 训练脚本 — RRSIS-D 遥感指代定位（bbox 任务）
# 关键差异（相对 train.sh）：
#   - dataset 指向 rs_train.jsonl（输出格式 <|box_start|>(x1,y1),(x2,y2)<|box_end|>，norm-1000）
#   - opsd_mask_mode=hint_legacy（实测 v2 IoU@0.5=0.771 最强，digit_entropy=0.805 仍软；jitter_box 第二 0.714）
#   - opsd_non_digit_weight=0.05（bbox 输出 12 个数字 token，降低结构 token 权重）
#   - opsd_monitor_teacher=true（记录 teacher digit_entropy / digit_p_at_gt / iou，用于诊断 hint 强度）
#   - max_completion_length=64（bbox 输出比 click 短）
run_name="gui-sd-4b_student_rs"
# 中间 checkpoint 写到 JFS 挂载点，避免打爆本地磁盘
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
        if rollout_ready; then
            break
        fi
        sleep 5
    done

    if ! rollout_ready; then
        echo "ERROR: rollout server ${rollout_host}:${rollout_port} is not ready; aborting training." >&2
        exit 1
    fi
fi

nnodes=${NNODES:-1}
nproc_per_node=${NPROC_PER_NODE:-7}
node_rank=${NODE_RANK:-${RANK:-0}}
master_addr=${MASTER_ADDR:-127.0.0.1}
master_port=${MASTER_PORT:-29500}

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
    --dataset "/data/codes/gui_grounding/data/rs_sub_dir/rs_train.jsonl" \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --learning_rate 2.5e-6 \
    --gradient_accumulation_steps 16 \
    --save_steps 10 \
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
    --opsd_mask_mode "gaussian" \
    --opsd_hint_mode "hint" \
    --opsd_hint_box_color "magenta" \
    --opsd_jitter_ratio 0.2 \
    --opsd_monitor_teacher true \
    --opsd_ema_decay 0.95
