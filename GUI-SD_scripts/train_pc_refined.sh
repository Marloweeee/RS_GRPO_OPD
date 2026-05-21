#!/bin/bash
run_name="gui-sd-8b_pc_refined"
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

# rollout 不需要分布式变量；如果上次失败后已有 rollout server 留在 8192，不要重复启动占 GPU。
if rollout_ready; then
    echo "Found existing healthy rollout server at ${rollout_host}:${rollout_port}; skip starting a duplicate."
else
    mkdir -p "./train_cache/${run_name}"
    rollout_log="./train_cache/${run_name}/rollout.log"
    echo "Starting rollout server; log -> ${rollout_log}"
    CUDA_VISIBLE_DEVICES=7 \
    IMAGE_MAX_TOKEN_NUM=10000 \
    swift rollout \
        --model_type "qwen3_vl" \
        --model "/mnt/jfs/copilot/yhl/checkpoint/opensource/Qwen3-VL-8B-Instruct" \
        --vllm_gpu_memory_utilization 0.8 \
        --vllm_max_model_len 20000 \
        --vllm_data_parallel_size 1 \
        --host "$rollout_host" \
        --port "$rollout_port" >"$rollout_log" 2>&1 &
    rollout_pid=$!

    # Qwen3-VL-8B + vLLM 冷启动 (权重 + cudagraph capture) 通常 3-6 分钟，给 10 分钟上限。
    max_wait_iters=${ROLLOUT_WAIT_ITERS:-120}
    echo "Waiting up to $((max_wait_iters * 5))s for rollout at ${rollout_host}:${rollout_port} (pid=${rollout_pid}) ..."
    for i in $(seq 1 "$max_wait_iters"); do
        if rollout_ready; then
            echo "[t+$((i * 5))s] rollout ready."
            break
        fi
        if ! kill -0 "$rollout_pid" 2>/dev/null; then
            echo "ERROR: rollout process (pid=${rollout_pid}) exited before becoming ready; see ${rollout_log}" >&2
            tail -n 40 "$rollout_log" >&2 || true
            exit 1
        fi
        if [ $((i % 12)) -eq 0 ]; then
            echo "[t+$((i * 5))s] still waiting; last rollout log lines:"
            tail -n 3 "$rollout_log" || true
        fi
        sleep 5
    done

    if ! rollout_ready; then
        echo "ERROR: rollout server ${rollout_host}:${rollout_port} is not ready after $((max_wait_iters * 5))s; aborting training." >&2
        tail -n 40 "$rollout_log" >&2 || true
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
    --model "/mnt/jfs/copilot/yhl/checkpoint/opensource/Qwen3-VL-8B-Instruct" \
    --model_type "qwen3_vl" \
    --teacher_model "/mnt/jfs/copilot/yhl/checkpoint/opensource/Qwen3-VL-8B-Instruct" \
    --teacher_model_type "qwen3_vl" \
    --train_type full \
    --dataset "/data/codes/gui_grounding/data/pc_refined_data/pc_refined_swift.jsonl" \
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
    --max_completion_length 128 \
    --output_dir output/${run_name} \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --log_completions true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --teacher_deepspeed zero3 \
    --attn_impl flash_attn \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host "$rollout_host" \
    --vllm_server_port "$rollout_port" \
    --opsd_token_weight_mode "uniform-entropy" \
    --opsd_non_digit_weight 0.1 \
    --opsd_mask_mode 'gaussian' \
    --opsd_ema_decay 0.95
