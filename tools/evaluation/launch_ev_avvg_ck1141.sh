#!/usr/bin/env bash
# Evaluate final ckpt-1141 of resume810 run on AVVG test set.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
rjob_name="${RJOB_NAME:-ev-avvg-ck1141-${timestamp}}"
log_root="${LOG_ROOT:-/data/codes/gui_grounding/data/logs/eval}"
submit_log="${LOG_PATH:-${log_root}/${rjob_name}.submit.log}"
worker_log="${WORKER_LOG:-${log_root}/${rjob_name}.worker.log}"

model="${MODEL:-/mnt/jfs/copilot/lhb/checkpoint/rs/avvg-mrpd/gui-sd-qwen3-4b-avvg_refgeo_sft_mrpd_metric_gaussian_e1_resume810_aos-20260530-105306/v0-20260530-105850/checkpoint-1141}"
eval_jsonl="${EVAL_JSONL:-/data/codes/gui_grounding/data/avvg_refgeo_full/avvg_test.jsonl}"
out_dir="${OUT_DIR:-/data/codes/gui_grounding/GUI-SD-code-main/output/ev-avvg-ck1141-${timestamp}}"

mkdir -p "$log_root" "$out_dir"
chmod +x "${repo_root}/tools/evaluation/run_single_ckpt_eval_worker.sh"

tmux new-session -d -s "ev-avvg-ck1141" "
    mkdir -p $(dirname "$submit_log") && \
    brainctl rjob launch \
        --cpu 28 --gpu 8 --memory 600000 \
        --group aos --charged-group=aos \
        --private-machine=group \
        --mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot \
        --positive-tags 'feature/gpfs=yes' \
        --predict-only && \
    brainctl launch \
        -d \
        --name '${rjob_name}' \
        --replica-restart=never --backoff-limit 1 --max-wait-duration 1h0m0s \
        --cpu 28 --gpu 8 --memory=600000 \
        --group aos --charged-group=aos \
        --private-machine=group \
        --mount=juicefs+s3://oss.i.shaipower.com/tkj-jfs:/mnt/jfs/copilot \
        --positive-tags 'feature/gpfs=yes' \
        --set-env 'REPO_ROOT=${repo_root}' \
        --set-env 'GUI_SD_ENV_ROOT=/data/codes/gui_grounding/conda_envs/GUI-SD' \
        --set-env 'MODEL=${model}' \
        --set-env 'EVAL_JSONL=${eval_jsonl}' \
        --set-env 'OUT_DIR=${out_dir}' \
        --set-env 'EVAL_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7' \
        --set-env 'TP=8' \
        --set-env 'MAX_MODEL_LEN=12000' \
        --set-env 'GPU_MEM_UTIL=0.88' \
        --set-env 'EVAL_BATCH_SIZE=512' \
        --set-env 'MAX_NEW_TOKENS=128' \
        --set-env 'WORKER_LOG=${worker_log}' \
        -- bash -lc '
            set -euo pipefail
            mkdir -p \$(dirname \"\$WORKER_LOG\")
            {
                echo \"[detached-single-split-worker] start=\$(date +%F\ %T\ %Z)\"
                echo \"[detached-single-split-worker] host=\$(hostname)\"
                bash \"\$REPO_ROOT/tools/evaluation/run_single_ckpt_eval_worker.sh\"
                status=\$?
                echo \"[detached-single-split-worker] status=\${status}\"
                echo \"[detached-single-split-worker] end=\$(date +%F\ %T\ %Z)\"
                exit \"\$status\"
            } > \"\$WORKER_LOG\" 2>&1
        ' 2>&1 | tee "${submit_log}"
    echo '[single-split-detached] submitted'
    echo '[single-split-detached] worker_log='"${worker_log}"
    echo '[single-split-detached] summary='"${out_dir}"'/summary.json'
"

echo "[launch-eval-ck1141] tmux session=ev-avvg-ck1141"
echo "[launch-eval-ck1141] submit_log=${submit_log}"
echo "[launch-eval-ck1141] worker_log=${worker_log}"
echo "[launch-eval-ck1141] out_dir=${out_dir}"
