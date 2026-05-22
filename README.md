# RS-GRPO-OPD：面向遥感目标定位的 GRPO + OPSD/SDPO 训练代码

本仓库是在 GUI-SD / ms-swift 基础上改造的遥感目标定位训练代码。核心目标是让
Qwen3-VL 在遥感 referring expression grounding 任务中输出更准的 bbox：

- 用 `GRPO` 一次 rollout 多个候选 bbox，并在组内做相对优势优化。
- 用 `bbox-geometry` reward 衡量候选框质量。
- 用 `SDPO/OPSD` 思路对低质量 rollout 引入带视觉 hint 的 teacher 分布约束。
- 使用 norm-1000 bbox 输出格式：`{"bbox_2d":[x1,y1,x2,y2]}`。

本文档只面向 **单机 8 张 RTX 4090 服务器**。默认训练方式是：

- `GPU 0-6`：7 卡分布式训练。
- `GPU 7`：本机 vLLM rollout server。
- 模型：推荐 `Qwen3-VL-4B-Instruct`。
- checkpoint、日志、cache 默认写到仓库外，避免污染 git 仓库。

多机训练和集群调度不是本 README 的目标。

## 1. 推荐目录

假设服务器上使用下面的目录：

```bash
/data/codes/gui_grounding/GUI-SD-code-main      # 本仓库
/data/models/Qwen3-VL-4B-Instruct              # base 模型
/data/datasets/rs_full/rs_train.jsonl          # 遥感训练集
/data/datasets/rs_full/rs_test.jsonl           # 遥感测试集
/data/runs/rs_grpo_opd                         # 训练输出、日志、评估结果
```

你可以换成自己的路径，启动脚本都支持环境变量覆盖。

## 2. 环境安装

建议使用 Python 3.10 和 CUDA 12 系列环境。

```bash
conda create -n GUI-SD python=3.10 -y
conda activate GUI-SD

cd /data/codes/gui_grounding/GUI-SD-code-main
pip install -r requirements.txt

# 安装训练必须组件：vLLM、DeepSpeed、Qwen-VL utils、TRL、Transformers 等。
# 这个脚本会安装一批推理/训练依赖，耗时较长，但最省心。
bash requirements/install_all.sh

# 确保 transformers 版本足够新，可以识别 Qwen3-VL。
pip install "transformers>=4.57,<4.58" "trl<0.25" -U

# 建议安装 flash-attn，4090 上显存和速度都会更稳。
# 如果编译失败，可以先跳过，训练时用 ATTN_IMPL=sdpa。
pip install flash-attn --no-build-isolation
```

确认环境能找到本仓库里的 `swift/`，而不是只用 pip 里的上游 ms-swift：

```bash
cd /data/codes/gui_grounding/GUI-SD-code-main
export PYTHONPATH="$PWD:${PYTHONPATH}"

python - <<'PY'
import torch
import swift
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), torch.cuda.device_count())
print("swift:", swift.__file__)
PY

swift --help | head
python - <<'PY'
import vllm
print("vllm:", vllm.__version__)
PY
```

正常情况下，`swift.__file__` 应该指向当前仓库下的 `swift` 目录。

## 3. 下载模型

推荐先把模型下载到本地磁盘，避免训练时边下边跑。

```bash
mkdir -p /data/models
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir /data/models/Qwen3-VL-4B-Instruct
```

训练时设置：

```bash
export MODEL_PATH=/data/models/Qwen3-VL-4B-Instruct
```

如果服务器网络稳定，也可以直接用 Hugging Face ID：

```bash
export MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct
```

但为了稳定复现，推荐本地路径。

## 4. 准备数据

训练脚本默认寻找：

```bash
$DATA_ROOT/rs_full/rs_train.jsonl
$DATA_ROOT/rs_full/rs_test.jsonl
```

因此最简单的目录是：

```bash
/data/datasets/rs_full/rs_train.jsonl
/data/datasets/rs_full/rs_test.jsonl
```

然后启动时设置：

```bash
export DATA_ROOT=/data/datasets
```

如果你手上是 RRSIS-D 原始数据，可以用转换脚本生成 jsonl：

```bash
python tools/data/convert_rrsisd_to_jsonl.py \
  --src_dir /path/to/RRSIS-D \
  --out_dir /data/datasets/rs_full \
  --image_prefix /path/to/RRSIS-D/JPEGImages \
  --also_all
```

转换后的每条样本包含：

- `messages`：Qwen3-VL 训练对话。
- `images`：图像路径。
- `solution.arguments.coordinate`：像素级 GT bbox。
- `additional_paras.image_size`：原图尺寸。

这些字段会被 reward、teacher hint 和评估脚本使用，不要随意删改。

## 5. 单机 8×4090 一键训练

推荐入口：

```bash
GUI-SD_scripts/train_8x4090.sh
```

首次运行：

```bash
conda activate GUI-SD
cd /data/codes/gui_grounding/GUI-SD-code-main

MODEL_PATH=/data/models/Qwen3-VL-4B-Instruct \
DATA_ROOT=/data/datasets \
RUN_ROOT=/data/runs/rs_grpo_opd \
bash GUI-SD_scripts/train_8x4090.sh
```

这个脚本会自动完成：

1. 检查 8 张 GPU。
2. 检查训练数据。
3. 设置 `PYTHONPATH`、`IMAGE_MAX_TOKEN_NUM` 和本地 cache。
4. 在 `GPU 7` 启动 vLLM rollout server。
5. 在 `GPU 0-6` 启动 `swift rlhf --rlhf_type grpo --use_sdpo true`。
6. 保存 checkpoint。
7. 训练结束后自动评估最新 checkpoint，并打印 `IoU@0.5`、`mIoU`、`IoU@0.7`。

长任务建议放在 tmux 中运行：

```bash
tmux new -s rs_grpo_4090

conda activate GUI-SD
cd /data/codes/gui_grounding/GUI-SD-code-main

MODEL_PATH=/data/models/Qwen3-VL-4B-Instruct \
DATA_ROOT=/data/datasets \
RUN_ROOT=/data/runs/rs_grpo_opd \
bash GUI-SD_scripts/train_8x4090.sh
```

查看实时日志：

```bash
tail -f /data/runs/rs_grpo_opd/logs/*.log
tail -f /data/runs/rs_grpo_opd/logs/*rollout*.log
```

查看显卡：

```bash
nvidia-smi
```

## 6. 默认训练配置

`train_8x4090.sh` 默认配置偏保守，目标是先在 8×4090 上稳定跑通。

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_PATH` | `Qwen/Qwen3-VL-4B-Instruct` | student 模型。推荐改成本地路径。 |
| `TEACHER_PATH` | `$MODEL_PATH` | SDPO teacher，默认 self-distillation。 |
| `REF_MODEL_PATH` | `$MODEL_PATH` | GRPO KL reference model。 |
| `TRAIN_CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6` | 训练用 7 张卡。 |
| `ROLLOUT_CUDA_VISIBLE_DEVICES` | `7` | rollout server 独占 1 张卡。 |
| `NPROC_PER_NODE` | `7` | 与训练 GPU 数一致。 |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `1` | 4090 24GB 上更稳。 |
| `GRADIENT_ACCUMULATION_STEPS` | `8` | 配合 7 卡得到 generation batch 56。 |
| `GENERATION_BATCH_SIZE` | `56` | 必须能被 `NUM_GENERATIONS=8` 整除。 |
| `NUM_GENERATIONS` | `8` | 每个 prompt 采样 8 个 bbox。 |
| `LR` | `1e-6` | 4B 全参 GRPO+SDPO 的保守学习率。 |
| `MAX_LENGTH` | `12000` | 4090 上比 20000 更稳。 |
| `VLLM_MAX_MODEL_LEN` | `12000` | rollout server 最大上下文。 |
| `VLLM_GPU_MEMORY_UTIL` | `0.70` | 避免 rollout 卡 OOM。 |
| `DEEPSPEED_CONFIG` | `zero3` | student/ref 使用 ZeRO-3。 |
| `TEACHER_DEEPSPEED_CONFIG` | `zero3` | teacher 使用 ZeRO-3。 |
| `SAVE_STEPS` | `250` | 每 250 step 保存一次。 |
| `SAVE_TOTAL_LIMIT` | `4` | 最多保留 4 个 checkpoint。 |

如果你的 4090 机器显存比较干净，可以尝试更快配置：

```bash
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=4 \
GENERATION_BATCH_SIZE=56 \
LR=1.5e-6 \
MODEL_PATH=/data/models/Qwen3-VL-4B-Instruct \
DATA_ROOT=/data/datasets \
RUN_ROOT=/data/runs/rs_grpo_opd_fast \
bash GUI-SD_scripts/train_8x4090.sh
```

如果出现 OOM，回到默认配置，或进一步降低：

```bash
MAX_LENGTH=8192 \
VLLM_MAX_MODEL_LEN=8192 \
VLLM_GPU_MEMORY_UTIL=0.60 \
EVAL_BATCH_SIZE=64 \
MODEL_PATH=/data/models/Qwen3-VL-4B-Instruct \
DATA_ROOT=/data/datasets \
RUN_ROOT=/data/runs/rs_grpo_opd_safe \
bash GUI-SD_scripts/train_8x4090.sh
```

如果 `flash_attn` 安装失败或运行报错，可以用 PyTorch SDPA：

```bash
ATTN_IMPL=sdpa \
MODEL_PATH=/data/models/Qwen3-VL-4B-Instruct \
DATA_ROOT=/data/datasets \
RUN_ROOT=/data/runs/rs_grpo_opd_sdpa \
bash GUI-SD_scripts/train_8x4090.sh
```

## 7. 输出位置

默认输出都在 `$RUN_ROOT` 下：

```bash
/data/runs/rs_grpo_opd/
├── artifacts/
│   ├── eval/
│   └── train_cache/
├── checkpoints/
│   └── rs-grpo-opd-qwen3vl4b-8x4090/
│       └── v0-*/
│           └── checkpoint-*/
└── logs/
    ├── *.log
    └── *rollout*.log
```

仓库中的 `.gitignore` 已经忽略 `output/`、`train_cache/`、checkpoint、日志和模型权重。
不要把 `$RUN_ROOT` 或模型目录复制进 git。

## 8. 手动评估

训练脚本默认在结束后评估最新 checkpoint。如果需要手动评估：

```bash
CKPT=/data/runs/rs_grpo_opd/checkpoints/rs-grpo-opd-qwen3vl4b-8x4090/v0-xxxx/checkpoint-xxx

CUDA_VISIBLE_DEVICES=0 IMAGE_MAX_TOKEN_NUM=10000 \
python tools/evaluation/eval_student.py \
  --model "$CKPT" \
  --test_jsonl /data/datasets/rs_full/rs_test.jsonl \
  --out_dir /data/runs/rs_grpo_opd/artifacts/eval/manual_checkpoint_xxx \
  --tp 1 \
  --max_model_len 12000 \
  --gpu_mem_util 0.75 \
  --eval_batch_size 128
```

输出文件：

```bash
summary.json       # mIoU、IoU@0.5、IoU@0.7、parse_rate、valid_rate
per_sample.jsonl   # 每个样本的预测 bbox、GT、IoU 和原始文本
```

查看核心指标：

```bash
cat /data/runs/rs_grpo_opd/artifacts/eval/manual_checkpoint_xxx/summary.json
```

如果要看 best-of-8 oracle 上限：

```bash
CUDA_VISIBLE_DEVICES=0 IMAGE_MAX_TOKEN_NUM=10000 \
python tools/evaluation/eval_student.py \
  --model "$CKPT" \
  --test_jsonl /data/datasets/rs_full/rs_test.jsonl \
  --out_dir /data/runs/rs_grpo_opd/artifacts/eval/bestof8_checkpoint_xxx \
  --tp 1 \
  --num_rollouts 8 \
  --rollout_temperature 0.7 \
  --top_p 0.95 \
  --max_model_len 12000 \
  --gpu_mem_util 0.75 \
  --eval_batch_size 64
```

## 9. 断点继续训练

如果训练中断，可以让脚本自动找当前 run 的最新 checkpoint：

```bash
AUTO_RESUME=true \
MODEL_PATH=/data/models/Qwen3-VL-4B-Instruct \
DATA_ROOT=/data/datasets \
RUN_ROOT=/data/runs/rs_grpo_opd \
bash GUI-SD_scripts/train_8x4090.sh
```

也可以手动指定：

```bash
RESUME_FROM_CHECKPOINT=/data/runs/rs_grpo_opd/checkpoints/rs-grpo-opd-qwen3vl4b-8x4090/v0-xxxx/checkpoint-xxx \
MODEL_PATH=/data/models/Qwen3-VL-4B-Instruct \
DATA_ROOT=/data/datasets \
RUN_ROOT=/data/runs/rs_grpo_opd \
bash GUI-SD_scripts/train_8x4090.sh
```

默认 `RESUME_ONLY_MODEL=true`，适合只保存模型权重的 checkpoint。

## 10. 常见问题

### 10.1 端口 8192 已被占用

如果之前的 rollout server 没退出，可以先查：

```bash
lsof -i :8192
```

换一个端口启动：

```bash
VLLM_SERVER_PORT=8292 \
MODEL_PATH=/data/models/Qwen3-VL-4B-Instruct \
DATA_ROOT=/data/datasets \
RUN_ROOT=/data/runs/rs_grpo_opd \
bash GUI-SD_scripts/train_8x4090.sh
```

### 10.2 rollout server OOM

降低 rollout 侧上下文和显存占用：

```bash
VLLM_MAX_MODEL_LEN=8192
VLLM_GPU_MEMORY_UTIL=0.60
```

### 10.3 训练 OOM

优先按顺序降低：

```bash
PER_DEVICE_TRAIN_BATCH_SIZE=1
MAX_LENGTH=8192
DEEPSPEED_CONFIG=zero3
TEACHER_DEEPSPEED_CONFIG=zero3
ATTN_IMPL=sdpa
```

如果仍然 OOM，可以尝试 teacher offload，但速度会明显下降：

```bash
OFFLOAD_TEACHER_MODEL=true
TEACHER_DEEPSPEED_CONFIG=zero3_offload
```

### 10.4 vLLM / Triton cache 报 stale file handle

`train_8x4090.sh` 默认把 cache 放到本机 `/tmp`：

```bash
CACHE_ROOT=/tmp/rs_grpo_opd_${USER}_rs-grpo-opd-qwen3vl4b-8x4090
```

如果你的 `/tmp` 空间不足，可以改到本地 SSD：

```bash
CACHE_ROOT=/data/cache/rs_grpo_opd
```

不要把这些 cache 放在不稳定的网络文件系统上。

### 10.5 训练速度很慢

先确认 7 张训练卡和 1 张 rollout 卡都在工作：

```bash
nvidia-smi
```

如果 rollout 卡利用率长期很低，通常是 rollout server 没启动成功或端口连接不对；
看 `logs/*rollout*.log`。如果训练卡利用率低，通常是数据读取慢或 batch 太小，可以尝试：

```bash
DATALOADER_NUM_WORKERS=8
DATASET_NUM_PROC=8
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
```

前提是显存不 OOM。

## 11. 关键代码位置

```bash
swift/rlhf_trainers/grpo_trainer.py       # GRPO + SDPO 主逻辑
swift/rlhf_trainers/args_mixin.py         # GRPO/SDPO 参数
swift/cus_rewards/bbox_iou_reward.py      # bbox reward 辅助
swift/rewards/orm.py                      # reward registry，包含 bbox-geometry
tools/data/convert_rrsisd_to_jsonl.py     # RRSIS-D 数据转换
tools/evaluation/eval_student.py          # greedy / best-of-N 评估
GUI-SD_scripts/train_8x4090.sh            # 单机 8×4090 一键训练入口
```

## 12. 最小成功标准

一次正常训练应该满足：

1. `GPU 7` 上 vLLM rollout server ready。
2. 日志出现 `Train: 0/...`，随后 `global_step/max_steps` 持续增加。
3. checkpoint 写入 `$RUN_ROOT/checkpoints/.../checkpoint-*`。
4. 训练结束后 `summary.json` 中有 `IoU@0.5`、`mIoU`、`IoU@0.7`。

看到这些后，说明单机 8×4090 流程已经跑通。

## 13. 参考

- GUI-SD: On-Policy Self-Distillation for GUI Grounding。
- Qwen3-VL-4B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- ms-swift: https://github.com/modelscope/ms-swift
- vLLM: https://github.com/vllm-project/vllm
