# 遥感目标定位实验复现与仓库说明

本文档记录当前仓库中与遥感目标定位相关的 OPSD/GRPO/SDPO 实验入口，方便后续上传
到 git 后交给其他人复现。仓库主体仍然是本地改造过的 `ms-swift`，训练和评估都会
优先使用当前目录下的 `swift/` 包，而不是系统里安装的上游 `ms-swift`。

## 1. 目录边界

- `swift/`：核心训练逻辑。GRPO+SDPO、OPSD teacher hint、bbox reward 等改动都在这里。
- `tools/data/`：遥感数据转换脚本。
- `tools/rollout/`：best-of-N rollout pool 和路由数据构建脚本。
- `tools/evaluation/`：学生模型、teacher hint、best-of-N oracle 评估脚本。
- `tools/training/`：可复用训练 worker 和 rjob launcher。
- `OPSD_Idea/`：实验计划、论文借鉴分析和阶段性结果记录。
- `GUI-SD_scripts/`：原 GUI-SD 训练/评估入口，保留用于兼容和历史实验复现。

以下内容不应进入 git：`output/`、`train_cache/`、`GUI-SD_scripts/train_cache/`、
`GUI-SD_scripts/output/`、`result/`、`output_files/`、`.omx/`、模型权重、checkpoint、
rollout cache、评估结果、rjob 日志和本地数据集。`.gitignore` 已经按这个边界处理。

## 2. 环境

默认环境路径：

```bash
export REPO_ROOT=/data/codes/gui_grounding/GUI-SD-code-main
export GUI_SD_ENV_ROOT=/data/codes/gui_grounding/conda_envs/GUI-SD
export PATH="${GUI_SD_ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
cd "${REPO_ROOT}"
```

大规模训练和评估需要使用 GPU rjob。长任务必须在新的 tmux 窗口里启动，避免直接占用
当前交互 shell。输出 checkpoint 和缓存默认写到 `/mnt/jfs/copilot/lhb/...`，不要写到
仓库本地目录。

## 3. 数据与模型路径

当前遥感全量数据默认路径：

```bash
TRAIN_JSONL=/data/codes/gui_grounding/data/rs_full/rs_train.jsonl
TEST_JSONL=/data/codes/gui_grounding/data/rs_full/rs_test.jsonl
```

当前 4B base 模型默认路径：

```bash
BASE_MODEL_PATH=/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-4B-Instruct
```

训练输出默认路径：

```bash
CKPT_ROOT=/mnt/jfs/copilot/lhb/checkpoint/rs/rs-sd
ARTIFACT_ROOT=/mnt/jfs/copilot/lhb/artifacts/rs/rs-sd
```

## 4. 核心训练入口

两节点 4B 端到端全量训练：

```bash
tmux new -s qwen3_4b_2node_rsfull
bash tools/training/launch_qwen3_4b_base_rsfull_grpo_sdpo_2node_rjob.sh
```

两阶段 4B 全量训练：

```bash
tmux new -s qwen3_4b_2stage_rsfull
bash tools/training/launch_qwen3_4b_rsfull_grpo_sdpo_2stage_2node_rjob.sh
```

两阶段脚本的策略是：

1. Stage 1 从 `Qwen3-VL-4B-Instruct` 开始，只训练 `80` step，保存 warm-start checkpoint。
2. Stage 2 自动加载 Stage 1 最新 checkpoint，并把 student/ref/teacher 都同步到该权重。
3. Stage 2 继续使用全量训练集训练一轮，结束后在全量测试集上输出 greedy 指标。

默认资源配置为 `gui_agent` 集群、两个 replica、每个 replica `8` GPU、`64` CPU、
`800000` MiB 内存。如果 `gui_agent` 提交失败，两阶段 launcher 会自动切到 `aos`。

## 5. 评估入口

单 checkpoint greedy 或 best-of-N oracle：

```bash
python tools/evaluation/eval_student.py \
  --model /path/to/checkpoint \
  --test_jsonl /data/codes/gui_grounding/data/rs_full/rs_test.jsonl \
  --out_dir /mnt/jfs/copilot/lhb/artifacts/rs/rs-sd/eval/manual_eval \
  --num_rollouts 1
```

遍历一个 run 下所有 checkpoint：

```bash
bash tools/evaluation/eval_student_all.sh <run_name>
```

## 6. 上传 git 前检查

建议在提交前执行：

```bash
find tools -name '*.sh' -print0 | xargs -0 -n1 bash -n
/data/codes/gui_grounding/conda_envs/GUI-SD/bin/python -m py_compile \
  tools/data/convert_rrsisd_to_jsonl.py \
  tools/evaluation/eval_student.py \
  tools/evaluation/eval_teacher_hint.py \
  tools/rollout/stage1_generate_rollout_pool.py \
  tools/rollout/stage2_build_routing_datasets.py \
  swift/rlhf_trainers/grpo_trainer.py \
  swift/rlhf_trainers/args_mixin.py \
  swift/cus_rewards/bbox_iou_reward.py \
  swift/rewards/orm.py
```

不要把 `/mnt` 下的模型权重、训练缓存、数据集和 rjob 日志复制进仓库。
