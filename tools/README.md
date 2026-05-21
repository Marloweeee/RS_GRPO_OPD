# 工具脚本目录

这个目录只保留可复用的实验工具脚本。一次性的 smoke 脚本、过期的子集 worker、
以及 Python 字节码缓存已经清理掉。

## 子目录

- `data/`：数据集转换和预处理脚本。
- `evaluation/`：greedy 评估、best-of-N oracle 评估和 hint 评估入口。
- `rollout/`：离线 K 条轨迹采样、rollout pool 生成和路由数据构建脚本。
- `training/`：可复用的 GRPO/SDPO/OPSD 训练 worker，以及需要保留的历史消融入口。

大规模 GPU 任务仍然需要在新的 tmux 窗口中通过 `brainctl rjob` 或
`brainctl launch` 调度运行。这些脚本主要作为 rjob worker 入口使用，不建议在
当前交互 shell 里直接跑长任务。
