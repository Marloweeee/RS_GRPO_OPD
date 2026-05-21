# Rollout 数据构建脚本

- `stage1_generate_rollout_pool.py`：对每个遥感 grounding 样本采样 K 条 bbox
  轨迹，计算 geometry reward，标注 good/failed/ambiguous 路由标签，并输出
  rollout pool 及统计摘要。
- `run_stage1_rollout_pool_rjob.sh`：Stage 1 K-rollout pool 生成任务的 worker
  入口，适合放到 GPU rjob 里运行。
- `stage2_build_routing_datasets.py`：基于 Stage 1 rollout pool 构建路由训练数据，
  包括 best-sibling SFT、GT-only SFT control 和 best-vs-worst DPO 数据。
  脚本也会写出小规模 sanity-check 子集，但它本身是正式的数据构建器。
- `run_stage2_dataset_build.sh`：从已有 rollout pool 构建 Stage 2 路由数据集的
  wrapper。
