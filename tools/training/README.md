# 训练脚本

- `run_grpo_sdpo_worker.sh`：通用 GRPO+SDPO/OPSD worker。它会启动或复用 rollout
  server，按环境变量指定的 student/teacher/ref 路径训练，保存到配置的 checkpoint
  根目录，最后对最新 checkpoint 跑 greedy 评估。
- `run_qwen3_4b_base_rsfull_grpo_sdpo_e1_worker.sh`：从
  `/mnt/jfs/copilot/lhb/checkpoint/opensource/Qwen3-VL-4B-Instruct` 出发，在全量
  遥感训练集上跑一轮 GRPO+SDPO。默认把 student/ref/teacher 都设为 4B base，
  使用 `per_device_train_batch_size=2`、`gradient_accumulation_steps=4`、
  `learning_rate=1.5e-6`，用于比 8B baseline 更充分地利用显存。
- `run_qwen3_4b_base_rsfull_grpo_sdpo_2node_worker.sh`：4B 全量遥感 GRPO+SDPO
  的两节点 rjob worker。每个 replica 使用 7 张卡训练、1 张卡启动本地 rollout
  server，通过共享 rendezvous 目录拼出两个 vLLM server，并只在 `NODE_RANK=0`
  上做最终 greedy 评估。默认 `per_device_train_batch_size=4`、
  `gradient_accumulation_steps=2`、`learning_rate=2e-6`。
- `launch_qwen3_4b_base_rsfull_grpo_sdpo_2node_rjob.sh`：上面两节点 worker 的
  rjob 启动脚本。会先执行 `brainctl launch --predict-only` 检查资源，再用
  `-P 2 --replica-prefix` 提交两个 8 卡 replica；默认使用 `gui_agent` 集群、
  每个 replica 请求 `64` CPU、`800000` MiB 内存。
- `run_qwen3_4b_rsfull_grpo_sdpo_2stage_2node_worker.sh`：4B 全量遥感两阶段
  worker。Stage 1 从 4B base 只训练 `80` step 并跳过评估；Stage 2 自动读取
  Stage 1 最新 checkpoint，把 student/ref/teacher 都同步到该 checkpoint 后，
  再用全量训练集训练一轮并做最终 greedy 测试集评估。
- `launch_qwen3_4b_rsfull_grpo_sdpo_2stage_2node_rjob.sh`：两阶段实验的 rjob
  启动脚本。默认先在 `gui_agent` 上做资源预测和提交；如果提交失败，自动切到
  `aos` 重试。资源形状保持为每个 replica `8` GPU、`64` CPU、`800000` MiB
  内存。
- `run_base_grpo_srpo_rsfull_e1_worker.sh`：全量遥感数据一轮训练入口，从
  Qwen3-VL base 模型开始跑 GRPO+SDPO/SRPO-style routing 实验。
- `run_grpo_sdpo_rsfull_e1_worker.sh`：全量遥感数据一轮训练入口，从 legacy
  `checkpoint-80` warm-start 模型开始跑 GRPO+SDPO。
- `run_stage_refresh_base_rsfull_s800_worker.sh`：从全量 base-run 的 checkpoint
  继续 800 step，并把 student/ref/teacher 都同步到该阶段 checkpoint，用于验证
  teacher/ref stage-refresh 思路。
- `train_eval_rs_ablation.sh`：历史 OPSD/GKD mask/hint 消融入口。保留它是因为它
  记录并可复现 legacy `checkpoint-80` warm-start 的来源。
