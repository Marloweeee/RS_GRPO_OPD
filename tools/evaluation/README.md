# 评估脚本

- `eval_student.py`：对单个 checkpoint 在遥感 jsonl 测试集上做评估。
  默认是 greedy 单次推理；设置 `--num_rollouts` 后可做 best-of-N oracle
  评估。
- `eval_teacher_hint.py`：评估 teacher/zero-shot 的不同 hint 模式，包括
  无 hint、soft window、jitter box、legacy hint 和 GT oracle hint。
- `eval_student_all.sh`：遍历一个 run 目录下的所有 `checkpoint-*`，逐个评估并
  输出紧凑的 checkpoint 对比表。
- `run_bestof8_eval_rjob.sh`：best-of-8 oracle 评估 worker。通常在 GPU rjob
  内使用，运行前设置 `MODEL`、`TEST_JSONL` 和 `OUT_DIR`。
- `run_latest_stage_refresh_eval_worker.sh`：自动查找 stage-refresh 实验下最新的
  完整 checkpoint，并在测试集上跑 greedy 评估。
