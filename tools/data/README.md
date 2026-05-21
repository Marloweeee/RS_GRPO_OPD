# 数据脚本

- `convert_rrsisd_to_jsonl.py`：把 RRSIS-D 的 `refs(unc).p` 和
  `instances.json` 转成 GUI-SD 训练格式，输出 `rs_train.jsonl`、
  `rs_val.jsonl`、`rs_test.jsonl`，也可以额外输出 `rs_all.jsonl`。
  assistant 答案使用 Qwen3-VL 的 norm-1000 bbox JSON 格式，同时在
  `solution` 中保留像素级 GT bbox，供训练和评估使用。
