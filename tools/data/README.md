# 数据脚本

- `convert_rrsisd_to_jsonl.py`：把 RRSIS-D 的 `refs(unc).p` 和
  `instances.json` 转成 GUI-SD 训练格式，输出 `rs_train.jsonl`、
  `rs_val.jsonl`、`rs_test.jsonl`，也可以额外输出 `rs_all.jsonl`。
  assistant 答案使用 Qwen3-VL 的 norm-1000 bbox JSON 格式，同时在
  `solution` 中保留像素级 GT bbox，供训练和评估使用。
- `convert_rsvg_dior_to_jsonl.py`：把 RSVG-DIOR 的 XML 标注转成同构
  jsonl，每个 XML object/ref 对应一条样本，输出
  `rsvg_train.jsonl`、`rsvg_val.jsonl`、`rsvg_test.jsonl` 和
  `rsvg_all.jsonl`。
- `convert_avvg_refgeo_to_jsonl.py`：把 AVVG/refGeo 的
  `avvg_train.jsonl`、`avvg_test.jsonl` 转成 GUI-SD 训练格式，输出到
  `/data/codes/gui_grounding/data/avvg_refgeo_full/`。该脚本会把越界
  bbox 裁剪到图片范围内，在 `additional_paras` 中保留原始 bbox 和
  polygon，并额外写出 `convert_stats.json`。
