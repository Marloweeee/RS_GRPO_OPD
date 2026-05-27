import re

import json


def extract_action(content):
    try:
        output_text = json.loads(content.split('<tool_call>\n')[1].split('\n</tool_call>')[0])

        if 'arguments' not in output_text or 'action' not in output_text['arguments']:
            return 'no action'
        return output_text
    except Exception:
        return 'no action'


def cold_extract_action(content):
    try:
        output_text = eval(content.split('<tool_call>\n')[-1].split('\n</tool_call>')[0])

        if 'arguments' not in output_text or 'action' not in output_text['arguments']:
            return 'no action'
        return output_text
    except Exception:
        return 'no action'


def extract_ground(text):
    json_match = re.search(r'```json(.*?)```', text, re.DOTALL)

    if json_match:
        json_str = json_match.group(1).strip()
        try:
            data = json.loads(json_str)
            point = data[0]['point_2d']
            return point

        except Exception:
            return 'no action'
    else:
        return 'no action'


# 多种 bbox 输出格式（首选 Qwen3-VL Instruct 官方 JSON：{"bbox_2d": [x1,y1,x2,y2]}）
# 其它兼容: <|box_start|>(...)<|box_end|> / <bbox>...</bbox> / <tool_call>(x,y),(x,y) / 裸 (x,y),(x,y)
# 所有坐标都期望是 norm-1000 整数
_BOX_JSON_RE = re.compile(r'"bbox_2d"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]')
_BOX_QWEN_RE = re.compile(
    r'<\|box_start\|>\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?\s*,\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?\s*<\|box_end\|>')
_BOX_XML_RE = re.compile(
    r'<bbox>\s*\(?\s*(-?\d+)\s*[, ]\s*(-?\d+)\s*\)?\s*[, ]\s*\(?\s*(-?\d+)\s*[, ]\s*(-?\d+)\s*\)?\s*</bbox>')
_BOX_TOOLCALL_RE = re.compile(r'<tool_call>\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*,\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)')
_BOX_PAIR_RE = re.compile(r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*,\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)')


def extract_bbox(text):
    """解析模型输出里的 bbox，返回 [x1, y1, x2, y2] (int) 或字符串 'no bbox'。
       首选 Qwen3-VL JSON：{"bbox_2d":[x,y,x,y]}；兼容多种 fallback。"""
    if not isinstance(text, str):
        return 'no bbox'
    m = (
        _BOX_JSON_RE.search(text) or _BOX_QWEN_RE.search(text) or _BOX_XML_RE.search(text)
        or _BOX_TOOLCALL_RE.search(text) or _BOX_PAIR_RE.search(text))
    if not m:
        return 'no bbox'
    try:
        x1, y1, x2, y2 = (int(m.group(i)) for i in range(1, 5))
    except Exception:
        return 'no bbox'
    if x2 <= x1 or y2 <= y1:
        return 'no bbox'
    return [x1, y1, x2, y2]
