"""
BBox IoU Reward — 给 RRSIS-D / 遥感指代定位用。

模型输出: <|box_start|>(nx1,ny1),(nx2,ny2)<|box_end|> 或 <bbox>...</bbox>
坐标空间:  norm-1000

奖励组合:
    R = w_format * 1[parseable] + w_iou * IoU(pred, gt_norm) + w_giou * GIoU(pred, gt_norm)
默认  w_format=0.1, w_iou=0.6, w_giou=0.3
GIoU 取值 [-1, 1]，clip 到 [0, 1] 后再加权，保持总 reward 区间稳定。
"""

import math
from typing import List
import json

from swift.custom_utils.format_func import extract_bbox
from swift.custom_utils.ground_func import bboxreal2norm


class ORM:
    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


def _iou_and_giou(pred, gt):
    """两个 [x1,y1,x2,y2] 框的 IoU 和 GIoU。"""
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt

    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    p_area = max(0, px2 - px1) * max(0, py2 - py1)
    g_area = max(0, gx2 - gx1) * max(0, gy2 - gy1)
    union = p_area + g_area - inter
    if union <= 0:
        return 0.0, 0.0
    iou = inter / union

    cx1 = min(px1, gx1)
    cy1 = min(py1, gy1)
    cx2 = max(px2, gx2)
    cy2 = max(py2, gy2)
    c_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)
    if c_area <= 0:
        return iou, iou
    giou = iou - (c_area - union) / c_area
    return iou, giou


class BBoxIoUReward(ORM):
    """R = w_format + w_iou·IoU + w_giou·max(0, GIoU)"""

    def __init__(self, w_format: float = 0.1, w_iou: float = 0.6, w_giou: float = 0.3):
        self.w_format = w_format
        self.w_iou = w_iou
        self.w_giou = w_giou

    def __call__(self, completions, solution, additional_paras, **kwargs) -> List[float]:
        rewards = []
        for predict_str, ground_truth, para in zip(completions, solution, additional_paras):
            if isinstance(para, str):
                para = json.loads(para)
            image_size = para['image_size']
            rewards.append(self._compute(predict_str, ground_truth, image_size))
        return rewards

    def _compute(self, predict_str: str, ground_truth: dict, image_size) -> float:
        pred = extract_bbox(predict_str)
        if pred == "no bbox":
            return 0.0

        try:
            gt_pixel = ground_truth['arguments']['coordinate']
            gt = bboxreal2norm(gt_pixel, image_size)  # → norm1000
        except Exception:
            return 0.0

        iou, giou = _iou_and_giou(pred, gt)
        # GIoU 可能为负，clip 到 [0,1] 防总分变负
        giou_clipped = max(0.0, giou)
        reward = self.w_format + self.w_iou * iou + self.w_giou * giou_clipped
        return round(reward, 4)


def _valid_norm_box(bbox):
    return (
        bbox != "no bbox"
        and 0 <= bbox[0] < bbox[2] <= 1000
        and 0 <= bbox[1] < bbox[3] <= 1000
    )


def _center_score(pred, gt):
    px = (pred[0] + pred[2]) / 2
    py = (pred[1] + pred[3]) / 2
    gx = (gt[0] + gt[2]) / 2
    gy = (gt[1] + gt[3]) / 2
    dist = math.sqrt((px - gx) ** 2 + (py - gy) ** 2)
    return max(0.0, 1.0 - dist / math.sqrt(2_000_000))


def _area_ratio_score(pred, gt):
    pa = max(0, pred[2] - pred[0]) * max(0, pred[3] - pred[1])
    ga = max(0, gt[2] - gt[0]) * max(0, gt[3] - gt[1])
    if pa <= 0 or ga <= 0:
        return 0.0
    ratio = pa / ga
    return min(ratio, 1.0 / ratio)


def _aspect_ratio_score(pred, gt):
    pw = max(1, pred[2] - pred[0])
    ph = max(1, pred[3] - pred[1])
    gw = max(1, gt[2] - gt[0])
    gh = max(1, gt[3] - gt[1])
    ratio = (pw / ph) / (gw / gh)
    return min(ratio, 1.0 / ratio)


class BBoxGeometryReward(ORM):
    """Stage1-aligned geometry reward for GRPO bbox rollouts."""

    def __call__(self, completions, solution, additional_paras, **kwargs) -> List[float]:
        rewards = []
        for predict_str, ground_truth, para in zip(completions, solution, additional_paras):
            if isinstance(para, str):
                para = json.loads(para)
            image_size = para['image_size']
            rewards.append(self._compute(predict_str, ground_truth, image_size))
        return rewards

    def _compute(self, predict_str: str, ground_truth: dict, image_size) -> float:
        pred = extract_bbox(predict_str)
        if pred == "no bbox":
            return 0.0

        try:
            gt_pixel = ground_truth['arguments']['coordinate']
            gt = bboxreal2norm(gt_pixel, image_size)
        except Exception:
            return 0.0

        parse_ok = 1.0
        valid_box = 1.0 if _valid_norm_box(pred) else 0.0
        iou = _iou_and_giou(pred, gt)[0] if valid_box else 0.0
        center = _center_score(pred, gt) if valid_box else 0.0
        area = _area_ratio_score(pred, gt) if valid_box else 0.0
        aspect = _aspect_ratio_score(pred, gt) if valid_box else 0.0
        reward = (
            0.10 * parse_ok
            + 0.10 * valid_box
            + 0.50 * iou
            + 0.15 * center
            + 0.10 * area
            + 0.05 * aspect
        )
        return round(reward, 4)


class BBoxIoUFormat(ORM):
    """单独的 format reward：可解析 = 1，否则 0。供日志/调试使用。"""

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        return [1.0 if extract_bbox(c) != "no bbox" else 0.0 for c in completions]
