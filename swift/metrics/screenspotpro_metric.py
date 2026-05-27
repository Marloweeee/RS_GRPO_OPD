# Copyright (c) Alibaba, Inc. and its affiliates.
import collections
import os
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from glob import glob
from typing import Dict, List, Literal

import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops.boxes import box_area
from tqdm import tqdm
from transformers.trainer_utils import EvalPrediction

from swift.custom_utils.format_func import extract_action, extract_ground
from swift.custom_utils.ground_func import (calculate_pred_norm_point, get_scroll_direction, pointnorm2real,
                                            pointreal2norm, qwen25_get_resize)
from swift.utils import read_from_jsonl


def compute_screenspotpro(data_list, metric):
    corr_action = 0
    total_eval = {
        'Dev': {
            'text': list(),
            'icon': list()
        },
        'Creative': {
            'text': list(),
            'icon': list()
        },
        'CAD': {
            'text': list(),
            'icon': list()
        },
        'Scientific': {
            'text': list(),
            'icon': list()
        },
        'Office': {
            'text': list(),
            'icon': list()
        },
        'OS': {
            'text': list(),
            'icon': list()
        }
    }

    num_wrong_format = 0
    num_action = len(data_list)
    for data in tqdm(data_list, total=len(data_list)):
        label, pred = data['solution'], data['response']
        raw_para = data['additional_paras']
        para = json.loads(raw_para) if isinstance(raw_para, str) else raw_para
        if 'ground_qwen3' in metric:
            pred_action = extract_ground(pred)
        elif 'navi_qwen3' in metric:
            pred_action = extract_action(pred)
        elif 'venus' in metric:
            try:
                pred_action = eval(pred)
            except Exception:
                pred_action = [0, 0]
        else:
            raise NotImplementedError

        gt_action = label

        if pred_action == 'no action':
            num_wrong_format += 1
            total_eval[para['group']][para['ui_type']].append(0)
        elif 'ground' in metric:
            pred_point = calculate_pred_norm_point(para['image_size'], pred_action, 'qwen3vl')
            pred_point = pointnorm2real(pred_point, para['image_size'])
            gt_bbox = gt_action['arguments']['coordinate']

        elif 'venus' in metric:
            pred_point = calculate_pred_norm_point(para['image_size'], pred_action, 'qwen3vl')
            pred_point = pointnorm2real(pred_point, para['image_size'])
            gt_bbox = gt_action['arguments']['coordinate']
            if gt_bbox[0] < pred_point[0] < gt_bbox[2] and gt_bbox[1] < pred_point[1] < gt_bbox[3]:
                corr_action += 1
                total_eval[para['group']][para['ui_type']].append(1)
            else:
                total_eval[para['group']][para['ui_type']].append(0)

        elif 'navi_qwen3' in metric:
            try:
                pred_point = calculate_pred_norm_point(para['image_size'], pred_action['arguments']['coordinate'],
                                                       'qwen3vl')
                pred_point = pointnorm2real(pred_point, para['image_size'])
            except Exception:
                pred_point = [0, 0]

            gt_bbox = gt_action['arguments']['coordinate']

            if gt_bbox[0] < pred_point[0] < gt_bbox[2] and gt_bbox[1] < pred_point[1] < gt_bbox[3]:
                corr_action += 1
                total_eval[para['group']][para['ui_type']].append(1)

            else:
                total_eval[para['group']][para['ui_type']].append(0)

    # === 计算每个 group 的 text/icon 准确率 ===
    metrics = {}
    for group, types in total_eval.items():
        group_metrics = {}
        for ui_type, results in types.items():
            if len(results) == 0:
                acc = 0.0
            else:
                acc = sum(results) / len(results) * 100
            group_metrics[ui_type] = f'{acc:.2f}'
        metrics[group] = group_metrics

    # === 计算总体指标 ===
    all_text = [x for g in total_eval.values() for x in g['text']]
    all_icon = [x for g in total_eval.values() for x in g['icon']]

    overall = {
        'text': f'{sum(all_text)/len(all_text)*100:.2f}' if all_text else '0.00',
        'icon': f'{sum(all_icon)/len(all_icon)*100:.2f}' if all_icon else '0.00',
        'Total Acc': f'{corr_action / num_action * 100:.2f}',
        'Total num': str(num_action),
        'Wrong format num': str(num_wrong_format),
    }

    print('\n=== Group-wise Accuracy ===')
    GROUP_ORDER = ['CAD', 'Dev', 'Creative', 'Scientific', 'Office', 'OS']
    for group in GROUP_ORDER:
        if group in metrics:
            vals = metrics[group]
            print(f"  {group}: text={vals['text']}%, icon={vals['icon']}%")
        else:
            print(f'  {group}: N/A')

    print('\n=== Overall ===')
    print(overall)

    return overall
