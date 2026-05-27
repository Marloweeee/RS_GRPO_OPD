"""
教师模型在不同 hint 模式下的 zero-shot bbox grounding 评测。

输入: rs_test.jsonl (我们刚转换出来的 RRSIS-D 测试集)
输出: 每个 hint mode 的 mIoU / IoU@0.5 / IoU@0.7 / parse_rate / digit-token 近似熵

5 种模式 (由弱到强):
    1. none         原图 + 无任何 hint 文本 (= student 起点)
    2. soft_window  方案 A: 高斯衰减 + 不画框 + 文本提示"in brighter region"
    3. jitter_box   方案 C: 原图 + 抖动品红框 + 文本提示"approximately within magenta box"
    4. hint_legacy  老 GUI 行为: 高斯衰减 + 画绿 GT 框 + 文本提示"within green rectangle"
    5. gt           最强 oracle: hint_legacy 图 + 文本里直接给 norm-1000 GT 坐标 (用作上界)
"""

import argparse
import math
import os
import random
import sys
import time
from collections import defaultdict

import json
import numpy as np
from PIL import Image, ImageDraw

from swift.custom_utils.format_func import extract_bbox  # noqa: E402

sys.path.insert(0, '/data/codes/gui_grounding/GUI-SD-code-main')

SYSTEM_PROMPT = 'You are a helpful assistant.'


def _gaussian_alpha(W, H, x1, y1, x2, y2, sigma_ratio=1.5, min_area_frac=0.1):
    bbox_w, bbox_h = x2 - x1, y2 - y1
    sigma = max(bbox_w, bbox_h) * sigma_ratio
    min_sigma = min(W, H) * math.sqrt(min_area_frac)
    sigma = max(sigma, min_sigma)
    xs = np.arange(W)[None, :]
    ys = np.arange(H)[:, None]
    dx = np.maximum(x1 - xs, 0) + np.maximum(xs - x2, 0)
    dy = np.maximum(y1 - ys, 0) + np.maximum(ys - y2, 0)
    dist = np.sqrt(dx.astype(np.float64)**2 + dy.astype(np.float64)**2)
    return np.exp(-dist**2 / (2 * sigma**2)).astype(np.float32)


def render_image(orig_img: Image.Image,
                 bbox_pixel,
                 mode: str,
                 jitter_ratio: float = 0.2,
                 hint_box_color: str = 'magenta'):
    x1, y1, x2, y2 = bbox_pixel
    W, H = orig_img.size
    bbox_w, bbox_h = x2 - x1, y2 - y1

    if mode == 'none':
        return orig_img.copy()

    if mode == 'soft_window':
        alpha = _gaussian_alpha(W, H, x1, y1, x2, y2)[:, :, None]
        arr = np.array(orig_img, dtype=np.float32)
        return Image.fromarray((arr * alpha).astype(np.uint8))

    if mode == 'jitter_box':
        res = orig_img.copy()
        jw = max(1, int(bbox_w * jitter_ratio))
        jh = max(1, int(bbox_h * jitter_ratio))
        jx1 = max(0, min(W - 1, x1 + random.randint(-jw, jw)))
        jy1 = max(0, min(H - 1, y1 + random.randint(-jh, jh)))
        jx2 = max(jx1 + 1, min(W, x2 + random.randint(-jw, jw)))
        jy2 = max(jy1 + 1, min(H, y2 + random.randint(-jh, jh)))
        lw = 5
        ImageDraw.Draw(res).rectangle([jx1 - lw, jy1 - lw, jx2 + lw, jy2 + lw], outline=hint_box_color, width=lw)
        return res

    if mode in ('hint_legacy', 'gt'):
        alpha = _gaussian_alpha(W, H, x1, y1, x2, y2)[:, :, None]
        arr = np.array(orig_img, dtype=np.float32)
        res = Image.fromarray((arr * alpha).astype(np.uint8))
        lw = 5
        ImageDraw.Draw(res).rectangle([x1 - lw, y1 - lw, x2 + lw, y2 + lw], outline='green', width=lw)
        return res

    raise ValueError(f'unknown mode: {mode}')


def build_user_text(base_text: str, mode: str, hint_box_color: str, gt_norm):
    if mode == 'none':
        return base_text
    if mode == 'soft_window':
        return base_text + ' Hint: The target is in the brighter (un-dimmed) region of the image.'
    if mode == 'jitter_box':
        return base_text + f' Hint: The target is approximately within the {hint_box_color} box.'
    if mode == 'hint_legacy':
        return base_text + ' Hint: The answer is located within the green rectangle.'
    if mode == 'gt':
        nx1, ny1, nx2, ny2 = gt_norm
        gt_str = '{"bbox_2d": [' + f'{nx1},{ny1},{nx2},{ny2}' + ']}'
        return base_text + f' Hint: The exact answer is {gt_str}.'
    raise ValueError(f'unknown mode: {mode}')


def xyxy_norm1000(bbox_pixel, image_size):
    W, H = image_size
    return [
        int(bbox_pixel[0] / W * 1000),
        int(bbox_pixel[1] / H * 1000),
        int(bbox_pixel[2] / W * 1000),
        int(bbox_pixel[3] / H * 1000),
    ]


def iou_xyxy(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    u = aa + bb - inter
    return inter / u if u > 0 else 0.0


def approx_entropy_from_logprob_dict(d):
    """vLLM 返回 {token_id: Logprob}. 在 top-K 上重归一化后算 entropy (低估真值，但模式间可比)。"""
    if not d:
        return 0.0
    logps = [lp.logprob for lp in d.values()]
    ps = [math.exp(lp) for lp in logps]
    s = sum(ps)
    if s <= 0:
        return 0.0
    ps = [p / s for p in ps]
    return -sum(p * math.log(p + 1e-12) for p in ps)


def load_test(jsonl_path, max_samples):
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            ap = d['additional_paras']
            if isinstance(ap, str):
                ap = json.loads(ap)
            image_size = ap['image_size']
            gt_pixel = d['solution']['arguments']['coordinate']
            # user 原文（去掉 assistant 段，仅留 ref 文本）
            user_text = d['messages'][1]['content']
            samples.append({
                'image': d['images'],
                'image_size': image_size,
                'gt_pixel': gt_pixel,
                'gt_norm': xyxy_norm1000(gt_pixel, image_size),
                'user_text': user_text,
                'category': ap.get('category', ''),
                'ref_id': ap.get('ref_id', ''),
            })
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_jsonl', default='/data/codes/gui_grounding/data/rs_sub_dir/rs_test.jsonl')
    ap.add_argument('--model', default='/mnt/jfs/copilot/yhl/checkpoint/opensource/Qwen3-VL-8B-Instruct')
    ap.add_argument('--out_dir', default='/data/codes/gui_grounding/data/rs_sub_dir/eval_teacher_hint')
    ap.add_argument('--max_samples', type=int, default=0, help='0 = all')
    ap.add_argument('--tp', type=int, default=8)
    ap.add_argument('--max_model_len', type=int, default=12000)
    ap.add_argument('--gpu_mem_util', type=float, default=0.85)
    ap.add_argument('--logprobs_topk', type=int, default=20)
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--jitter_ratio', type=float, default=0.2)
    ap.add_argument('--hint_box_color', type=str, default='magenta')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--modes', nargs='+', default=['none', 'soft_window', 'jitter_box', 'hint_legacy', 'gt'])
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    samples = load_test(args.test_jsonl, args.max_samples)
    print(f'[eval] loaded {len(samples)} samples from {args.test_jsonl}', flush=True)

    print(f'[eval] booting vLLM (model={args.model}, tp={args.tp}) ...', flush=True)
    os.environ.setdefault('IMAGE_MAX_TOKEN_NUM', '10000')
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        limit_mm_per_prompt={'image': 1},
        dtype='bfloat16',
        trust_remote_code=True,
        enforce_eager=False,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = processor.tokenizer
    digit_token_ids = set(tokenizer.encode(str(i), add_special_tokens=False)[-1] for i in range(10))

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        logprobs=args.logprobs_topk,
        seed=args.seed,
    )

    all_results = {}

    for mode in args.modes:
        print(f'\n[eval] === mode: {mode} ===', flush=True)
        t0 = time.time()
        reqs = []
        for s in samples:
            img = Image.open(s['image']).convert('RGB')
            rendered = render_image(
                img, s['gt_pixel'], mode, jitter_ratio=args.jitter_ratio, hint_box_color=args.hint_box_color)
            user_text = build_user_text(s['user_text'], mode, args.hint_box_color, s['gt_norm'])
            messages = [
                {
                    'role': 'system',
                    'content': SYSTEM_PROMPT
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image'
                        },
                        {
                            'type': 'text',
                            'text': user_text
                        },
                    ]
                },
            ]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            reqs.append({'prompt': prompt, 'multi_modal_data': {'image': rendered}})

        outputs = llm.generate(reqs, sampling_params=sp)

        ious, parse_ok = [], 0
        digit_ent_list, all_ent_list = [], []
        per_sample = []
        for s, out in zip(samples, outputs):
            o = out.outputs[0]
            text = o.text
            pred_bbox = extract_bbox(text)
            if pred_bbox != 'no bbox':
                parse_ok += 1
                iou = iou_xyxy(pred_bbox, s['gt_norm'])
            else:
                iou = 0.0
            ious.append(iou)

            # 数字 token 上的近似熵
            tok_ids = o.token_ids
            if o.logprobs is not None:
                for tid, lp_dict in zip(tok_ids, o.logprobs):
                    ent = approx_entropy_from_logprob_dict(lp_dict)
                    all_ent_list.append(ent)
                    if tid in digit_token_ids:
                        digit_ent_list.append(ent)

            per_sample.append({
                'ref_id': s['ref_id'],
                'category': s['category'],
                'gt_norm': s['gt_norm'],
                'pred_bbox': pred_bbox if pred_bbox != 'no bbox' else None,
                'iou': iou,
                'text': text,
            })

        m_iou = sum(ious) / max(1, len(ious))
        iou_50 = sum(i > 0.5 for i in ious) / max(1, len(ious))
        iou_70 = sum(i > 0.7 for i in ious) / max(1, len(ious))
        parse_rate = parse_ok / max(1, len(samples))
        mean_digit_ent = (sum(digit_ent_list) / len(digit_ent_list)) if digit_ent_list else 0.0
        mean_all_ent = (sum(all_ent_list) / len(all_ent_list)) if all_ent_list else 0.0
        dt = time.time() - t0
        result = {
            'mode': mode,
            'n': len(samples),
            'time_s': round(dt, 1),
            'mIoU': round(m_iou, 4),
            'IoU@0.5': round(iou_50, 4),
            'IoU@0.7': round(iou_70, 4),
            'parse_rate': round(parse_rate, 4),
            'digit_entropy_topk': round(mean_digit_ent, 4),
            'all_entropy_topk': round(mean_all_ent, 4),
            'n_digit_tokens': len(digit_ent_list),
        }
        all_results[mode] = result
        print(f'[eval] {mode}: {json.dumps(result, indent=2)}', flush=True)

        with open(os.path.join(args.out_dir, f'{mode}.jsonl'), 'w') as f:
            for r in per_sample:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 总表
    print('\n\n=== SUMMARY ===', flush=True)
    cols = ['mode', 'mIoU', 'IoU@0.5', 'IoU@0.7', 'parse_rate', 'digit_entropy_topk', 'all_entropy_topk', 'time_s']
    print(' | '.join(f'{c:>17s}' if c != 'mode' else f'{c:<14s}' for c in cols))
    print('-' * 165)
    for mode in args.modes:
        r = all_results[mode]
        print(f'{r["mode"]:<14s} | ' + ' | '.join(f'{r[k]:>17.4f}' if isinstance(r[k], float) else f'{r[k]:>17}'
                                                  for k in cols[1:]))

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n[eval] saved summary to {args.out_dir}/summary.json', flush=True)


if __name__ == '__main__':
    main()
