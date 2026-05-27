"""
单个 student ckpt 在 rs_test 上的评测。

输入: 一个 ckpt 目录 (HF-compatible 格式) + rs_test.jsonl
输出: <out_dir>/{summary.json, per_sample.jsonl}
指标: mIoU / IoU@0.5 / IoU@0.7 / parse_rate / mIoU|parsed

当 --num_rollouts > 1 时，同一个 prompt 一次性采样多个 bbox rollout，
按几何 reward 选出 best candidate。这个分支用于第一阶段的
geometry-rubric / best-of-N rollout 验证，不改变默认 greedy 评测行为。

System / user / assistant 模板严格对齐 convert_rrsisd_to_jsonl.py 的训练格式
（Qwen3-VL 官方 grounding cookbook: "Locate X, output its bbox coordinates using JSON format." → {"bbox_2d":[...]}）。
"""

import argparse
import math
import os
import sys
import time

import json
from PIL import Image

from swift.custom_utils.format_func import extract_bbox  # noqa: E402

sys.path.insert(0, '/data/codes/gui_grounding/GUI-SD-code-main')

SYSTEM_PROMPT = 'You are a helpful assistant.'


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


def center_score_xyxy(a, b):
    ax = (a[0] + a[2]) / 2
    ay = (a[1] + a[3]) / 2
    bx = (b[0] + b[2]) / 2
    by = (b[1] + b[3]) / 2
    dist = math.sqrt((ax - bx)**2 + (ay - by)**2)
    # Coordinates are norm-1000, so the image diagonal is about 1414.
    return max(0.0, 1.0 - dist / math.sqrt(2_000_000))


def area_ratio_score_xyxy(a, b):
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    if aa <= 0 or bb <= 0:
        return 0.0
    r = aa / bb
    return min(r, 1.0 / r)


def aspect_ratio_score_xyxy(a, b):
    aw = max(1, a[2] - a[0])
    ah = max(1, a[3] - a[1])
    bw = max(1, b[2] - b[0])
    bh = max(1, b[3] - b[1])
    r = (aw / ah) / (bw / bh)
    return min(r, 1.0 / r)


def valid_norm_box(bbox):
    return (bbox != 'no bbox' and 0 <= bbox[0] < bbox[2] <= 1000 and 0 <= bbox[1] < bbox[3] <= 1000)


def geometry_reward(pred_bbox, gt_bbox):
    if pred_bbox == 'no bbox':
        return {
            'reward': 0.0,
            'parse_ok': 0.0,
            'valid_box': 0.0,
            'iou': 0.0,
            'center_score': 0.0,
            'area_ratio_score': 0.0,
            'aspect_ratio_score': 0.0,
        }

    parse_ok = 1.0
    valid_box = 1.0 if valid_norm_box(pred_bbox) else 0.0
    iou = iou_xyxy(pred_bbox, gt_bbox) if valid_box else 0.0
    center = center_score_xyxy(pred_bbox, gt_bbox) if valid_box else 0.0
    area = area_ratio_score_xyxy(pred_bbox, gt_bbox) if valid_box else 0.0
    aspect = aspect_ratio_score_xyxy(pred_bbox, gt_bbox) if valid_box else 0.0
    reward = (0.10 * parse_ok + 0.10 * valid_box + 0.50 * iou + 0.15 * center + 0.10 * area + 0.05 * aspect)
    return {
        'reward': reward,
        'parse_ok': parse_ok,
        'valid_box': valid_box,
        'iou': iou,
        'center_score': center,
        'area_ratio_score': area,
        'aspect_ratio_score': aspect,
    }


def xyxy_norm1000(bbox_pixel, image_size):
    W, H = image_size
    return [
        int(bbox_pixel[0] / W * 1000),
        int(bbox_pixel[1] / H * 1000),
        int(bbox_pixel[2] / W * 1000),
        int(bbox_pixel[3] / H * 1000),
    ]


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
    ap.add_argument('--model', required=True, help='ckpt dir (HF-compatible)')
    ap.add_argument('--test_jsonl', default='/data/codes/gui_grounding/data/rs_sub_dir/rs_test.jsonl')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--max_samples', type=int, default=0, help='0 = all')
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--max_model_len', type=int, default=12000)
    ap.add_argument('--gpu_mem_util', type=float, default=0.85)
    ap.add_argument('--max_new_tokens', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument(
        '--num_rollouts',
        type=int,
        default=1,
        help='number of sampled bbox rollouts per prompt; 1 keeps original greedy eval')
    ap.add_argument(
        '--rollout_temperature',
        type=float,
        default=None,
        help='sampling temperature for --num_rollouts > 1; defaults to 0.7 when sampling')
    ap.add_argument('--top_p', type=float, default=0.95)
    ap.add_argument(
        '--eval_batch_size',
        type=int,
        default=0,
        help='number of prompts per vLLM.generate call; 0 = all prompts at once')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    samples = load_test(args.test_jsonl, args.max_samples)
    print(f'[eval_student] loaded {len(samples)} samples', flush=True)

    print(f'[eval_student] boot vLLM, model={args.model}, tp={args.tp}', flush=True)
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
    temperature = args.rollout_temperature
    if temperature is None:
        temperature = 0.0 if args.num_rollouts == 1 else 0.7
    sp_kwargs = {
        'temperature': temperature,
        'max_tokens': args.max_new_tokens,
        'seed': args.seed,
    }
    if args.num_rollouts > 1:
        sp_kwargs.update({'n': args.num_rollouts, 'top_p': args.top_p})
    sp = SamplingParams(**sp_kwargs)

    def build_request(s):
        img = Image.open(s['image']).convert('RGB')
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
                        'text': s['user_text']
                    },
                ]
            },
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return {'prompt': prompt, 'multi_modal_data': {'image': img}}

    t0 = time.time()

    if args.eval_batch_size and args.eval_batch_size > 0:
        outputs = []
        for start in range(0, len(samples), args.eval_batch_size):
            end = min(start + args.eval_batch_size, len(samples))
            print(f'[eval_student] prepare batch {start}:{end} / {len(samples)}', flush=True)
            reqs = [build_request(s) for s in samples[start:end]]
            print(f'[eval_student] generate batch {start}:{end} / {len(samples)}', flush=True)
            outputs.extend(llm.generate(reqs, sampling_params=sp))
            for req in reqs:
                req['multi_modal_data']['image'].close()
    else:
        reqs = [build_request(s) for s in samples]
        outputs = llm.generate(reqs, sampling_params=sp)

    ious, parse_ok = [], 0
    valid_ok, mean_best_reward = 0, 0.0
    per_sample = []
    for s, out in zip(samples, outputs):
        candidates = []
        for ridx, cand in enumerate(out.outputs):
            text = cand.text
            pred_bbox = extract_bbox(text)
            reward_info = geometry_reward(pred_bbox, s['gt_norm'])
            candidates.append({
                'rollout_id': ridx,
                'pred_bbox': pred_bbox if pred_bbox != 'no bbox' else None,
                'text': text,
                **{k: round(v, 6)
                   for k, v in reward_info.items()},
            })
        best = max(candidates, key=lambda c: (c['reward'], c['iou'], c['valid_box']))
        pred_bbox = best['pred_bbox'] if best['pred_bbox'] is not None else 'no bbox'
        if pred_bbox != 'no bbox':
            parse_ok += 1
            iou = iou_xyxy(pred_bbox, s['gt_norm'])
        else:
            iou = 0.0
        if best['valid_box']:
            valid_ok += 1
        mean_best_reward += best['reward']
        ious.append(iou)
        per_sample.append({
            'ref_id': s['ref_id'],
            'category': s['category'],
            'gt_norm': s['gt_norm'],
            'pred_bbox': pred_bbox if pred_bbox != 'no bbox' else None,
            'iou': iou,
            'reward': best['reward'],
            'best_rollout_id': best['rollout_id'],
            'text': best['text'],
            'candidates': candidates,
        })

    parsed = [s for s in per_sample if s['pred_bbox'] is not None]
    n = len(samples)
    metrics = {
        'model': args.model,
        'n': n,
        'num_rollouts': args.num_rollouts,
        'rollout_temperature': temperature,
        'top_p': args.top_p if args.num_rollouts > 1 else None,
        'time_s': round(time.time() - t0, 1),
        'mIoU': round(sum(ious) / max(1, n), 4),
        'IoU@0.5': round(sum(i > 0.5 for i in ious) / max(1, n), 4),
        'IoU@0.7': round(sum(i > 0.7 for i in ious) / max(1, n), 4),
        'parse_rate': round(parse_ok / max(1, n), 4),
        'valid_rate': round(valid_ok / max(1, n), 4),
        'mean_best_reward': round(mean_best_reward / max(1, n), 4),
        'mIoU_parsed': round(sum(s['iou'] for s in parsed) / max(1, len(parsed)), 4),
    }

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out_dir, 'per_sample.jsonl'), 'w') as f:
        for r in per_sample:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print('\n=== student eval ===', flush=True)
    print(json.dumps(metrics, indent=2))
    print(f'saved to {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
