"""
Evaluate a checkpoint with OPSD-style visual hints and best/pass@K sampling.

This is used to measure the upper bound of a SFT checkpoint when it acts as a
teacher with privileged GT visual hints. The default hint modes match the
training-side OPSD teacher inputs:
  - gaussian: Gaussian dimming + green GT rectangle + green-rectangle text hint.
  - zoom_in: keep a local target window + green GT rectangle + green-rectangle text hint.
"""

import argparse
import math
import os
import random
import sys
import time

import json
import numpy as np
from PIL import Image, ImageDraw
from tools.evaluation.eval_student import geometry_reward, iou_xyxy, xyxy_norm1000  # noqa: E402

from swift.custom_utils.format_func import extract_bbox  # noqa: E402

sys.path.insert(0, '/data/codes/gui_grounding/GUI-SD-code-main')

SYSTEM_PROMPT = 'You are a helpful assistant.'


def load_samples(jsonl_path, max_samples):
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            additional = data['additional_paras']
            if isinstance(additional, str):
                additional = json.loads(additional)
            image_size = additional['image_size']
            gt_pixel = data['solution']['arguments']['coordinate']
            samples.append({
                'image': data['images'],
                'image_size': image_size,
                'gt_pixel': gt_pixel,
                'gt_norm': xyxy_norm1000(gt_pixel, image_size),
                'user_text': data['messages'][1]['content'],
                'category': additional.get('category', ''),
                'ref_id': additional.get('ref_id', ''),
                'sample_id': data.get('sample_id'),
            })
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def render_hint_image(image,
                      bbox_pixel,
                      mode,
                      sigma_ratio=1.5,
                      min_area_frac=0.1,
                      zoom_ratio=2.0,
                      jitter_ratio=0.2,
                      hint_box_color='magenta'):
    x1, y1, x2, y2 = map(int, bbox_pixel)
    width, height = image.size
    bbox_w, bbox_h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    draw_box = (x1, y1, x2, y2, 'green')

    if mode == 'none':
        result = image.copy()
        draw_box = None
    elif mode == 'zoom_in':
        result = Image.new('RGB', (width, height), 'black')
        crop_x1 = max(0, cx - width // 4)
        crop_y1 = max(0, cy - height // 4)
        crop_x2 = min(width, cx + width // 4)
        crop_y2 = min(height, cy + height // 4)
        if crop_x1 < crop_x2 and crop_y1 < crop_y2:
            result.paste(image.crop((crop_x1, crop_y1, crop_x2, crop_y2)), (crop_x1, crop_y1))
    elif mode == 'adaptive':
        pad_w = max(int(bbox_w * zoom_ratio), int(width * math.sqrt(min_area_frac) / 2))
        pad_h = max(int(bbox_h * zoom_ratio), int(height * math.sqrt(min_area_frac) / 2))
        crop_x1 = max(0, cx - pad_w)
        crop_y1 = max(0, cy - pad_h)
        crop_x2 = min(width, cx + pad_w)
        crop_y2 = min(height, cy + pad_h)
        result = Image.new('RGB', (width, height), 'black')
        if crop_x1 < crop_x2 and crop_y1 < crop_y2:
            result.paste(image.crop((crop_x1, crop_y1, crop_x2, crop_y2)), (crop_x1, crop_y1))
    elif mode in {'gaussian', 'soft_window'}:
        sigma = max(max(bbox_w, bbox_h) * sigma_ratio, min(width, height) * math.sqrt(min_area_frac))
        arr = np.array(image, dtype=np.float32)
        xs = np.arange(width)[None, :]
        ys = np.arange(height)[:, None]
        dx = np.maximum(x1 - xs, 0) + np.maximum(xs - x2, 0)
        dy = np.maximum(y1 - ys, 0) + np.maximum(ys - y2, 0)
        dist = np.sqrt(dx.astype(np.float64)**2 + dy.astype(np.float64)**2)
        alpha = np.exp(-dist**2 / (2 * sigma**2)).astype(np.float32)[:, :, None]
        result = Image.fromarray((arr * alpha).astype(np.uint8))
        if mode == 'soft_window':
            draw_box = None
    elif mode == 'jitter_box':
        result = image.copy()
        jw_max = max(1, int(bbox_w * jitter_ratio))
        jh_max = max(1, int(bbox_h * jitter_ratio))
        jx1 = max(0, min(width - 1, x1 + random.randint(-jw_max, jw_max)))
        jy1 = max(0, min(height - 1, y1 + random.randint(-jh_max, jh_max)))
        jx2 = max(jx1 + 1, min(width, x2 + random.randint(-jw_max, jw_max)))
        jy2 = max(jy1 + 1, min(height, y2 + random.randint(-jh_max, jh_max)))
        draw_box = (jx1, jy1, jx2, jy2, hint_box_color)
    elif mode == 'no_mask':
        result = image.copy()
        draw_box = None
    else:
        raise ValueError(f'unknown hint mode: {mode}')

    if draw_box is not None:
        bx1, by1, bx2, by2, color = draw_box
        line_width = 5
        ImageDraw.Draw(result).rectangle(
            [bx1 - line_width, by1 - line_width, bx2 + line_width, by2 + line_width],
            outline=color,
            width=line_width,
        )
    return result


def build_hint_text(user_text, mode, hint_box_color):
    if mode in {'none', 'no_mask'}:
        return user_text
    if mode == 'soft_window':
        return user_text + ' Hint: The target is in the brighter (un-dimmed) region of the image.'
    if mode == 'jitter_box':
        return user_text + f' Hint: The target is approximately within the {hint_box_color} box.'
    return user_text + ' Hint: The answer is located within the green rectangle.'


def threshold_metrics(ious, prefix=''):
    metrics = {f'{prefix}mIoU': round(sum(ious) / max(1, len(ious)), 4)}
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        metrics[f'{prefix}IoU@{threshold:.1f}'] = round(sum(i > threshold for i in ious) / max(1, len(ious)), 4)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--test_jsonl', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument(
        '--hint_mode',
        default='gaussian',
        choices=['none', 'zoom_in', 'adaptive', 'gaussian', 'soft_window', 'jitter_box', 'no_mask'])
    parser.add_argument('--num_rollouts', type=int, default=8)
    parser.add_argument('--rollout_temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--tp', type=int, default=8)
    parser.add_argument('--max_model_len', type=int, default=12000)
    parser.add_argument('--gpu_mem_util', type=float, default=0.88)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--opsd_gaussian_sigma_ratio', type=float, default=1.5)
    parser.add_argument('--opsd_min_area_frac', type=float, default=0.1)
    parser.add_argument('--opsd_zoom_ratio', type=float, default=2.0)
    parser.add_argument('--opsd_jitter_ratio', type=float, default=0.2)
    parser.add_argument('--opsd_hint_box_color', default='magenta')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault('IMAGE_MAX_TOKEN_NUM', '10000')

    samples = load_samples(args.test_jsonl, args.max_samples)
    print(f'[hint-pass8] loaded {len(samples)} samples from {args.test_jsonl}', flush=True)
    print(f'[hint-pass8] boot vLLM model={args.model} tp={args.tp} hint_mode={args.hint_mode}', flush=True)

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

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
    sampling_params = SamplingParams(
        temperature=args.rollout_temperature,
        top_p=args.top_p,
        n=args.num_rollouts,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    def build_request(sample):
        image = Image.open(sample['image']).convert('RGB')
        hint_image = render_hint_image(
            image,
            sample['gt_pixel'],
            args.hint_mode,
            sigma_ratio=args.opsd_gaussian_sigma_ratio,
            min_area_frac=args.opsd_min_area_frac,
            zoom_ratio=args.opsd_zoom_ratio,
            jitter_ratio=args.opsd_jitter_ratio,
            hint_box_color=args.opsd_hint_box_color,
        )
        image.close()
        messages = [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role':
                'user',
                'content': [
                    {
                        'type': 'image'
                    },
                    {
                        'type': 'text',
                        'text': build_hint_text(sample['user_text'], args.hint_mode, args.opsd_hint_box_color)
                    },
                ]
            },
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return {'prompt': prompt, 'multi_modal_data': {'image': hint_image}}

    t0 = time.time()
    outputs = []
    batch_size = args.eval_batch_size if args.eval_batch_size and args.eval_batch_size > 0 else len(samples)
    for start in range(0, len(samples), batch_size):
        end = min(start + batch_size, len(samples))
        print(f'[hint-pass8] generate batch {start}:{end} / {len(samples)}', flush=True)
        requests = [build_request(s) for s in samples[start:end]]
        outputs.extend(llm.generate(requests, sampling_params=sampling_params))
        for request in requests:
            request['multi_modal_data']['image'].close()

    first_ious = []
    best_ious = []
    first_parse = 0
    first_valid = 0
    any_parse = 0
    any_valid = 0
    per_sample = []
    for sample, output in zip(samples, outputs):
        candidates = []
        for rollout_id, candidate in enumerate(output.outputs):
            text = candidate.text
            pred_bbox = extract_bbox(text)
            reward_info = geometry_reward(pred_bbox, sample['gt_norm'])
            candidates.append({
                'rollout_id': rollout_id,
                'pred_bbox': pred_bbox if pred_bbox != 'no bbox' else None,
                'text': text,
                **{k: round(v, 6)
                   for k, v in reward_info.items()},
            })
        best = max(candidates, key=lambda c: (c['iou'], c['reward'], c['valid_box']))
        first = candidates[0]
        first_iou = first['iou']
        best_iou = best['iou']
        first_ious.append(first_iou)
        best_ious.append(best_iou)
        first_parse += int(first['parse_ok'] > 0)
        first_valid += int(first['valid_box'] > 0)
        any_parse += int(any(c['parse_ok'] > 0 for c in candidates))
        any_valid += int(any(c['valid_box'] > 0 for c in candidates))
        row = {
            'sample_id': sample['sample_id'],
            'ref_id': sample['ref_id'],
            'category': sample['category'],
            'gt_norm': sample['gt_norm'],
            'hint_mode': args.hint_mode,
            'first_iou': first_iou,
            'best_iou': best_iou,
            'best_rollout_id': best['rollout_id'],
            'best_pred_bbox': best['pred_bbox'],
            'first_pred_bbox': first['pred_bbox'],
            'candidates': candidates,
        }
        for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
            row[f'pass@{args.num_rollouts}@{threshold:.1f}'] = best_iou > threshold
        per_sample.append(row)

    n = len(samples)
    metrics = {
        'model': args.model,
        'test_jsonl': args.test_jsonl,
        'hint_mode': args.hint_mode,
        'n': n,
        'num_rollouts': args.num_rollouts,
        'rollout_temperature': args.rollout_temperature,
        'top_p': args.top_p,
        'time_s': round(time.time() - t0, 1),
        'first_parse_rate': round(first_parse / max(1, n), 4),
        'first_valid_rate': round(first_valid / max(1, n), 4),
        'any_parse_rate': round(any_parse / max(1, n), 4),
        'any_valid_rate': round(any_valid / max(1, n), 4),
    }
    metrics.update(threshold_metrics(first_ious, prefix='first_'))
    metrics.update(threshold_metrics(best_ious, prefix='best@%d_' % args.num_rollouts))
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        metrics[f'pass@{args.num_rollouts}@{threshold:.1f}'] = round(
            sum(i > threshold for i in best_ious) / max(1, n), 4)

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out_dir, 'per_sample.jsonl'), 'w') as f:
        for row in per_sample:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print('[hint-pass8] summary_path=' + os.path.join(args.out_dir, 'summary.json'), flush=True)
    print('[hint-pass8] metrics=' + json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
