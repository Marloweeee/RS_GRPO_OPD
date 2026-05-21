"""
Stage 1 offline rollout-pool generation for BBox-SRPO / sample-routed OPSD.

For each remote-sensing grounding sample, the student samples K bbox responses.
The script parses each bbox, computes geometry reward, assigns route labels
(good / failed / ambiguous), and writes:
  - rollout_pool.jsonl: one record per original sample, with all K candidates
  - summary.json: aggregate oracle-gap and routing statistics
  - category_summary.json: per-category breakdown
  - area_summary.json: small/medium/large GT-area breakdown
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

from PIL import Image

sys.path.insert(0, '/data/codes/gui_grounding/GUI-SD-code-main')
from swift.custom_utils.format_func import extract_bbox  # noqa: E402


SYSTEM_PROMPT = "You are a helpful assistant."


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
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def center_score_xyxy(a, b):
    ax = (a[0] + a[2]) / 2
    ay = (a[1] + a[3]) / 2
    bx = (b[0] + b[2]) / 2
    by = (b[1] + b[3]) / 2
    dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
    return max(0.0, 1.0 - dist / math.sqrt(2_000_000))


def area_ratio_score_xyxy(a, b):
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    if aa <= 0 or bb <= 0:
        return 0.0
    ratio = aa / bb
    return min(ratio, 1.0 / ratio)


def aspect_ratio_score_xyxy(a, b):
    aw = max(1, a[2] - a[0])
    ah = max(1, a[3] - a[1])
    bw = max(1, b[2] - b[0])
    bh = max(1, b[3] - b[1])
    ratio = (aw / ah) / (bw / bh)
    return min(ratio, 1.0 / ratio)


def valid_norm_box(bbox):
    return (
        bbox != 'no bbox'
        and 0 <= bbox[0] < bbox[2] <= 1000
        and 0 <= bbox[1] < bbox[3] <= 1000
    )


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
    reward = (
        0.10 * parse_ok
        + 0.10 * valid_box
        + 0.50 * iou
        + 0.15 * center
        + 0.10 * area
        + 0.05 * aspect
    )
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
    width, height = image_size
    return [
        int(bbox_pixel[0] / width * 1000),
        int(bbox_pixel[1] / height * 1000),
        int(bbox_pixel[2] / width * 1000),
        int(bbox_pixel[3] / height * 1000),
    ]


def bbox_area_frac_norm1000(bbox):
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]) / 1_000_000


def area_bucket(gt_norm):
    frac = bbox_area_frac_norm1000(gt_norm)
    if frac < 0.01:
        return 'small'
    if frac < 0.05:
        return 'medium'
    return 'large'


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
            gt_norm = xyxy_norm1000(gt_pixel, image_size)
            samples.append({
                'sample_id': data.get('sample_id'),
                'image': data['images'],
                'image_size': image_size,
                'gt_pixel': gt_pixel,
                'gt_norm': gt_norm,
                'gt_area_frac': bbox_area_frac_norm1000(gt_norm),
                'area_bucket': area_bucket(gt_norm),
                'user_text': data['messages'][1]['content'],
                'assistant_text': data['messages'][2]['content'] if len(data.get('messages', [])) > 2 else '',
                'category': additional.get('category', ''),
                'ref_id': additional.get('ref_id', ''),
            })
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def mean(values):
    return sum(values) / max(1, len(values))


def stdev(values):
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def round_float(value, digits=6):
    if isinstance(value, float):
        return round(value, digits)
    return value


def route_candidates(candidates, tau_good, tau_fail, delta):
    rewards = [c['reward'] for c in candidates]
    group_mean = mean(rewards)
    group_std = stdev(rewards)
    denom = group_std + 1e-6
    for cand in candidates:
        adv = (cand['reward'] - group_mean) / denom
        cand['advantage'] = round_float(adv)
        if cand['valid_box'] and cand['iou'] >= tau_good and cand['reward'] >= group_mean:
            route = 'good'
        elif (not cand['valid_box']) or cand['iou'] < tau_fail or adv < -delta:
            route = 'failed'
        else:
            route = 'ambiguous'
        cand['route'] = route
    best = max(candidates, key=lambda c: (c['reward'], c['iou'], c['valid_box']))
    worst = min(candidates, key=lambda c: (c['reward'], c['iou'], c['valid_box']))
    return group_mean, group_std, best, worst


def summarize_records(records, num_rollouts):
    n = len(records)
    greedy_ious = []
    best_ious = []
    random_ious = []
    mean_iou_per_group = []
    route_counts = defaultdict(int)
    correction_counts = defaultdict(int)
    parse_total = 0
    valid_total = 0
    candidate_total = 0
    group_reward_stds = []

    for rec in records:
        candidates = rec['rollouts']
        candidate_total += len(candidates)
        if candidates:
            greedy_ious.append(candidates[0]['iou'])
            random_ious.append(candidates[0]['iou'])
            mean_iou_per_group.append(mean([c['iou'] for c in candidates]))
        best_ious.append(rec['group_best_iou'])
        group_reward_stds.append(rec['group_std_reward'])
        correction_counts[rec['correction_source']] += 1
        for cand in candidates:
            route_counts[cand['route']] += 1
            parse_total += int(bool(cand['parse_ok']))
            valid_total += int(bool(cand['valid_box']))

    return {
        'n': n,
        'num_rollouts': num_rollouts,
        'candidate_total': candidate_total,
        'greedy_mIoU_rollout0': round_float(mean(greedy_ious), 4),
        'greedy_IoU@0.5_rollout0': round_float(mean([v > 0.5 for v in greedy_ious]), 4),
        'greedy_IoU@0.7_rollout0': round_float(mean([v > 0.7 for v in greedy_ious]), 4),
        'best_of_k_mIoU': round_float(mean(best_ious), 4),
        'best_of_k_IoU@0.5': round_float(mean([v > 0.5 for v in best_ious]), 4),
        'best_of_k_IoU@0.7': round_float(mean([v > 0.7 for v in best_ious]), 4),
        'random_one_mIoU_proxy_rollout0': round_float(mean(random_ious), 4),
        'mean_of_k_mIoU': round_float(mean(mean_iou_per_group), 4),
        'oracle_gap_IoU@0.5': round_float(
            mean([v > 0.5 for v in best_ious]) - mean([v > 0.5 for v in greedy_ious]), 4),
        'oracle_gap_mIoU': round_float(mean(best_ious) - mean(greedy_ious), 4),
        'parse_rate_candidates': round_float(parse_total / max(1, candidate_total), 4),
        'valid_rate_candidates': round_float(valid_total / max(1, candidate_total), 4),
        'group_reward_std_mean': round_float(mean(group_reward_stds), 4),
        'group_reward_std_p50': round_float(percentile(group_reward_stds, 50), 4),
        'route_counts': dict(route_counts),
        'route_ratio': {k: round_float(v / max(1, candidate_total), 4) for k, v in sorted(route_counts.items())},
        'correction_source_counts': dict(correction_counts),
        'correction_source_ratio': {k: round_float(v / max(1, n), 4) for k, v in sorted(correction_counts.items())},
    }


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    idx = (len(values) - 1) * pct / 100
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[int(idx)]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def grouped_summary(records, key, num_rollouts):
    groups = defaultdict(list)
    for rec in records:
        groups[rec.get(key, '') or 'unknown'].append(rec)
    return {name: summarize_records(recs, num_rollouts) for name, recs in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--jsonl', default='/data/codes/gui_grounding/data/rs_sub_dir/rs_train.jsonl')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--max_samples', type=int, default=0, help='0 = all')
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--max_model_len', type=int, default=12000)
    parser.add_argument('--gpu_mem_util', type=float, default=0.85)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--num_rollouts', type=int, default=8)
    parser.add_argument('--rollout_temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tau_good', type=float, default=0.5)
    parser.add_argument('--tau_fail', type=float, default=0.3)
    parser.add_argument('--delta', type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    samples = load_samples(args.jsonl, args.max_samples)
    print(f'[stage1] loaded {len(samples)} samples from {args.jsonl}', flush=True)

    os.environ.setdefault('IMAGE_MAX_TOKEN_NUM', '10000')
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    print(f'[stage1] boot vLLM model={args.model} tp={args.tp}', flush=True)
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

    requests = []
    t0 = time.time()
    for sample in samples:
        image = Image.open(sample['image']).convert('RGB')
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': sample['user_text']},
            ]},
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        requests.append({'prompt': prompt, 'multi_modal_data': {'image': image}})

    outputs = llm.generate(requests, sampling_params=sampling_params)
    records = []
    for sample, output in zip(samples, outputs):
        candidates = []
        for rollout_id, candidate in enumerate(output.outputs):
            text = candidate.text
            bbox = extract_bbox(text)
            reward_info = geometry_reward(bbox, sample['gt_norm'])
            cand_record = {
                'rollout_id': rollout_id,
                'text': text,
                'bbox': bbox if bbox != 'no bbox' else None,
            }
            cand_record.update({k: round_float(v) for k, v in reward_info.items()})
            candidates.append(cand_record)

        group_mean, group_std, best, worst = route_candidates(
            candidates, tau_good=args.tau_good, tau_fail=args.tau_fail, delta=args.delta)
        correction_source = (
            'best_sibling' if best['iou'] >= args.tau_good
            else 'teacher_or_gt_needed'
        )
        records.append({
            'sample_id': sample['sample_id'],
            'ref_id': sample['ref_id'],
            'category': sample['category'],
            'image': sample['image'],
            'image_size': sample['image_size'],
            'gt_pixel': sample['gt_pixel'],
            'gt_bbox_norm': sample['gt_norm'],
            'gt_area_frac': round_float(sample['gt_area_frac']),
            'area_bucket': sample['area_bucket'],
            'prompt': sample['user_text'],
            'assistant_gt': sample['assistant_text'],
            'rollouts': candidates,
            'group_mean_reward': round_float(group_mean),
            'group_std_reward': round_float(group_std),
            'group_best_id': best['rollout_id'],
            'group_best_iou': round_float(best['iou']),
            'group_best_reward': round_float(best['reward']),
            'group_worst_id': worst['rollout_id'],
            'group_worst_iou': round_float(worst['iou']),
            'correction_source': correction_source,
        })

    pool_path = os.path.join(args.out_dir, 'rollout_pool.jsonl')
    with open(pool_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    summary = summarize_records(records, args.num_rollouts)
    summary.update({
        'model': args.model,
        'jsonl': args.jsonl,
        'out_dir': args.out_dir,
        'time_s': round_float(time.time() - t0, 1),
        'rollout_temperature': args.rollout_temperature,
        'top_p': args.top_p,
        'tau_good': args.tau_good,
        'tau_fail': args.tau_fail,
        'delta': args.delta,
    })

    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, 'category_summary.json'), 'w') as f:
        json.dump(grouped_summary(records, 'category', args.num_rollouts), f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, 'area_summary.json'), 'w') as f:
        json.dump(grouped_summary(records, 'area_bucket', args.num_rollouts), f, indent=2, ensure_ascii=False)

    print('\n=== Stage 1 Summary ===', flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f'[stage1] saved rollout pool to {pool_path}', flush=True)


if __name__ == '__main__':
    main()
