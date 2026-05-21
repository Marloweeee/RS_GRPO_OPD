"""
Build Stage 2 offline datasets from a Stage 1 rollout pool.

Outputs:
  - stage2a_best_sibling_sft.jsonl
      SFT compression target. If the group contains a good sibling, use the
      best student bbox; otherwise fall back to the GT bbox correction target.
  - stage2a_gt_sft.jsonl
      GT-only SFT control with the same prompts.
  - stage2b_best_vs_worst_dpo.jsonl
      Preference data. The chosen response is the Stage 2A target; the rejected
      response is the worst distinct rollout from the same group.
  - *_smoke.jsonl
      Small deterministic subsets for trainer smoke checks.
  - summary.json
      Counts, target-source ratios, and chosen/rejected quality statistics.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, '/data/codes/gui_grounding/GUI-SD-code-main')
from swift.custom_utils.format_func import extract_bbox  # noqa: E402


SYSTEM_PROMPT = 'You are a helpful assistant.'


def clean_bbox_json(bbox):
    return json.dumps({'bbox_2d': [int(v) for v in bbox]}, separators=(',', ':'))


def mean(values):
    return sum(values) / max(1, len(values))


def round_float(value, digits=6):
    if isinstance(value, float):
        return round(value, digits)
    return value


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


def gt_answer(record):
    bbox = record['gt_bbox_norm']
    return clean_bbox_json(bbox), bbox, 1.0, 1.0


def get_candidate(record, rollout_id):
    for cand in record['rollouts']:
        if cand['rollout_id'] == rollout_id:
            return cand
    raise KeyError(f'rollout_id={rollout_id} not found for sample_id={record.get("sample_id")}')


def best_candidate(record):
    best_id = record.get('group_best_id')
    if best_id is not None:
        return get_candidate(record, best_id)
    return max(record['rollouts'], key=lambda c: (c.get('reward', 0), c.get('iou', 0), c.get('valid_box', 0)))


def sorted_worst_candidates(record):
    return sorted(record['rollouts'], key=lambda c: (c.get('reward', 0), c.get('iou', 0), c.get('valid_box', 0)))


def build_base_record(record, assistant_text):
    additional = {
        'image_size': record['image_size'],
        'platform': 'satellite',
        'ui_type': 'object',
        'category': record.get('category', ''),
        'ref_id': record.get('ref_id', ''),
    }
    return {
        'solution': {
            'name': 'rs_grounding',
            'arguments': {
                'action': 'locate',
                'coordinate': record['gt_pixel'],
            },
        },
        'images': record['image'],
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': record['prompt']},
            {'role': 'assistant', 'content': assistant_text},
        ],
        'additional_paras': json.dumps(additional, ensure_ascii=False),
        'sample_id': record.get('sample_id'),
    }


def choose_stage2a_target(record, tau_good):
    best = best_candidate(record)
    if best.get('bbox') is not None and best.get('iou', 0.0) >= tau_good:
        return {
            'source': 'best_sibling',
            'text': clean_bbox_json(best['bbox']),
            'bbox': best['bbox'],
            'iou': best.get('iou', 0.0),
            'reward': best.get('reward', 0.0),
            'rollout_id': best.get('rollout_id'),
        }
    text, bbox, iou, reward = gt_answer(record)
    return {
        'source': 'gt_fallback',
        'text': text,
        'bbox': bbox,
        'iou': iou,
        'reward': reward,
        'rollout_id': None,
    }


def choose_rejected(record, chosen_text):
    for cand in sorted_worst_candidates(record):
        bbox = cand.get('bbox')
        if bbox is None:
            continue
        text = clean_bbox_json(bbox)
        if text != chosen_text:
            return {
                'text': text,
                'bbox': bbox,
                'iou': cand.get('iou', 0.0),
                'reward': cand.get('reward', 0.0),
                'rollout_id': cand.get('rollout_id'),
                'route': cand.get('route'),
            }
    return None


def add_stage2_meta(row, meta):
    row['stage2_meta'] = meta
    return row


def load_pool(path, max_samples):
    records = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            records.append(record)
            if max_samples and len(records) >= max_samples:
                break
    return records


def write_jsonl(path, records):
    with open(path, 'w') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def smoke_subset(records, count):
    if count <= 0 or len(records) <= count:
        return records
    by_source = defaultdict(list)
    for row in records:
        source = row.get('stage2_meta', {}).get('target_source', 'unknown')
        by_source[source].append(row)
    subset = []
    keys = sorted(by_source)
    while len(subset) < count and any(by_source.values()):
        for key in keys:
            if by_source[key] and len(subset) < count:
                subset.append(by_source[key].pop(0))
    return subset


def validate_rows(rows, is_dpo=False):
    parse_ok = 0
    rejected_parse_ok = 0
    image_exists = 0
    for row in rows:
        if os.path.exists(row['images']):
            image_exists += 1
        bbox = extract_bbox(row['messages'][-1]['content'])
        if bbox != 'no bbox':
            parse_ok += 1
        if is_dpo:
            rbbox = extract_bbox(row['rejected_response'])
            if rbbox != 'no bbox':
                rejected_parse_ok += 1
    result = {
        'n': len(rows),
        'image_exists_rate': round_float(image_exists / max(1, len(rows)), 4),
        'chosen_parse_rate': round_float(parse_ok / max(1, len(rows)), 4),
    }
    if is_dpo:
        result['rejected_parse_rate'] = round_float(rejected_parse_ok / max(1, len(rows)), 4)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--tau_good', type=float, default=0.5)
    parser.add_argument('--max_samples', type=int, default=0, help='0 = all')
    parser.add_argument('--smoke_count', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pool_records = load_pool(args.pool, args.max_samples)
    sft_rows = []
    gt_sft_rows = []
    dpo_rows = []
    skipped_dpo = 0

    target_sources = Counter()
    chosen_ious = []
    rejected_ious = []
    dpo_margins = []

    for record in pool_records:
        target = choose_stage2a_target(record, args.tau_good)
        target_sources[target['source']] += 1
        chosen_ious.append(target['iou'])

        sft_meta = {
            'stage': 'stage2a_best_sibling_sft',
            'target_source': target['source'],
            'target_iou': round_float(target['iou']),
            'target_reward': round_float(target['reward']),
            'target_rollout_id': target['rollout_id'],
            'group_best_iou': record.get('group_best_iou'),
            'group_worst_iou': record.get('group_worst_iou'),
            'area_bucket': record.get('area_bucket'),
            'category': record.get('category'),
        }
        sft_rows.append(add_stage2_meta(build_base_record(record, target['text']), sft_meta))

        gt_text, gt_bbox, gt_iou, gt_reward = gt_answer(record)
        gt_meta = dict(sft_meta)
        gt_meta.update({
            'stage': 'stage2a_gt_sft_control',
            'target_source': 'gt',
            'target_iou': gt_iou,
            'target_reward': gt_reward,
            'target_rollout_id': None,
        })
        gt_sft_rows.append(add_stage2_meta(build_base_record(record, gt_text), gt_meta))

        rejected = choose_rejected(record, target['text'])
        if rejected is None:
            skipped_dpo += 1
            continue
        dpo_row = build_base_record(record, target['text'])
        dpo_row['rejected_response'] = rejected['text']
        dpo_meta = dict(sft_meta)
        dpo_meta.update({
            'stage': 'stage2b_best_vs_worst_dpo',
            'rejected_iou': round_float(rejected['iou']),
            'rejected_reward': round_float(rejected['reward']),
            'rejected_rollout_id': rejected['rollout_id'],
            'rejected_route': rejected['route'],
            'preference_iou_margin': round_float(target['iou'] - rejected['iou']),
            'preference_reward_margin': round_float(target['reward'] - rejected['reward']),
        })
        dpo_rows.append(add_stage2_meta(dpo_row, dpo_meta))
        rejected_ious.append(rejected['iou'])
        dpo_margins.append(target['iou'] - rejected['iou'])

    paths = {
        'stage2a_sft': os.path.join(args.out_dir, 'stage2a_best_sibling_sft.jsonl'),
        'stage2a_gt_sft': os.path.join(args.out_dir, 'stage2a_gt_sft.jsonl'),
        'stage2b_dpo': os.path.join(args.out_dir, 'stage2b_best_vs_worst_dpo.jsonl'),
        'stage2a_sft_smoke': os.path.join(args.out_dir, 'stage2a_best_sibling_sft_smoke.jsonl'),
        'stage2a_gt_sft_smoke': os.path.join(args.out_dir, 'stage2a_gt_sft_smoke.jsonl'),
        'stage2b_dpo_smoke': os.path.join(args.out_dir, 'stage2b_best_vs_worst_dpo_smoke.jsonl'),
    }
    write_jsonl(paths['stage2a_sft'], sft_rows)
    write_jsonl(paths['stage2a_gt_sft'], gt_sft_rows)
    write_jsonl(paths['stage2b_dpo'], dpo_rows)
    write_jsonl(paths['stage2a_sft_smoke'], smoke_subset(sft_rows, args.smoke_count))
    write_jsonl(paths['stage2a_gt_sft_smoke'], smoke_subset(gt_sft_rows, args.smoke_count))
    write_jsonl(paths['stage2b_dpo_smoke'], smoke_subset(dpo_rows, args.smoke_count))

    summary = {
        'pool': args.pool,
        'out_dir': args.out_dir,
        'tau_good': args.tau_good,
        'input_records': len(pool_records),
        'stage2a_sft_records': len(sft_rows),
        'stage2a_gt_sft_records': len(gt_sft_rows),
        'stage2b_dpo_records': len(dpo_rows),
        'stage2b_dpo_skipped': skipped_dpo,
        'target_source_counts': dict(target_sources),
        'target_source_ratio': {
            k: round_float(v / max(1, len(sft_rows)), 4)
            for k, v in sorted(target_sources.items())
        },
        'chosen_mIoU': round_float(mean(chosen_ious), 4),
        'chosen_IoU@0.5': round_float(mean([v > 0.5 for v in chosen_ious]), 4),
        'rejected_mIoU': round_float(mean(rejected_ious), 4),
        'rejected_IoU@0.5': round_float(mean([v > 0.5 for v in rejected_ious]), 4),
        'dpo_iou_margin_mean': round_float(mean(dpo_margins), 4),
        'validation': {
            'stage2a_sft': validate_rows(sft_rows),
            'stage2a_gt_sft': validate_rows(gt_sft_rows),
            'stage2b_dpo': validate_rows(dpo_rows, is_dpo=True),
        },
        'paths': paths,
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
