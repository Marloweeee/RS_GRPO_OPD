"""
RRSIS-D → anno.jsonl 转换脚本

读取  refs(unc).p + instances.json (COCO 格式)
输出  train/val/test 三份 jsonl，与 anno.jsonl 同构（messages / images / solution / additional_paras / sample_id）

关键改造：
  - GT bbox 在 RRSIS-D 的 instances.json 里**已经是 [x1,y1,x2,y2] 像素**（不是 COCO xywh，经核对）
  - 输出格式对齐 Qwen3-VL 官方 grounding cookbook：
      user:      Locate {sent}, output its bbox coordinates using JSON format.
      assistant: {"bbox_2d": [nx1, ny1, nx2, ny2]}   (norm-1000)
    教师 SFT 阶段就是按这套训出来的，KD 软分布最自然。
  - user 用 <|object_ref_start|>...<|object_ref_end|> 包裹 referring expression
  - sample_id 偏移 100000 避免与 UI 数据撞 id
  - solution.arguments.coordinate 保持 4 元素像素 bbox（mask_processor / monitoring 需要）
"""

import argparse
import os
import pickle
from collections import defaultdict

import json

SYSTEM_PROMPT = 'You are a helpful assistant.'


def clip_xyxy(bbox, image_size):
    x1, y1, x2, y2 = bbox
    W, H = image_size
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(0, min(int(x2), W))
    y2 = max(0, min(int(y2), H))
    return [x1, y1, x2, y2]


def xyxy_to_norm1000(bbox, image_size):
    x1, y1, x2, y2 = bbox
    W, H = image_size
    return [
        int(x1 / W * 1000),
        int(y1 / H * 1000),
        int(x2 / W * 1000),
        int(y2 / H * 1000),
    ]


def make_record(ref, ann, img, cat_name, image_prefix, sample_id_offset):
    sent = ref['sentences'][0]['sent'].strip()
    image_size = [img['width'], img['height']]
    bbox_xyxy = clip_xyxy(ann['bbox'], image_size)
    nx1, ny1, nx2, ny2 = xyxy_to_norm1000(bbox_xyxy, image_size)

    user_text = f'Locate {sent}, output its bbox coordinates using JSON format.'
    assistant_text = json.dumps({'bbox_2d': [nx1, ny1, nx2, ny2]}, separators=(',', ': '))

    record = {
        'solution': {
            'name': 'rs_grounding',
            'arguments': {
                'action': 'locate',
                'coordinate': bbox_xyxy
            },
        },
        'images':
        os.path.join(image_prefix, ref['file_name']),
        'messages': [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': user_text
            },
            {
                'role': 'assistant',
                'content': assistant_text
            },
        ],
        'additional_paras':
        json.dumps({
            'image_size': image_size,
            'platform': 'satellite',
            'ui_type': 'object',
            'category': cat_name,
            'ref_id': ref['ref_id'],
        }),
        'sample_id':
        sample_id_offset + ref['ann_id'],
    }
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', default='/data/codes/gui_grounding/data/rs_sub_dir')
    parser.add_argument('--out_dir', default='/data/codes/gui_grounding/data/rs_sub_dir')
    parser.add_argument(
        '--image_prefix',
        default='/data/codes/gui_grounding/data/rs_grounding/JPEGImages',
        help='图像绝对路径前缀；jsonl 里 images 字段 = prefix/<file_name>')
    parser.add_argument('--sample_id_offset', type=int, default=100000)
    parser.add_argument('--also_all', action='store_true', help='额外输出一个合并 all.jsonl')
    args = parser.parse_args()

    refs_path = os.path.join(args.src_dir, 'refs(unc).p')
    inst_path = os.path.join(args.src_dir, 'instances.json')

    print(f'reading {refs_path}')
    with open(refs_path, 'rb') as f:
        refs = pickle.load(f)
    print(f'reading {inst_path}')
    with open(inst_path) as f:
        inst = json.load(f)

    ann_by_id = {a['id']: a for a in inst['annotations']}
    img_by_id = {im['id']: im for im in inst['images']}
    cat_by_id = {c['id']: c['name'] for c in inst['categories']}

    os.makedirs(args.out_dir, exist_ok=True)
    buckets = defaultdict(list)
    skipped = 0
    for ref in refs:
        ann = ann_by_id.get(ref['ann_id'])
        img = img_by_id.get(ref['image_id'])
        if ann is None or img is None:
            skipped += 1
            continue
        cat_name = cat_by_id.get(ref['category_id'], 'unknown')
        rec = make_record(ref, ann, img, cat_name, args.image_prefix, args.sample_id_offset)
        buckets[ref['split']].append(rec)

    for split, recs in buckets.items():
        out_path = os.path.join(args.out_dir, f'rs_{split}.jsonl')
        with open(out_path, 'w') as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'  wrote {out_path}: {len(recs)} records')

    if args.also_all:
        out_path = os.path.join(args.out_dir, 'rs_all.jsonl')
        all_recs = sum(buckets.values(), [])
        with open(out_path, 'w') as f:
            for r in all_recs:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'  wrote {out_path}: {len(all_recs)} records (combined)')

    if skipped:
        print(f'skipped {skipped} refs (missing ann/img)')


if __name__ == '__main__':
    main()
