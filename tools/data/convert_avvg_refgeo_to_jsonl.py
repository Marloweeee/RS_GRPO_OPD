"""
AVVG refGeo -> GUI-SD jsonl converter.

The source AVVG refGeo files contain one referring expression and one
axis-aligned bbox per row. The output schema matches the RRSIS-D / RSVG-DIOR
jsonl files used by the current GUI-SD training and evaluation scripts:
  - solution.arguments.coordinate stores clipped pixel-level xyxy bbox.
  - assistant message stores normalized 0-1000 bbox_2d.
  - additional_paras preserves image size, source ids, raw bbox, and polygon.
"""

import argparse
import os
from pathlib import Path

import json
from PIL import Image

SYSTEM_PROMPT = 'You are a helpful assistant.'


def clip_xyxy(bbox, image_size):
    x1, y1, x2, y2 = bbox
    width, height = image_size
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width))
    y2 = max(0, min(int(round(y2)), height))
    return [x1, y1, x2, y2]


def xyxy_to_norm1000(bbox, image_size):
    x1, y1, x2, y2 = bbox
    width, height = image_size
    return [
        int(x1 / width * 1000),
        int(y1 / height * 1000),
        int(x2 / width * 1000),
        int(y2 / height * 1000),
    ]


def is_valid_xyxy(bbox):
    x1, y1, x2, y2 = bbox
    return x2 > x1 and y2 > y1


def read_jsonl(path):
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def write_jsonl(path, records):
    with open(path, 'w') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def get_image_size(image_path, size_cache):
    if image_path not in size_cache:
        with Image.open(image_path) as image:
            size_cache[image_path] = list(image.size)
    return size_cache[image_path]


def make_record(row, split, ref_id, image_prefix, sample_id_offset, size_cache):
    image_id = row['image_id']
    image_path = os.path.join(image_prefix, image_id)
    if not os.path.exists(image_path):
        return None, 'missing_image'

    raw_bbox = row.get('bbox')
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None, 'bad_bbox_format'

    image_size = get_image_size(image_path, size_cache)
    clipped_bbox = clip_xyxy(raw_bbox, image_size)
    if not is_valid_xyxy(clipped_bbox):
        return None, 'invalid_after_clip'

    norm_bbox = xyxy_to_norm1000(clipped_bbox, image_size)
    question = str(row.get('question', '')).strip()
    if not question:
        return None, 'empty_question'

    user_text = f'Locate {question}, output its bbox coordinates using JSON format.'
    assistant_text = json.dumps({'bbox_2d': norm_bbox}, separators=(',', ': '))
    was_clipped = [int(round(v)) for v in raw_bbox] != clipped_bbox

    record = {
        'solution': {
            'name': 'rs_grounding',
            'arguments': {
                'action': 'locate',
                'coordinate': clipped_bbox
            },
        },
        'images':
        image_path,
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
        json.dumps(
            {
                'image_size': image_size,
                'platform': 'uav',
                'ui_type': 'object',
                'category': 'car',
                'ref_id': ref_id,
                'source_dataset': 'AVVG-refGeo',
                'source_split': split,
                'image_id': image_id,
                'question_id': row.get('question_id'),
                'raw_bbox': raw_bbox,
                'raw_poly': row.get('poly'),
                'bbox_was_clipped': was_clipped,
            },
            ensure_ascii=False),
        'sample_id':
        sample_id_offset + ref_id,
    }
    return record, 'ok_clipped' if was_clipped else 'ok'


def convert_split(src_path, split, image_prefix, sample_id_offset, ref_id_start, size_cache):
    records = []
    stats = {
        'source_rows': 0,
        'written_rows': 0,
        'unique_images': 0,
        'clipped_rows': 0,
        'missing_image': 0,
        'bad_bbox_format': 0,
        'invalid_after_clip': 0,
        'empty_question': 0,
    }
    image_ids = set()
    ref_id = ref_id_start

    for _, row in read_jsonl(src_path):
        stats['source_rows'] += 1
        if row.get('image_id'):
            image_ids.add(row['image_id'])
        record, status = make_record(row, split, ref_id, image_prefix, sample_id_offset, size_cache)
        if record is None:
            stats[status] += 1
            continue
        records.append(record)
        stats['written_rows'] += 1
        if status == 'ok_clipped':
            stats['clipped_rows'] += 1
        ref_id += 1

    stats['unique_images'] = len(image_ids)
    return records, stats, ref_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', default='/data/codes/gui_grounding/data/AVVG/refGeo')
    parser.add_argument('--out_dir', default='/data/codes/gui_grounding/data/avvg_refgeo_full')
    parser.add_argument(
        '--image_prefix',
        default='/data/codes/gui_grounding/data/AVVG/refGeo/images/avvg',
        help='Absolute image directory written into the jsonl images field.',
    )
    parser.add_argument('--sample_id_offset', type=int, default=400000)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    metainfo_dir = src_dir / 'metainfo'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    size_cache = {}
    ref_id = 0
    all_records = []
    all_stats = {
        'src_dir': str(src_dir),
        'image_prefix': args.image_prefix,
        'out_dir': str(out_dir),
        'sample_id_offset': args.sample_id_offset,
        'splits': {},
    }

    for split in ('train', 'test'):
        src_path = metainfo_dir / f'avvg_{split}.jsonl'
        records, stats, ref_id = convert_split(
            src_path=src_path,
            split=split,
            image_prefix=args.image_prefix,
            sample_id_offset=args.sample_id_offset,
            ref_id_start=ref_id,
            size_cache=size_cache,
        )
        out_path = out_dir / f'avvg_{split}.jsonl'
        write_jsonl(out_path, records)
        all_records.extend(records)
        all_stats['splits'][split] = stats
        print(f'wrote {out_path}: {len(records)} records')

    all_path = out_dir / 'avvg_all.jsonl'
    write_jsonl(all_path, all_records)
    all_stats['all'] = {
        'written_rows': len(all_records),
        'unique_images': len({json.loads(record['additional_paras'])['image_id']
                              for record in all_records}),
        'sample_id_min': min((record['sample_id'] for record in all_records), default=None),
        'sample_id_max': max((record['sample_id'] for record in all_records), default=None),
    }
    stats_path = out_dir / 'convert_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f'wrote {all_path}: {len(all_records)} records')
    print(f'wrote {stats_path}')


if __name__ == '__main__':
    main()
