"""
RSVG-DIOR -> GUI-SD jsonl converter.

The output schema matches the existing RRSIS-D jsonl files used by the
training/evaluation scripts:
  - solution.arguments.coordinate keeps pixel-level xyxy bbox.
  - assistant message uses normalized 0-1000 bbox_2d.
  - one record is created for every XML object/ref, not one per image.
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path


SYSTEM_PROMPT = "You are a helpful assistant."


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


def read_split_ids(path):
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def text_of(node, name, default=""):
    child = node.find(name)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def parse_xml_records(xml_path, image_prefix, sample_id_offset, ref_id_start):
    root = ET.parse(xml_path).getroot()
    filename = text_of(root, "filename") or f"{xml_path.stem}.jpg"
    size = root.find("size")
    if size is None:
        raise ValueError(f"missing <size> in {xml_path}")

    width = int(float(text_of(size, "width")))
    height = int(float(text_of(size, "height")))
    image_size = [width, height]
    image_path = os.path.join(image_prefix, filename)

    records = []
    for object_index, obj in enumerate(root.findall("object")):
        category = text_of(obj, "name", "unknown")
        description = text_of(obj, "description", category)
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"missing <bndbox> in {xml_path}, object {object_index}")

        bbox_xyxy = [
            float(text_of(box, "xmin")),
            float(text_of(box, "ymin")),
            float(text_of(box, "xmax")),
            float(text_of(box, "ymax")),
        ]
        bbox_xyxy = clip_xyxy(bbox_xyxy, image_size)
        norm_bbox = xyxy_to_norm1000(bbox_xyxy, image_size)

        ref_id = ref_id_start + object_index
        user_text = f"Locate {description}, output its bbox coordinates using JSON format."
        assistant_text = json.dumps({"bbox_2d": norm_bbox}, separators=(",", ": "))

        records.append({
            "solution": {
                "name": "rs_grounding",
                "arguments": {"action": "locate", "coordinate": bbox_xyxy},
            },
            "images": image_path,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            "additional_paras": json.dumps({
                "image_size": image_size,
                "platform": "satellite",
                "ui_type": "object",
                "category": category,
                "ref_id": ref_id,
                "source_dataset": "RSVG-DIOR",
                "image_id": xml_path.stem,
                "object_index": object_index,
            }, ensure_ascii=False),
            "sample_id": sample_id_offset + ref_id,
        })

    return records


def write_jsonl(path, records):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", default="/data/codes/gui_grounding/data/RSVG-DIOR")
    parser.add_argument("--out_dir", default="/data/codes/gui_grounding/data/rsvg_dior_full")
    parser.add_argument(
        "--image_prefix",
        default="/data/codes/gui_grounding/data/RSVG-DIOR/JPEGImages",
        help="Absolute image directory written into the jsonl images field.",
    )
    parser.add_argument("--sample_id_offset", type=int, default=300000)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    ann_dir = src_dir / "Annotations"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_ids = {
        "train": read_split_ids(src_dir / "train.txt"),
        "val": read_split_ids(src_dir / "val.txt"),
        "test": read_split_ids(src_dir / "test.txt"),
    }

    records = []
    ref_id_start = 0
    for xml_path in sorted(ann_dir.glob("*.xml")):
        image_records = parse_xml_records(
            xml_path=xml_path,
            image_prefix=args.image_prefix,
            sample_id_offset=args.sample_id_offset,
            ref_id_start=ref_id_start,
        )
        records.extend(image_records)
        ref_id_start += len(image_records)

    expected_ids = set(range(len(records)))
    actual_ids = set().union(*(set(ids) for ids in split_ids.values()))
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:20]
        extra = sorted(actual_ids - expected_ids)[:20]
        raise ValueError(
            f"split ids do not match flattened refs: records={len(records)}, "
            f"missing={missing}, extra={extra}"
        )

    record_by_id = {json.loads(r["additional_paras"])["ref_id"]: r for r in records}
    for split, ids in split_ids.items():
        out_path = out_dir / f"rsvg_{split}.jsonl"
        write_jsonl(out_path, [record_by_id[i] for i in ids])
        print(f"wrote {out_path}: {len(ids)} records")

    all_path = out_dir / "rsvg_all.jsonl"
    write_jsonl(all_path, records)
    print(f"wrote {all_path}: {len(records)} records")


if __name__ == "__main__":
    main()
