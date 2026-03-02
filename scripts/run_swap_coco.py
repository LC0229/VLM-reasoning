#!/usr/bin/env python3
"""
Run swap (or contradictory) experiment on COCO: pick one person+horse image as "riding"
and one as "standing", then run foreground/background swap and CLIP.
Uses COCO val2017. Needs COCO root with annotations/ and val2017/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# so "from run_swap_and_contradictory import ..." works when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

# We hardcode person+horse for "riding vs standing"; see WORKFLOW.md and docs/COCO_STRUCTURE_AND_GROUPS.md.
PERSON_CID = 1
HORSE_CID = 17


def get_person_horse_image_ids(coco_api):
    """Return list of image ids that have at least one person and one horse."""
    cat_ids = coco_api.getCatIds(catNms=["person", "horse"])
    img_ids = set()
    for cid in cat_ids:
        img_ids.update(coco_api.getImgIds(catIds=[cid]))
    # keep only images that have BOTH
    out = []
    for iid in img_ids:
        ann_ids = coco_api.getAnnIds(imgIds=[iid], catIds=[PERSON_CID, HORSE_CID])
        anns = coco_api.loadAnns(ann_ids)
        has_person = any(a["category_id"] == PERSON_CID for a in anns)
        has_horse = any(a["category_id"] == HORSE_CID for a in anns)
        if has_person and has_horse:
            out.append(iid)
    return out


def bbox_coco_to_xyxy(bbox):
    """COCO bbox [x, y, w, h] -> (x1, y1, x2, y2)."""
    x, y, w, h = bbox
    return (int(x), int(y), int(x + w), int(y + h))


def union_bbox(coco_api, img_id):
    """Union of all person and horse bboxes for this image. Return (x1,y1,x2,y2)."""
    ann_ids = coco_api.getAnnIds(imgIds=[img_id], catIds=[PERSON_CID, HORSE_CID])
    anns = coco_api.loadAnns(ann_ids)
    if not anns:
        return None
    boxes = [bbox_coco_to_xyxy(a["bbox"]) for a in anns]
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)


def _pick_riding_standing(coco_root, coco_api, img_ids, use_captions):
    """Pick one image as 'riding' and one as 'standing'. If use_captions, filter by caption text."""
    if not use_captions:
        return img_ids[0], img_ids[1]
    cap_file = coco_root / "annotations" / "captions_val2017.json"
    if not cap_file.exists():
        print("Warning: captions_val2017.json not found; using first two images.", file=sys.stderr)
        return img_ids[0], img_ids[1]
    with open(cap_file) as f:
        cap_data = json.load(f)
    # Build image_id -> list of caption texts
    id_to_caps = {}
    for ann in cap_data.get("annotations", []):
        iid = ann["image_id"]
        if iid not in id_to_caps:
            id_to_caps[iid] = []
        id_to_caps[iid].append(ann.get("caption", "").lower())
    riding_phrases = ["riding", "on horseback", "on a horse", "mounted"]
    standing_phrases = ["standing next to", "standing beside", "beside the horse", "next to a horse", "near the horse"]
    riding_ids = []
    standing_ids = []
    for iid in img_ids:
        if iid not in id_to_caps:
            continue
        caps = " ".join(id_to_caps[iid])
        if any(p in caps for p in riding_phrases) and not any(p in caps for p in ["standing next", "standing beside", "beside"]):
            riding_ids.append(iid)
        elif any(p in caps for p in standing_phrases):
            standing_ids.append(iid)
    if riding_ids and standing_ids:
        return riding_ids[0], standing_ids[0]
    print("Warning: no caption-based riding/standing pair found; using first two images.", file=sys.stderr)
    return img_ids[0], img_ids[1]


def main():
    parser = argparse.ArgumentParser(description="Run swap/contradictory on COCO person+horse")
    parser.add_argument("--coco-root", type=Path, required=True, help="COCO root (contains annotations/ and val2017/)")
    parser.add_argument("--mode", choices=["swap", "contradictory"], default="swap")
    parser.add_argument("--save-dir", type=Path, default=Path("results/swap_experiment"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-captions", action="store_true", help="Use captions to pick real riding vs standing images (needs annotations/captions_val2017.json)")
    args = parser.parse_args()

    coco_root = args.coco_root.resolve()
    ann_file = coco_root / "annotations" / "instances_val2017.json"
    img_dir = coco_root / "val2017"
    if not ann_file.exists():
        print(f"Error: {ann_file} not found. Download COCO val2017 and annotations.", file=sys.stderr)
        sys.exit(1)
    if not img_dir.is_dir():
        print(f"Error: {img_dir} not found.", file=sys.stderr)
        sys.exit(1)

    try:
        from pycocotools.coco import COCO
    except ImportError:
        print("Install: pip install pycocotools", file=sys.stderr)
        sys.exit(1)

    coco = COCO(str(ann_file))
    img_ids = get_person_horse_image_ids(coco)
    if len(img_ids) < 2:
        print("Error: need at least 2 images with person+horse in val2017.", file=sys.stderr)
        sys.exit(1)

    # Pick which two images to use. By default we take the first two (order from COCO);
    # They may not look like riding/standing; use --use-captions or see WORKFLOW.md.
    riding_id, standing_id = _pick_riding_standing(coco_root, coco, img_ids, args.use_captions)
    riding_info = coco.loadImgs([riding_id])[0]
    standing_info = coco.loadImgs([standing_id])[0]
    path_riding = img_dir / riding_info["file_name"]
    path_standing = img_dir / standing_info["file_name"]
    if not path_riding.exists() or not path_standing.exists():
        print(f"Error: image not found: {path_riding} or {path_standing}", file=sys.stderr)
        sys.exit(1)

    bbox_riding = union_bbox(coco, riding_id)
    bbox_standing = union_bbox(coco, standing_id)
    assert bbox_riding and bbox_standing

    # Run composite + CLIP
    from run_swap_and_contradictory import (
        get_device,
        load_clip,
        run_clip,
        composite_swap,
        composite_contradictory,
    )
    from PIL import Image

    img_riding = Image.open(path_riding).convert("RGB")
    img_standing = Image.open(path_standing).convert("RGB")
    device = get_device(args.device or "mps")
    print(f"COCO images: riding={path_riding.name} standing={path_standing.name}")
    print(f"Using device: {device}")

    if args.mode == "swap":
        c1, c2 = composite_swap(img_riding, img_standing, bbox_riding, bbox_standing)
        label1, label2 = "riding_fg+standing_bg", "standing_fg+riding_bg"
    else:
        c1, c2 = composite_contradictory(img_riding, img_standing, bbox_riding, bbox_standing)
        label1, label2 = "riding_fg_on_standing_bg", "standing_fg_on_riding_bg"

    args.save_dir.mkdir(parents=True, exist_ok=True)
    c1.save(args.save_dir / "composite1.png")
    c2.save(args.save_dir / "composite2.png")
    img_riding.save(args.save_dir / "original_riding.png")
    img_standing.save(args.save_dir / "original_standing.png")
    print(f"Saved to {args.save_dir}")

    prompts = [
        "a photo of a person riding a horse",
        "a photo of a person standing next to a horse",
    ]
    processor, model = load_clip()
    preds, logits = run_clip(processor, model, [c1, c2], prompts, device)
    names = ["riding", "standing"]
    print("\n--- Results ---")
    print(f"Composite 1 ({label1}) -> predicted: {names[preds[0]]}")
    print(f"Composite 2 ({label2}) -> predicted: {names[preds[1]]}")
    if preds[0] == 0 and preds[1] == 1:
        print("  Model follows FOREGROUND.")
    elif preds[0] == 1 and preds[1] == 0:
        print("  Model follows BACKGROUND.")
    else:
        print("  Mixed.")
    print("Logits:", logits.tolist())

    import json
    (args.save_dir / "result.json").write_text(
        json.dumps({
            "mode": args.mode,
            "riding_image": riding_info["file_name"],
            "standing_image": standing_info["file_name"],
            "composite1_pred": names[preds[0]],
            "composite2_pred": names[preds[1]],
            "logits": logits.tolist(),
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
