#!/usr/bin/env python3
"""
对多对 (riding, standing) 图像【每组单独】做 swap，再汇总结果。
Process each (riding, standing) pair separately, then aggregate.
Requires --use-captions to get riding vs standing groups; otherwise use run_swap_coco for single pair.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_swap_coco import (
    get_person_horse_image_ids,
    union_bbox,
    _pick_riding_standing,
    PERSON_CID,
    HORSE_CID,
)


def _get_riding_standing_lists(coco_root, coco_api, img_ids, use_captions):
    """Return (riding_ids, standing_ids). If not use_captions, split person_horse into two halves (arbitrary)."""
    if not use_captions:
        mid = len(img_ids) // 2
        return img_ids[:mid], img_ids[mid:]  # arbitrary split
    cap_file = coco_root / "annotations" / "captions_val2017.json"
    if not cap_file.exists():
        return [], []
    with open(cap_file) as f:
        cap_data = json.load(f)
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
        if any(p in caps for p in riding_phrases) and not any(x in caps for x in ["standing next", "standing beside", "beside"]):
            riding_ids.append(iid)
        elif any(p in caps for p in standing_phrases):
            standing_ids.append(iid)
    return riding_ids, standing_ids


def _run_one_swap(coco_api, img_dir, riding_id, standing_id, processor, model, device, mode):
    """Run swap for one pair; return (pred1, pred2) 0=riding, 1=standing."""
    from PIL import Image
    from run_swap_and_contradictory import composite_swap, composite_contradictory

    ri = coco_api.loadImgs([riding_id])[0]
    si = coco_api.loadImgs([standing_id])[0]
    path_r = img_dir / ri["file_name"]
    path_s = img_dir / si["file_name"]
    if not path_r.exists() or not path_s.exists():
        return None
    bbox_r = union_bbox(coco_api, riding_id)
    bbox_s = union_bbox(coco_api, standing_id)
    if not bbox_r or not bbox_s:
        return None
    img_r = Image.open(path_r).convert("RGB")
    img_s = Image.open(path_s).convert("RGB")
    if mode == "swap":
        c1, c2 = composite_swap(img_r, img_s, bbox_r, bbox_s)
    else:
        c1, c2 = composite_contradictory(img_r, img_s, bbox_r, bbox_s)
    from run_swap_and_contradictory import run_clip
    prompts = ["a photo of a person riding a horse", "a photo of a person standing next to a horse"]
    preds, _ = run_clip(processor, model, [c1, c2], prompts, device)
    return (int(preds[0]), int(preds[1]))


def main():
    parser = argparse.ArgumentParser(description="Run swap on many (riding, standing) pairs; process each pair separately.")
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["swap", "contradictory"], default="swap")
    parser.add_argument("--max-pairs", type=int, default=10, help="Max number of (riding, standing) pairs to run")
    parser.add_argument("--save-dir", type=Path, default=Path("results/swap_batch"))
    parser.add_argument("--use-captions", action="store_true", help="Use captions to get riding/standing groups (recommended)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    coco_root = args.coco_root.resolve()
    ann_file = coco_root / "annotations" / "instances_val2017.json"
    img_dir = coco_root / "val2017"
    if not ann_file.exists() or not img_dir.is_dir():
        print("Error: need COCO root with annotations/ and val2017/", file=sys.stderr)
        sys.exit(1)

    from pycocotools.coco import COCO
    coco = COCO(str(ann_file))
    img_ids = get_person_horse_image_ids(coco)
    if len(img_ids) < 2:
        print("Error: need at least 2 person+horse images.", file=sys.stderr)
        sys.exit(1)

    riding_ids, standing_ids = _get_riding_standing_lists(coco_root, coco, img_ids, args.use_captions)
    if not riding_ids or not standing_ids:
        print("Error: need --use-captions and captions_val2017.json to get riding/standing groups.", file=sys.stderr)
        print("Or run without batch: python scripts/run_swap_coco.py ...", file=sys.stderr)
        sys.exit(1)

    from run_swap_and_contradictory import get_device, load_clip
    device = get_device(args.device or "mps")
    processor, model = load_clip()
    model = model.to(device)
    print(f"Riding group: {len(riding_ids)} images, Standing group: {len(standing_ids)} images. Device: {device}")

    n = min(args.max_pairs, len(riding_ids), len(standing_ids))
    results = []
    for i in range(n):
        riding_id = riding_ids[i]
        standing_id = standing_ids[i]
        out = _run_one_swap(coco, img_dir, riding_id, standing_id, processor, model, device, args.mode)
        if out is None:
            continue
        pred1, pred2 = out
        # 0=riding, 1=standing. follow_fg: composite1->riding, composite2->standing
        if pred1 == 0 and pred2 == 1:
            result_type = "follow_foreground"
        elif pred1 == 1 and pred2 == 0:
            result_type = "follow_background"
        else:
            result_type = "mixed"
        ri = coco.loadImgs([riding_id])[0]
        si = coco.loadImgs([standing_id])[0]
        results.append({
            "riding_id": riding_id,
            "standing_id": standing_id,
            "riding_file": ri["file_name"],
            "standing_file": si["file_name"],
            "composite1_pred": "riding" if pred1 == 0 else "standing",
            "composite2_pred": "riding" if pred2 == 0 else "standing",
            "result_type": result_type,
        })
        print(f"  Pair {i+1}/{n}: {ri['file_name']} + {si['file_name']} -> {result_type}")

    # 汇总
    follow_fg = sum(1 for r in results if r["result_type"] == "follow_foreground")
    follow_bg = sum(1 for r in results if r["result_type"] == "follow_background")
    mixed = sum(1 for r in results if r["result_type"] == "mixed")
    total = len(results)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.save_dir / "batch_results.json"
    out_path.write_text(json.dumps({"mode": args.mode, "n_pairs": total, "results": results, "summary": {"follow_foreground": follow_fg, "follow_background": follow_bg, "mixed": mixed}}, indent=2), encoding="utf-8")

    print("\n--- 汇总 / Summary ---")
    print(f"  对每组单独处理了 {total} 对 (riding, standing)。")
    print(f"  Follow foreground: {follow_fg}/{total}")
    print(f"  Follow background: {follow_bg}/{total}")
    print(f"  Mixed: {mixed}/{total}")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
