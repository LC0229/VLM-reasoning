#!/usr/bin/env python3
"""
Exploratory experiments: Swap (foreground A + background B) and Contradictory background.
Answers: Does the model follow foreground or background?

Usage:
  # Swap: paste riding foreground onto standing background, and vice versa.
  python scripts/run_swap_and_contradictory.py --mode swap \\
    --riding path/to/riding.jpg --standing path/to/standing.jpg \\
    --bbox-riding "x1,y1,x2,y2" --bbox-standing "x1,y1,x2,y2" [--save-dir out/]

  # Contradictory: riding foreground on full standing image (as bg), and vice versa.
  python scripts/run_swap_and_contradictory.py --mode contradictory \\
    --riding path/to/riding.jpg --standing path/to/standing.jpg \\
    --bbox-riding "x1,y1,x2,y2" --bbox-standing "x1,y1,x2,y2" [--save-dir out/]

Bbox format: x1,y1,x2,y2 (top-left and bottom-right in pixel coordinates).
If bbox not given, full image is used as foreground.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_bbox(s: str | None, im: Image.Image):
    """Parse 'x1,y1,x2,y2' or None -> (x1,y1,x2,y2). If None, use full image."""
    w, h = im.size
    if not s or s.strip() == "":
        return (0, 0, w, h)
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be x1,y1,x2,y2")
    return tuple(parts)


def crop_to_bbox(im: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    return im.crop((x1, y1, x2, y2))


def composite_swap(
    img_riding: Image.Image,
    img_standing: Image.Image,
    bbox_riding: tuple[int, int, int, int],
    bbox_standing: tuple[int, int, int, int],
) -> tuple[Image.Image, Image.Image]:
    """
    Composite 1: standing image with standing_bbox replaced by resized riding foreground.
    Composite 2: riding image with riding_bbox replaced by resized standing foreground.
    """
    # Riding foreground = crop from riding image
    fg_riding = crop_to_bbox(img_riding, bbox_riding)
    fg_standing = crop_to_bbox(img_standing, bbox_standing)

    x1_s, y1_s, x2_s, y2_s = bbox_standing
    x1_r, y1_r, x2_r, y2_r = bbox_riding
    w_s, h_s = x2_s - x1_s, y2_s - y1_s
    w_r, h_r = x2_r - x1_r, y2_r - y1_r

    # Composite 1: standing background + riding foreground (paste riding fg into standing bbox)
    out1 = img_standing.copy()
    fg_riding_resized = fg_riding.resize((w_s, h_s), Image.Resampling.LANCZOS)
    out1.paste(fg_riding_resized, (x1_s, y1_s))
    # Composite 2: riding background + standing foreground
    out2 = img_riding.copy()
    fg_standing_resized = fg_standing.resize((w_r, h_r), Image.Resampling.LANCZOS)
    out2.paste(fg_standing_resized, (x1_r, y1_r))

    return out1, out2


def composite_contradictory(
    img_riding: Image.Image,
    img_standing: Image.Image,
    bbox_riding: tuple[int, int, int, int],
    bbox_standing: tuple[int, int, int, int],
) -> tuple[Image.Image, Image.Image]:
    """
    Contradictory background: paste foreground of one image onto the *full* other image (resized).
    Out1: riding foreground on full standing image (as background).
    Out2: standing foreground on full riding image (as background).
    """
    fg_riding = crop_to_bbox(img_riding, bbox_riding)
    fg_standing = crop_to_bbox(img_standing, bbox_standing)
    w_r, h_r = img_riding.size
    w_s, h_s = img_standing.size

    # Out1: background = full standing image (resized to riding size); paste riding fg at riding bbox
    x1_r, y1_r, x2_r, y2_r = bbox_riding
    bw_r, bh_r = x2_r - x1_r, y2_r - y1_r
    bg_standing = img_standing.resize((w_r, h_r), Image.Resampling.LANCZOS)
    out1 = bg_standing.copy()
    fg_riding_resized = fg_riding.resize((bw_r, bh_r), Image.Resampling.LANCZOS)
    out1.paste(fg_riding_resized, (x1_r, y1_r))

    # Out2: background = full riding image (resized to standing size); paste standing fg at standing bbox
    x1_s, y1_s, x2_s, y2_s = bbox_standing
    bw_s, bh_s = x2_s - x1_s, y2_s - y1_s
    bg_riding = img_riding.resize((w_s, h_s), Image.Resampling.LANCZOS)
    out2 = bg_riding.copy()
    fg_standing_resized = fg_standing.resize((bw_s, bh_s), Image.Resampling.LANCZOS)
    out2.paste(fg_standing_resized, (x1_s, y1_s))

    return out1, out2


def get_device(prefer: str | None = None) -> str:
    """Prefer MPS (Apple) > CUDA > CPU. Use prefer if given and available."""
    import torch
    if prefer:
        if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if prefer == "cuda" and torch.cuda.is_available():
            return "cuda"
        if prefer == "cpu":
            return "cpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_clip():
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        print("Install: pip install transformers", file=sys.stderr)
        raise
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return processor, model


def run_clip(processor, model, images: list[Image.Image], prompts: list[str], device: str):
    """Run CLIP on images with prompts. Returns list of (image_idx -> prompt_idx with max score)."""
    import torch
    model = model.to(device)
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)
    with torch.no_grad():
        out = model(**inputs)
    # logits_per_image: (n_images, n_prompts)
    logits = out.logits_per_image.cpu().numpy()
    preds = logits.argmax(axis=1)
    return preds.tolist(), logits


def main():
    parser = argparse.ArgumentParser(description="Swap and contradictory background experiments")
    parser.add_argument("--mode", choices=["swap", "contradictory"], required=True)
    parser.add_argument("--riding", type=Path, required=True, help="Path to riding image")
    parser.add_argument("--standing", type=Path, required=True, help="Path to standing image")
    parser.add_argument("--bbox-riding", type=str, default=None, help="x1,y1,x2,y2 for riding image")
    parser.add_argument("--bbox-standing", type=str, default=None, help="x1,y1,x2,y2 for standing image")
    parser.add_argument("--save-dir", type=Path, default=None, help="Save composite images here")
    parser.add_argument("--device", type=str, default=None, help="Device: mps, cuda, or cpu. Default: auto (mps if available, else cuda, else cpu)")
    args = parser.parse_args()

    if not args.riding.exists():
        print(f"Error: {args.riding} not found", file=sys.stderr)
        sys.exit(1)
    if not args.standing.exists():
        print(f"Error: {args.standing} not found", file=sys.stderr)
        sys.exit(1)

    img_riding = Image.open(args.riding).convert("RGB")
    img_standing = Image.open(args.standing).convert("RGB")
    bbox_riding = parse_bbox(args.bbox_riding, img_riding)
    bbox_standing = parse_bbox(args.bbox_standing, img_standing)

    if args.mode == "swap":
        c1, c2 = composite_swap(img_riding, img_standing, bbox_riding, bbox_standing)
        label1, label2 = "riding_fg+standing_bg", "standing_fg+riding_bg"
    else:
        c1, c2 = composite_contradictory(img_riding, img_standing, bbox_riding, bbox_standing)
        label1, label2 = "riding_fg_on_standing_bg", "standing_fg_on_riding_bg"

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        c1.save(args.save_dir / "composite1.png")
        c2.save(args.save_dir / "composite2.png")
        print(f"Saved composite images to {args.save_dir}")

    # CLIP
    prompts = [
        "a photo of a person riding a horse",
        "a photo of a person standing next to a horse",
    ]
    processor, model = load_clip()
    device = get_device(args.device or "mps")
    print(f"Using device: {device}")
    preds, logits = run_clip(processor, model, [c1, c2], prompts, device)

    # Report
    names = ["riding", "standing"]
    print("\n--- Results ---")
    print(f"Composite 1 ({label1}) -> predicted: {names[preds[0]]}")
    print(f"Composite 2 ({label2}) -> predicted: {names[preds[1]]}")
    print("\nInterpretation:")
    if args.mode == "swap":
        if preds[0] == 0 and preds[1] == 1:
            print("  Model follows FOREGROUND (riding fg + standing bg -> riding; standing fg + riding bg -> standing).")
        elif preds[0] == 1 and preds[1] == 0:
            print("  Model follows BACKGROUND (composite 1 -> standing, composite 2 -> riding).")
        else:
            print("  Mixed or same prediction for both; check logits.")
    else:
        print("  Same logic: does prediction follow foreground or background?")
    print("\nLogits (image x prompt):", logits.tolist())

    # Save JSON for report
    out_json = {
        "mode": args.mode,
        "composite1": {"label": label1, "predicted": names[preds[0]], "pred_idx": int(preds[0])},
        "composite2": {"label": label2, "predicted": names[preds[1]], "pred_idx": int(preds[1])},
        "prompts": prompts,
        "logits": [[float(x) for x in row] for row in logits],
    }
    if args.save_dir:
        (args.save_dir / "result.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")
        print(f"Saved result.json to {args.save_dir}")


if __name__ == "__main__":
    main()
