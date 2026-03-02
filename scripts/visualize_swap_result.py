#!/usr/bin/env python3
"""
Visualize swap or contradictory experiment: 2x2 grid of images with predictions.
Usage:
  python scripts/visualize_swap_result.py results/swap_experiment -o results/swap_experiment/fig_swap.png
  python scripts/visualize_swap_result.py results/contradictory_experiment -o results/contradictory_experiment/fig_contradictory.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def load_image(path: Path, max_size: int = 400):
    """Load and optionally resize for display."""
    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    h, w = arr.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        im = Image.fromarray(arr).resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        arr = np.array(im)
    return arr


def main():
    parser = argparse.ArgumentParser(description="Visualize swap/contradictory experiment")
    parser.add_argument("result_dir", type=Path, help="e.g. results/swap_experiment or results/contradictory_experiment")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output figure path (default: <result_dir>/fig_result.png)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    d = args.result_dir.resolve()
    if not d.is_dir():
        raise FileNotFoundError(f"Not a directory: {d}")

    result_file = d / "result.json"
    if not result_file.exists():
        raise FileNotFoundError(f"Missing {result_file}. Run the experiment first.")
    with open(result_file) as f:
        res = json.load(f)

    # Image paths
    orig_riding = d / "original_riding.png"
    orig_standing = d / "original_standing.png"
    c1_path = d / "composite1.png"
    c2_path = d / "composite2.png"
    for p in (orig_riding, orig_standing, c1_path, c2_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    mode = res.get("mode", "swap")
    pred1 = res.get("composite1_pred", "?")
    pred2 = res.get("composite2_pred", "?")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    # Order: original riding, original standing, composite 1, composite 2
    images = [
        load_image(orig_riding),
        load_image(orig_standing),
        load_image(c1_path),
        load_image(c2_path),
    ]
    # Human-readable labels: "riding" = person ON horse, "standing" = person BESIDE horse
    L = {"riding": "Person ON horse", "standing": "Person BESIDE horse"}
    pred1_readable = L.get(pred1, pred1)
    pred2_readable = L.get(pred2, pred2)
    titles = [
        "Original image A\n(Person ON horse)",
        "Original image B\n(Person BESIDE horse)",
        f"Composite 1: A's subject on B's scene\n→ Model said: {pred1_readable}",
        f"Composite 2: B's subject on A's scene\n→ Model said: {pred2_readable}",
    ]
    if mode == "contradictory":
        titles[2] = f"A's subject pasted on B's full image\n→ Model said: {pred1_readable}"
        titles[3] = f"B's subject pasted on A's full image\n→ Model said: {pred2_readable}"

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        "Does the model follow the main subject (foreground) or the scene (background)?\n"
        f"Experiment: {mode}",
        fontsize=11,
    )
    plt.tight_layout()

    out = args.output or (d / "fig_result.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
