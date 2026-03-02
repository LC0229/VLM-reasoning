#!/usr/bin/env python3
"""Create two placeholder images so you can run the swap script without real COCO images."""
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Run: pip install Pillow")
    raise

out_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
out_dir.mkdir(parents=True, exist_ok=True)
w, h = 224, 224
Image.new("RGB", (w, h), (180, 120, 80)).save(out_dir / "riding_placeholder.png")
Image.new("RGB", (w, h), (100, 150, 200)).save(out_dir / "standing_placeholder.png")
print(f"Created {out_dir}/riding_placeholder.png and standing_placeholder.png")
