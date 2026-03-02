#!/bin/bash
# Download COCO 2017 val images + annotations into data/coco (for run_swap_coco.py)
set -e
COCO_ROOT="${1:-data/coco}"
mkdir -p "$COCO_ROOT"
cd "$COCO_ROOT"

echo "Downloading COCO 2017 val images (~1GB)..."
curl -L -o val2017.zip "http://images.cocodataset.org/zips/val2017.zip"
unzip -q -o val2017.zip
rm -f val2017.zip

echo "Downloading annotations (~241MB)..."
curl -L -o annotations_trainval2017.zip "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
unzip -q -o annotations_trainval2017.zip
rm -f annotations_trainval2017.zip

echo "Done. COCO root: $(pwd)"
echo "Run: python scripts/run_swap_coco.py --coco-root $(pwd)"
