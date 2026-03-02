#!/bin/bash
# Create and activate a virtual environment for VLM-reasoning (run from repo root)
set -e
cd "$(dirname "$0")/.."

VENV_DIR="${1:-.venv}"
echo "Creating venv at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
echo ""
echo "Done. Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Then run e.g.:"
echo "  python scripts/run_swap_coco.py --coco-root data/coco"
