# Environment setup for VLM-reasoning

## Option 1: venv (recommended)

From the repo root:

```bash
# Create and activate (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Or use the script:

```bash
bash scripts/setup_env.sh
# then
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Option 2: conda

```bash
conda create -n vlmreasoning python=3.10 -y
conda activate vlmreasoning
pip install -r requirements.txt
```

## Verify

```bash
python -c "import torch; import transformers; from PIL import Image; print('OK')"
python scripts/run_swap_coco.py --coco-root data/coco  # needs COCO downloaded
```

## Notes

- **MPS** (Apple Silicon GPU) is used automatically by the swap script when available.
- COCO data goes in `data/coco/` (see `scripts/download_coco_val.sh`).
