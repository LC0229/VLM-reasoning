# VLM Robustness Evaluation — How to Run, Techniques, and Gaps

This document summarizes how to run the project, which techniques are implemented, and what is still missing relative to the proposal.

---

## How to Run

### 1. Environment (Conda)

```bash
cd /path/to/VLM-reasoning
conda activate cv
pip install -r requirements.txt
```

### 2. MS COCO Data (required for full evaluation)

- Download **COCO Val2017** images and **instances_val2017.json** from [cocodataset.org](https://cocodataset.org/#download).
- Set paths (e.g. in the notebook or as env vars):
  - `COCO_ROOT`: root directory for COCO (e.g. `./data/coco`). Images are expected at `{COCO_ROOT}/{coco_image_subdir}/{file_name}`.
  - `COCO_ANN`: path to `instances_val2017.json`.

In the notebook, the config uses:
- `coco_root`: default `./data/coco`.
- `coco_image_subdir`: default `"val2017"` (so images at `./data/coco/val2017/*.jpg`). Set to `""` if your `coco_root` already points directly to the folder containing the images.
- `coco_ann_file`: default `./data/coco/annotations/instances_val2017.json`.

### 3. Run the notebook

```bash
jupyter notebook vlm_robustness_eval.ipynb
```

Run all cells in order. If COCO paths are not set or files are missing, the notebook will report “COCO data not found” and skip evaluation; set the paths and re-run to get full results.

### 4. Optional: quick run without COCO

Without COCO, the notebook still runs but does not perform baseline or perturbed evaluation. You can point `coco_root` and `coco_ann_file` to a small subset (e.g. a few images and a minimal annotation JSON) for a quick test.

---

## Techniques Used

- **Vision–language model**: **CLIP** (Radford et al., ICML 2021) via Hugging Face `transformers` (`CLIPModel` + `CLIPProcessor`). Used in a fixed, inference-only setup (no fine-tuning).
- **Datasets**: **MS COCO** subset filtered to categories *boat*, *airplane*, *horse* (strong background–object association). Uses official COCO annotations and bounding boxes.
- **Foreground/background separation** (so results are not “pure cv” rectangles):
  - **COCO segmentation**: When the annotation includes polygon/RLE segmentation, it is decoded to a precise object mask.
  - **rembg** (optional): Set `perturbation.use_rembg: True` and `pip install rembg` for AI-based foreground segmentation when COCO seg is missing.
  - **Feathered bbox**: If only a bbox is available, a soft mask boundary (`feather_sigma`) is used so blur/mask blend naturally.
- **Background perturbations** (applied using the foreground mask):
  - **Gaussian blur**: Blur the background and blend with the foreground using the (feathered) mask.
  - **Masking**: Replace background with a constant gray; same soft mask.
  - **Partial crop**: Crop to a padded region around the bbox to reduce background.
- **Evaluation**:
  - **Fixed text prompts**: e.g. “a photo of a boat”, “a photo of an airplane”, “a photo of a horse” — same prompts for baseline and perturbed runs.
  - **Metrics**: (1) **Accuracy**: predicted class (argmax over prompt similarities) vs ground-truth category; (2) **Image–text similarity**: CLIP similarity (e.g. softmax over logits) for the correct prompt.
- **Analysis**:
  - **Quantitative**: Baseline vs each perturbation (accuracy and mean similarity); table and bar plots.
  - **Qualitative**: Failure cases where the model is correct on the original image but wrong after a perturbation.

---

## What Is Still Missing

Relative to the proposal, the following are **not** implemented in this notebook:

1. **ImageNet-R / ImageNet-A**  
   No loader or evaluation for OOD robustness under distribution shift. Proposal: run the same (or no) perturbations on ImageNet-R/ImageNet-A and report accuracy/similarity.

2. **ObjectNet / CLEVR**  
   No integration of ObjectNet or CLEVR as object-centric control datasets to compare behavior when background shortcuts are reduced.

3. **Background replacement**  
   Only blur, mask, and crop are implemented. “Replacement with unrelated scenes” would require a foreground/background segmentation (e.g. rembg, SAM, or COCO masks) and pasting onto a new background; not implemented.

4. **Other VLMs**  
   Only CLIP is used. Repeating the same pipeline for LLaVA, BLIP-2, or other VLMs would allow comparison of robustness across models.

5. **Larger-scale and statistical reporting**  
   No confidence intervals, multiple seeds, or statistical tests; no full COCO-scale or per-category breakdown (only a single aggregate accuracy and mean similarity).

6. **COCO path handling**  
   The default `coco_root` may need to be set to the directory that actually contains the images (e.g. `.../val2017`) so that `image_path = os.path.join(coco_root, info["file_name"])` is correct for your download layout.

Implementing the items above would align the codebase fully with the proposal and support stronger claims about VLM robustness to background and context changes.
