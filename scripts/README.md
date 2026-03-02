# Scripts

## Two different experiments (different commands)

| Experiment | Command | What it does |
|------------|---------|--------------|
| **1. Swap** | `python scripts/run_swap_coco.py --coco-root data/coco` | Picks two COCO person+horse images, builds **riding fg + standing bg** and **standing fg + riding bg**, runs CLIP. Saves to `results/swap_experiment/`. |
| **2. Contradictory** | `python scripts/run_swap_coco.py --coco-root data/coco --mode contradictory --save-dir results/contradictory_experiment` | Same two images; pastes **riding fg on full standing image** (and vice versa). Saves to `results/contradictory_experiment/`. |
| **3. Batch (每组单独)** | `python scripts/run_swap_coco_batch.py --coco-root data/coco --use-captions --max-pairs 10` | 对多对 (riding, standing) **每对单独**跑 swap，汇总到 `results/swap_batch/batch_results.json`. 需 captions 分出 riding/standing 组. |

**Visualize either result (2×2 grid with predictions):**
```bash
python scripts/visualize_swap_result.py results/swap_experiment -o results/swap_experiment/fig_result.png
python scripts/visualize_swap_result.py results/contradictory_experiment -o results/contradictory_experiment/fig_result.png
```

**Main project results (object + relational accuracy by condition):** use `export_results.py` and `plot_accuracy_comparison.py` on your `results/my_results.json` → tables + bar chart. See results/README.md.

---

## Exploratory experiments (Section 8 of DIRECTIONS_AND_PRIORITIES.md)

### Swap + Contradictory — `run_swap_coco.py` (COCO) or `run_swap_and_contradictory.py` (custom images)

Implements:
- **互换实验 (Swap):** 骑马前景 + 站旁边背景 / 站旁边前景 + 骑马背景 → 看模型跟前景还是背景走。
- **矛盾背景 (Contradictory):** 骑马前景贴在「站旁边」整张图上当背景，反之亦然。

**Usage (from repo root):**

```bash
# Swap experiment (need two images + bboxes from your COCO riding/standing subset)
python scripts/run_swap_and_contradictory.py --mode swap \
  --riding path/to/riding.jpg --standing path/to/standing.jpg \
  --bbox-riding "100,80,400,350" --bbox-standing "120,90,380,340" \
  --save-dir results/swap_experiment

# Contradictory background
python scripts/run_swap_and_contradictory.py --mode contradictory \
  --riding path/to/riding.jpg --standing path/to/standing.jpg \
  --bbox-riding "100,80,400,350" --bbox-standing "120,90,380,340" \
  --save-dir results/contradictory_experiment
```

- **Bbox:** `x1,y1,x2,y2` in pixels (from COCO person+horse union bbox). If omitted, full image is used as foreground.
- **Output:** Prints which composite was predicted as riding vs standing; saves composite images and `result.json` if `--save-dir` is set.

**Interpretation:** If model follows *foreground*: composite1 (riding fg + standing bg) → riding, composite2 → standing. If it follows *background*, you get the opposite. See doc Section 8.

---

### Other exploratory ideas (not in this script)

- **Mask content** (灰/白/纹理/错误场景): Change what you fill the masked background with in your existing notebook (different fill color or paste another image in the background region).
- **Blur strength curve / Progressive masking:** Vary sigma or % masked in your existing eval loop and plot accuracy vs parameter.
- **Background vs relation region:** In your notebook, add a perturbation that masks only the relation-critical region (e.g. person–horse contact) instead of the full background; compare accuracy.

See **docs/DIRECTIONS_AND_PRIORITIES.md** Section 8 for full list.

---

## Results export and figure

- `export_results.py` — results JSON → LaTeX tables, summary.md, CSV.
- `plot_accuracy_comparison.py` — CSV → bar chart (object vs relational accuracy by condition).

Run from repo root; see **results/README.md**.
