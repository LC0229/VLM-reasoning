# Results

This folder holds evaluation results and generated report assets.

## Files

| File | Description |
|------|-------------|
| `example_results.json` | Example schema for the results JSON. Replace with your own numbers. |
| `my_results.json` | **You create this** — save from your notebook or eval script (see schema below). |
| `generated/` | Output of `scripts/export_results.py` (tables.tex, summary.md, results_flat.csv). Optional: `python scripts/plot_accuracy_comparison.py results/generated/results_flat.csv -o results/generated/fig_accuracy.png` for the bar chart. |

## How to generate report-ready outputs

1. Run your evaluation (e.g. `vlm_robustness_eval.ipynb`) and save a JSON in this folder with the structure of `example_results.json`:
   - `object_level`: `n_samples`, `conditions`, `accuracy` (condition → %)
   - `relational`: same
   - `stability` (optional): per-condition `P_correct_after_given_correct_before`, `P_correct_after_given_wrong_before`

2. From the repo root:
   ```bash
   python scripts/export_results.py results/my_results.json --out-dir results/generated
   ```

3. Use `results/generated/tables.tex` in your report (add `\usepackage{booktabs}` if needed), and `summary.md` / `results_flat.csv` for quick reference or plotting.

See **docs/RESEARCH_OUTPUT.md** for the full research-output checklist and report structure.
