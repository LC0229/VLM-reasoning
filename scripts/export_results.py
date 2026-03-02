#!/usr/bin/env python3
"""
Export evaluation results to research-ready formats: LaTeX tables, Markdown summary, CSV.
Usage:
  python scripts/export_results.py results/my_results.json
  python scripts/export_results.py results/my_results.json --out-dir results/generated
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_results(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def export_latex_tables(data: dict) -> str:
    """Generate LaTeX for main accuracy table and stability table."""
    lines = []

    # --- Table 1: Accuracy by condition ---
    if "object_level" in data:
        obj = data["object_level"]
        conds = obj.get("conditions", ["baseline", "blur", "mask", "crop"])
        accs = obj.get("accuracy", {})
        n = obj.get("n_samples", "—")
        lines.append("% --- Object-level (COCO) ---")
        lines.append("\\begin{table}[ht]")
        lines.append("\\centering")
        lines.append("\\caption{Object-level accuracy (\\%) on COCO subset. $n = " + str(n) + "$ images.}")
        lines.append("\\label{tab:object-accuracy}")
        lines.append("\\begin{tabular}{l" + "c" * len(conds) + "}")
        lines.append("\\toprule")
        lines.append("Condition & " + " & ".join(c.replace("_", " ").title() for c in conds) + " \\\\")
        lines.append("\\midrule")
        vals = [str(round(accs.get(c, 0), 1)) for c in conds]
        lines.append("Accuracy (\\%) & " + " & ".join(vals) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{table}")
        lines.append("")
    else:
        lines.append("% Add 'object_level' with keys: conditions, accuracy, n_samples")
        lines.append("")

    # --- Table 2: Relational accuracy by condition ---
    if "relational" in data:
        rel = data["relational"]
        conds = rel.get("conditions", ["baseline", "blur", "mask", "crop"])
        accs = rel.get("accuracy", {})
        n = rel.get("n_samples", "—")
        lines.append("% --- Relational (riding vs standing) ---")
        lines.append("\\begin{table}[ht]")
        lines.append("\\centering")
        lines.append("\\caption{Relational accuracy (\\%) on riding vs. standing. $n = " + str(n) + "$ images.}")
        lines.append("\\label{tab:relational-accuracy}")
        lines.append("\\begin{tabular}{l" + "c" * len(conds) + "}")
        lines.append("\\toprule")
        lines.append("Condition & " + " & ".join(c.replace("_", " ").title() for c in conds) + " \\\\")
        lines.append("\\midrule")
        vals = [str(round(accs.get(c, 0), 1)) for c in conds]
        lines.append("Accuracy (\\%) & " + " & ".join(vals) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{table}")
        lines.append("")
    else:
        lines.append("% Add 'relational' with keys: conditions, accuracy, n_samples")
        lines.append("")

    # --- Table 3: Stability metrics ---
    if "stability" in data:
        stab = data["stability"]
        conds = stab.get("conditions", ["blur", "mask", "crop"])
        lines.append("% --- Stability: P(correct after | correct before), P(correct after | wrong before) ---")
        lines.append("\\begin{table}[ht]")
        lines.append("\\centering")
        lines.append("\\caption{Stability under perturbations.}")
        lines.append("\\label{tab:stability}")
        lines.append("\\begin{tabular}{lcc}")
        lines.append("\\toprule")
        lines.append("Perturbation & $P(\\text{correct after} \\mid \\text{correct before})$ & $P(\\text{correct after} \\mid \\text{wrong before})$ \\\\")
        lines.append("\\midrule")
        for c in conds:
            s = stab.get(c, {})
            p_cc = s.get("P_correct_after_given_correct_before", 0)
            p_cw = s.get("P_correct_after_given_wrong_before", 0)
            lines.append(c.replace("_", " ").title() + " & " + f"{p_cc:.2f}" + " & " + f"{p_cw:.2f}" + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{table}")
    else:
        lines.append("% Add 'stability' with keys: conditions, and per-condition P_correct_after_given_correct_before, P_correct_after_given_wrong_before")

    return "\n".join(lines)


def export_markdown_summary(data: dict) -> str:
    """Generate a short Markdown summary for the report or README."""
    lines = ["# Results Summary", ""]
    if "object_level" in data:
        obj = data["object_level"]
        lines.append("## Object-level (COCO)")
        lines.append(f"- Samples: {obj.get('n_samples', '—')}")
        for cond, acc in obj.get("accuracy", {}).items():
            lines.append(f"- **{cond}**: {acc:.1f}%")
        lines.append("")
    if "relational" in data:
        rel = data["relational"]
        lines.append("## Relational (riding vs standing)")
        lines.append(f"- Samples: {rel.get('n_samples', '—')}")
        for cond, acc in rel.get("accuracy", {}).items():
            lines.append(f"- **{cond}**: {acc:.1f}%")
        lines.append("")
    if "stability" in data:
        lines.append("## Stability")
        for cond, s in data["stability"].items():
            if cond == "conditions" or not isinstance(s, dict):
                continue
            p1 = s.get("P_correct_after_given_correct_before", 0)
            p2 = s.get("P_correct_after_given_wrong_before", 0)
            lines.append(f"- **{cond}**: P(correct after | correct before) = {p1:.2f}, P(correct after | wrong before) = {p2:.2f}")
        lines.append("")
    return "\n".join(lines)


def export_csv(data: dict) -> str:
    """Flatten key numbers to CSV for plotting or further analysis."""
    rows = [["task", "condition", "metric", "value"]]
    if "object_level" in data:
        for cond, acc in data["object_level"].get("accuracy", {}).items():
            rows.append(["object", cond, "accuracy_pct", f"{acc:.2f}"])
    if "relational" in data:
        for cond, acc in data["relational"].get("accuracy", {}).items():
            rows.append(["relational", cond, "accuracy_pct", f"{acc:.2f}"])
    if "stability" in data:
        for cond, s in data["stability"].items():
            if cond == "conditions" or not isinstance(s, dict):
                continue
            rows.append(["stability", cond, "P_correct_after_given_correct_before", f"{s.get('P_correct_after_given_correct_before', 0):.4f}"])
            rows.append(["stability", cond, "P_correct_after_given_wrong_before", f"{s.get('P_correct_after_given_wrong_before', 0):.4f}"])
    return "\n".join(",".join(r) for r in rows)


def main():
    parser = argparse.ArgumentParser(description="Export results to LaTeX, Markdown, CSV.")
    parser.add_argument("results_json", type=Path, help="Path to results JSON file")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: same as results file)")
    parser.add_argument("--no-latex", action="store_true", help="Skip LaTeX table output")
    parser.add_argument("--no-md", action="store_true", help="Skip Markdown output")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV output")
    args = parser.parse_args()

    if not args.results_json.exists():
        print(f"Error: {args.results_json} not found.", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or args.results_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_results(args.results_json)

    if not args.no_latex:
        latex = export_latex_tables(data)
        out_path = out_dir / "tables.tex"
        out_path.write_text(latex, encoding="utf-8")
        print(f"Wrote {out_path}")

    if not args.no_md:
        md = export_markdown_summary(data)
        out_path = out_dir / "summary.md"
        out_path.write_text(md, encoding="utf-8")
        print(f"Wrote {out_path}")

    if not args.no_csv:
        csv = export_csv(data)
        out_path = out_dir / "results_flat.csv"
        out_path.write_text(csv, encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
