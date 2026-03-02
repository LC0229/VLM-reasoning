#!/usr/bin/env python3
"""
Plot accuracy by condition: object-level vs relational (Figure 1 for the report).
Usage:
  python scripts/plot_accuracy_comparison.py results/generated/results_flat.csv
  python scripts/plot_accuracy_comparison.py results/generated/results_flat.csv -o results/generated/fig_accuracy.pdf
"""
import argparse
import csv
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path, help="Path to results_flat.csv")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output figure path (default: same dir as CSV, name fig_accuracy.png)")
    args = parser.parse_args()

    if not args.csv_path.exists():
        raise FileNotFoundError(args.csv_path)

    # Parse CSV: keep only task, condition, value for accuracy_pct
    rows = []
    with open(args.csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("metric") == "accuracy_pct":
                rows.append({
                    "task": row["task"],
                    "condition": row["condition"],
                    "value": float(row["value"]),
                })

    # Build condition order (baseline first, then blur, mask, crop)
    order = ["baseline", "blur", "mask", "crop"]
    conditions = [c for c in order if any(r["condition"] == c for r in rows)]
    object_acc = []
    rel_acc = []
    for c in conditions:
        for r in rows:
            if r["condition"] == c and r["task"] == "object":
                object_acc.append(r["value"])
                break
        else:
            object_acc.append(0.0)
        for r in rows:
            if r["condition"] == c and r["task"] == "relational":
                rel_acc.append(r["value"])
                break
        else:
            rel_acc.append(0.0)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return

    x = np.arange(len(conditions))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, object_acc, w, label="Object-level", color="steelblue")
    ax.bar(x + w / 2, rel_acc, w, label="Relational (riding vs standing)", color="coral")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Condition")
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in conditions])
    ax.legend()
    ax.set_ylim(0, 100)
    fig.tight_layout()

    out = args.output or (args.csv_path.parent / "fig_accuracy.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
