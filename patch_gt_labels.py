#!/usr/bin/env python3
"""Add ground-truth (GT) labels to 6_color_accessible_clip.ipynb."""
import json
import sys

path = "6_color_accessible_clip.ipynb"
try:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
except Exception as e:
    print(f"Failed to load notebook: {e}", file=sys.stderr)
    sys.exit(1)

def join_source(cell):
    return "" if cell["cell_type"] != "code" else "".join(cell.get("source", []))

def set_source(cell, new_lines):
    if cell["cell_type"] != "code":
        return
    cell["source"] = [line if line.endswith("\n") else line + "\n" for line in new_lines]

changes = []

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = join_source(cell)
    lines = [s.rstrip("\n") for s in cell.get("source", [])]
    new_lines = None

    # 6b: add GT explanation before "Accuracy vs GT:"
    if "Accuracy vs GT:" in src and "Ground truth:" in src and "(GT = correct answer" not in src:
        for j, line in enumerate(lines):
            if line.strip() == 'print("Accuracy vs GT:")':
                # insert before this line
                lines.insert(j, 'print("(GT = correct answer from dataset: most frequent COCO category per image.)")')
                new_lines = lines
                changes.append("6b: added GT explanation")
                break

    # 7b: add [GT: ...] to failure print
    if "failures (" in src and "orig}  ->  {after}" in src and "[GT:" not in src:
        for j, line in enumerate(lines):
            if "fname}:  {orig}  ->  {after}" in line and "print" in line:
                # add gt_lab and append [GT: {gt_lab}] to the print
                indent = len(line) - len(line.lstrip())
                prefix = " " * indent
                # insert gt_lab before this print
                lines.insert(j, prefix + 'gt_lab = r.get("gt") or "N/A"')
                j2 = j + 1
                old_print = lines[j2]
                if 'print(f"  {i+1}. {fname}:  {orig}  ->  {after}")' in old_print:
                    lines[j2] = old_print.replace(
                        'print(f"  {i+1}. {fname}:  {orig}  ->  {after}")',
                        'print(f"  {i+1}. {fname}:  {orig}  ->  {after}  [GT: {gt_lab}]")'
                    )
                new_lines = lines
                changes.append("7b: added GT to failure lines")
                break

    # 7c: add | GT: {gt_lab} to plot titles
    if "set_title(f\"Original: {orig_pred}\"" in src and "| GT:" not in src:
        for j, line in enumerate(lines):
            if 'set_title(f"Original: {orig_pred}"' in line:
                indent = len(line) - len(line.lstrip())
                prefix = " " * indent
                lines.insert(j, prefix + 'gt_lab = r.get("gt") or "N/A"')
                j2 = j + 1
                lines[j2] = line.replace(
                    'set_title(f"Original: {orig_pred}", fontsize=10)',
                    'set_title(f"Original: {orig_pred} | GT: {gt_lab}", fontsize=10)'
                )
                # fix the "After" title in same cell
                for k in range(j2 + 1, len(lines)):
                    if 'set_title(f"After {transform_name}: {after_pred}"' in lines[k]:
                        lines[k] = lines[k].replace(
                            'set_title(f"After {transform_name}: {after_pred}", fontsize=10)',
                            'set_title(f"After {transform_name}: {after_pred} | GT: {gt_lab}", fontsize=10)'
                        )
                        break
                new_lines = lines
                changes.append("7c: added GT to plot titles")
                break

    if new_lines is not None:
        set_source(cell, new_lines)

if not changes:
    print("No target cells found to patch (or already patched).")
    sys.exit(0)

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Patched:", ", ".join(changes))
