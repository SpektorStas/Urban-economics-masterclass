"""Mechanically build the editable notebook from pipeline_source.py."""

import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5


root = Path(__file__).resolve().parents[1]
source = (root / "pipeline_source.py").read_text(encoding="utf-8")
notebook = {
    "cells": [], "metadata": {"kernelspec": {
        "display_name": "Python 3.11 (MCD masterclass)", "language": "python", "name": "mcd_masterclass"
    }}, "nbformat": 4, "nbformat_minor": 5,
}

kind = None
lines = []


def append_cell():
    if kind is None or not lines:
        return
    body = "".join(lines).strip("\n") + "\n"
    if kind == "markdown":
        body = "\n".join(
            line[2:] if line.startswith("# ") else line[1:] if line.startswith("#") else line
            for line in body.splitlines()
        ).strip("\n")
        cell = {"cell_type": "markdown", "metadata": {}, "source": body.splitlines(keepends=True)}
    else:
        cell = {"cell_type": "code", "metadata": {}, "source": body.splitlines(keepends=True),
                "execution_count": None, "outputs": []}
    cell["id"] = uuid5(NAMESPACE_URL, f"mcd-firms-cell-{len(notebook['cells'])}").hex[:8]
    notebook["cells"].append(cell)


for line in source.splitlines(keepends=True):
    if line.startswith("# %% [markdown]"):
        append_cell()
        kind, lines = "markdown", []
    elif line.startswith("# %%"):
        append_cell()
        kind, lines = "code", []
    else:
        lines.append(line)
append_cell()

target = root / "mcd_firms_pipeline.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Built {target} ({len(notebook['cells'])} cells)")

