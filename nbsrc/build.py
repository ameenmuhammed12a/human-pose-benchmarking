"""Assemble pose_benchmarking.ipynb from the cell sources in nbsrc/cells/*.py + *.md."""
import os, glob, json, nbformat

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CELLS = os.path.join(os.path.dirname(__file__), "cells")

def main():
    nb = nbformat.v4.new_notebook()
    nb.cells = []
    for path in sorted(glob.glob(os.path.join(CELLS, "*.*"))):
        src = open(path, encoding="utf-8").read().rstrip("\n")
        if not src.strip():
            continue
        if path.endswith(".md"):
            nb.cells.append(nbformat.v4.new_markdown_cell(src))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(src))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    out = os.path.join(ROOT, "pose_benchmarking.ipynb")
    nbformat.write(nb, out)
    print(f"wrote {out} with {len(nb.cells)} cells")

if __name__ == "__main__":
    main()
