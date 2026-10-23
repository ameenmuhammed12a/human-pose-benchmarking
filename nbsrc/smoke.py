"""Concatenate the .py cells and run them headlessly to catch errors fast."""
import glob, os, sys, re
import matplotlib
matplotlib.use("Agg")
cells = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "cells", "*.py")))
upto = sys.argv[1] if len(sys.argv) > 1 else "999"
src = []
for c in cells:
    if os.path.basename(c)[:3] > upto:
        continue
    s = open(c, encoding="utf-8").read()
    s = re.sub(r"^%matplotlib inline$", "", s, flags=re.M)
    src.append(f"\n# ===== {os.path.basename(c)} =====\n" + s)
code = "\n".join(src)
open(os.path.join(os.path.dirname(__file__), "_smoke_concat.py"), "w", encoding="utf-8").write(code)
g = {"__name__": "__main__"}
exec(compile(code, "notebook_cells", "exec"), g)
print("\n>>> SMOKE OK")
