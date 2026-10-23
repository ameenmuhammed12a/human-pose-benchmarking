"""Execute pose_benchmarking.ipynb in place, keeping all outputs."""
import os, sys, time, nbformat
from nbclient import NotebookClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
path = os.path.join(ROOT, "pose_benchmarking.ipynb")
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=5400, kernel_name="python3",
                        resources={"metadata": {"path": ROOT}},
                        allow_errors=True, store_widget_state=False)
t0 = time.time()
client.execute()
nbformat.write(nb, path)
print(f"executed in {time.time()-t0:.0f}s -> {path}")

fails = 0
for i, c in enumerate(nb.cells):
    for out in c.get("outputs", []):
        if out.get("output_type") == "error":
            fails += 1
            print(f"\n!! ERROR in cell {i}: {out.get('ename')}: {out.get('evalue')}")
            print("   " + "\n   ".join(out.get("traceback", [])[-6:]))
print(f"\ncells with errors: {fails}")
sys.exit(1 if fails else 0)
