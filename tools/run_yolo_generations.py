#!/usr/bin/env python3
"""YOLO generation comparison: YOLOv8-pose vs YOLO11-pose vs YOLO26-pose.

Runs the project's own pipeline cells unchanged (loaders, mapping, metrics,
evaluation) and benchmarks the three Ultralytics pose generations on the same
images under the identical protocol, including the JRDB-Pose occlusion split.

Usage:  python run_yolo_generations.py
Outputs (results/): yolo_generations_summary.csv, yolo_generations_occlusion.csv,
                    yolo_generations_comparison.png, yolo_generations_occlusion.png
"""
import os, sys, re, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if ROOT.name == "tools": ROOT = ROOT.parent
os.chdir(ROOT)
CELLS = ROOT / ("nbsrc_new/cells" if (ROOT / "nbsrc_new").is_dir() else "nbsrc/cells")

def run_cell(name, g):
    src = (CELLS / name).read_text(encoding="utf-8")
    src = re.sub(r"^%matplotlib.*$", "", src, flags=re.M)      # notebook magic -> no-op
    exec(compile(src, name, "exec"), g)

G = {"__name__": "__main__"}
import matplotlib
matplotlib.use("Agg")

for cell in ["011_config.py", "013_env.py", "103_loaders.py", "104_cp_loader.py",
             "105_jrdb_loader.py", "301_base.py", "302_yolo.py"]:
    print(f"--- exec {cell}")
    run_cell(cell, G)

np = G["np"]; pd = G["pd"]; plt = G["plt"]
YOLOPoseWrapper = G["YOLOPoseWrapper"]

# ---- datasets --------------------------------------------------------------------
crowdpose = G["CrowdPoseDataset"](G["CROWDPOSE_ANN_FILE"], G["CROWDPOSE_IMAGE_DIR"])
jrdb      = G["JRDBPoseDataset"](G["JRDB_LABEL_DIR"], G["JRDB_IMAGE_DIR"])
cp_samples = crowdpose.load(limit=G["N_CROWDPOSE_IMAGES"])
jr_samples = jrdb.load(limit=G["N_JRDB_IMAGES"])
print(f"\nCrowdPose: {len(cp_samples)} images | JRDB-Pose: {len(jr_samples)} images")
G["crowdpose"], G["jrdb"] = crowdpose, jrdb
G["cp_samples"], G["jr_samples"] = cp_samples, jr_samples
for cell in ["401_mapping.py", "403_to_common.py", "501_metrics.py", "502_evaluate.py"]:
    print(f"--- exec {cell}")
    run_cell(cell, G)
evaluate = G["evaluate"]
CROWDPOSE14_TO_COMMON = G["CROWDPOSE14_TO_COMMON"]
JRDB17_TO_COMMON = G["JRDB17_TO_COMMON"]
# register the same COCO-17 -> common mapping for the newer YOLO generations
MM = G["MODEL_MAPPINGS"]
MM["YOLO11-pose"] = MM["YOLOv8-pose"]
MM["YOLO26-pose"] = MM["YOLOv8-pose"]

# ---- the three YOLO generations ---------------------------------------------------
MODEL_DIR = G["MODEL_DIR"]
gens = [("YOLOv8-pose",  MODEL_DIR / "yolov8n-pose.pt"),
        ("YOLO11-pose",  MODEL_DIR / "yolo11n-pose.pt"),
        ("YOLO26-pose",  MODEL_DIR / "yolo26n-pose.pt")]
models = []
for name, w in gens:
    m = YOLOPoseWrapper(w)
    m.name = name
    models.append(m)
models = [m for m in models if m.available]
assert len(models) == 3, "all three YOLO generations must load"

# warm-up (excluded from timing)
warm = cp_samples[0].load_bgr()
for m in models:
    m.predict(warm)
print("warm-up done\n")

# ---- benchmark --------------------------------------------------------------------
summaries, recs = [], []
for ds, samples, gt_map in [(crowdpose, cp_samples, CROWDPOSE14_TO_COMMON),
                            (jrdb, jr_samples, JRDB17_TO_COMMON)]:
    for m in models:
        print(f"[run] {m.name} on {ds.name} ({len(samples)} images)...", flush=True)
        t0 = time.perf_counter()
        s, df = evaluate(m, ds, samples, gt_map)
        print(f"   PCK@0.5 {s['PCK@0.5_%']:.1f}% | OKS {s['OKS_mean(matched)']:.3f} | "
              f"recall {s['detection_recall_%']:.1f}% | missing {s['missing_kpt_rate_%']:.1f}% | "
              f"{s['FPS']:.1f} FPS | wall {time.perf_counter()-t0:.0f}s", flush=True)
        summaries.append(s)
        if not df.empty:
            pass  # records already carry model/dataset
            recs.append(df)

res = pd.DataFrame(summaries)
res.to_csv("results/yolo_generations_summary.csv", index=False)
records = pd.concat(recs, ignore_index=True)
print("\nrecords:", len(records), "| columns:", list(records.columns))

# ---- occlusion split (JRDB visibility labels), same as notebook Section 6 ---------
jr = records[records["dataset"] == "JRDB-Pose"].copy()
rows = []
for mname, gm in jr.groupby("model"):
    for v, label in [(2, "visible"), (1, "occluded"), (0, "invisible")]:
        grp = gm[gm["visibility"] == v]
        if grp.empty: continue
        pck = 100.0 * grp["correct_pck05"].mean()
        pck2 = 100.0 * grp["correct_pck02"].mean()
        miss = 100.0 * grp["missing"].mean()
        rows.append(dict(model=mname, group=label, keypoints=len(grp),
                         pck05=round(pck, 2), pck02=round(pck2, 2), missing=round(miss, 2)))
occ = pd.DataFrame(rows)
occ.to_csv("results/yolo_generations_occlusion.csv", index=False)
print(occ.to_string(index=False))

# drops
drop_rows = []
for mname in occ["model"].unique():
    v = occ[(occ.model == mname) & (occ.group == "visible")].iloc[0]
    o = occ[(occ.model == mname) & (occ.group == "occluded")].iloc[0]
    drop_rows.append(dict(model=mname, visible_pck=v.pck05, occluded_pck=o.pck05,
                          drop_pp=round(v.pck05 - o.pck05, 2),
                          missing_visible=v.missing, missing_occluded=o.missing))
drops = pd.DataFrame(drop_rows)
drops.to_csv("results/yolo_generations_occlusion_drop.csv", index=False)
print(drops.to_string(index=False))
print("\nDONE")
