"""Gather the outputs that answer the four supervisor questions into supervisor_review/.

    python tools/build_review_folder.py

The four questions, and where each is answered:
  Q1  Show a couple of samples for each dataset        -> 01_dataset_samples/
  Q2  Show the keypoints for each                      -> 02_keypoints/
  Q3  Unified keypoint set: show some result of that   -> 03_unified_keypoint_set/
  Q4  What are you comparing, and on what criteria?    -> 04_comparison_criteria/

It re-uses the notebook's own loader and drawing code (executed from nbsrc/cells/) so the
figures here cannot drift from the ones in the notebook. Only the cells up to the drawing
and mapping helpers are run -- no model inference -- so this takes a few seconds.
"""
import os
import re
import glob
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
CELLS = ROOT / "nbsrc" / "cells"
OUT = ROOT / "supervisor_review"

# Cells needed for loaders + drawing + the unified mapping. No model cells, no benchmark.
WANTED_PREFIXES = ("011", "013", "103", "104", "105", "107", "201",
                   "301", "401", "403")   # 301 supplies the native keypoint name lists


def run_notebook_cells() -> dict:
    """Execute the subset of notebook cells we need and return their namespace."""
    src = []
    for path in sorted(CELLS.glob("*.py")):
        if path.name[:3] not in WANTED_PREFIXES:
            continue
        code = re.sub(r"^%matplotlib inline$", "", path.read_text(encoding="utf-8"), flags=re.M)
        src.append(f"\n# ===== {path.name} =====\n{code}")
    ns = {"__name__": "__main__"}
    os.chdir(ROOT)                      # config cell uses Path.cwd()
    exec(compile("\n".join(src), "notebook_cells", "exec"), ns)
    return ns


def save_pair(ns, sample, skeleton, by_visibility, stem, raw_dir: Path, kp_dir: Path,
              pair_dir: Path, title: str):
    """Write the raw frame, the same frame with keypoints, and a side-by-side comparison."""
    img = sample.load_bgr()
    if img is None:
        print(f"    could not read {sample.image_path}")
        return None
    cv2.imwrite(str(raw_dir / f"{stem}_raw.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

    annotated_rgb = ns["render_gt"](sample, skeleton, color_by_visibility=by_visibility)
    cv2.imwrite(str(kp_dir / f"{stem}_keypoints.jpg"),
                cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original frame", fontsize=10)
    axes[1].imshow(annotated_rgb)
    axes[1].set_title("Ground-truth keypoints", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(pair_dir / f"{stem}_raw_vs_keypoints.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return annotated_rgb


def draw_unified_only(ns, sample, mapping, stem, out_dir: Path, dataset_name: str):
    """Draw ONLY the unified joints, labelled, to show the two datasets reduced to one set."""
    img = sample.load_bgr()
    if img is None:
        return
    common = ns["COMMON_KEYPOINTS"]
    canvas = img.copy()
    # Biggest person in the frame, so the labels are readable.
    person = max(sample.persons, key=lambda p: p.bbox[2] * p.bbox[3])
    uni = ns["gt_to_common"](person, mapping)
    ns["draw_skeleton"](canvas, uni, ns["COMMON_SKELETON"], color=(214, 120, 42),
                        radius=5, thickness=3, draw_invisible=False)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.imshow(rgb)
    for i, (x, y, v) in enumerate(uni):
        if v > 0 and np.isfinite(x):
            ax.annotate(f"{i}", (x, y), textcoords="offset points", xytext=(7, 5),
                        fontsize=7.5, color="#0b3f75",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.8))
    ax.axis("off")
    ax.set_title(f"{dataset_name} — the same 12 unified joints\n"
                 f"(native skeleton has {len(person.keypoint_names)} keypoints)", fontsize=10)
    legend = "   ".join(f"{i}:{n}" for i, n in enumerate(common))
    fig.text(0.5, 0.015, legend, ha="center", fontsize=6.6, family="monospace", color="#444")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_dir / f"{stem}_unified_12_joints.png", bbox_inches="tight", dpi=115)
    plt.close(fig)


# Hand-written parts of the pack that this script must NEVER delete:
#   - the whole 00_sample_selection/ demonstration
#   - every .md file anywhere in the pack (README, ABOUT_THE_UNIFIED_SET, CRITERIA, ...)
# Only the generated images and CSVs are cleared before a rebuild.
PRESERVE_DIRS = {"00_sample_selection"}


def clear_generated_files():
    """Delete the files this script generates, leaving hand-written content alone."""
    if not OUT.exists():
        return
    kept = removed = 0
    for path in sorted(OUT.rglob("*"), reverse=True):     # deepest first
        rel = path.relative_to(OUT)
        if rel.parts[0] in PRESERVE_DIRS:
            continue
        if path.is_file():
            if path.suffix.lower() == ".md":              # hand-written documentation
                kept += 1
                continue
            path.unlink()
            removed += 1
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    print(f"Cleared {removed} generated files; kept {kept} hand-written .md files "
          f"and {'/'.join(sorted(PRESERVE_DIRS))}/\n")


def main():
    clear_generated_files()
    print("Executing the notebook's loader / drawing / mapping cells ...\n")
    ns = run_notebook_cells()

    cp, jr = ns["crowdpose"], ns["jrdb"]
    cp_samples, jr_samples = ns["cp_samples"], ns["jr_samples"]

    d_raw_cp = OUT / "01_dataset_samples" / "crowdpose"
    d_raw_jr = OUT / "01_dataset_samples" / "jrdb_pose"
    d_kp_cp = OUT / "02_keypoints" / "crowdpose"
    d_kp_jr = OUT / "02_keypoints" / "jrdb_pose"
    d_pair = OUT / "02_keypoints" / "side_by_side"
    d_uni = OUT / "03_unified_keypoint_set"
    d_crit = OUT / "04_comparison_criteria"
    for d in (d_raw_cp, d_raw_jr, d_kp_cp, d_kp_jr, d_pair, d_uni, d_crit):
        d.mkdir(parents=True, exist_ok=True)

    # ---- Q1 + Q2: three samples per dataset, raw and with keypoints ----------------
    print("\nQ1/Q2  sample frames, raw and with ground-truth keypoints")
    n_each = 3
    cp_pick = ns["pick_samples"](cp_samples, n_each) if cp.samples else []
    jr_pick = ns["pick_samples"](jr_samples, n_each, spread_key="scene") if jr.samples else []

    for i, s in enumerate(cp_pick, 1):
        name = s.meta.get("file_name", s.image_path.name)
        ci = s.meta.get("crowdIndex")
        save_pair(ns, s, cp.skeleton, False, f"crowdpose_{i}", d_raw_cp, d_kp_cp, d_pair,
                  f"CrowdPose  {name}  —  {len(s.persons)} people"
                  + (f", crowdIndex {ci:.2f}" if ci is not None else ""))
        print(f"    CrowdPose sample {i}: {name} ({len(s.persons)} people)")

    for i, s in enumerate(jr_pick, 1):
        vis = np.concatenate([p.keypoints[:, 2] for p in s.persons]).astype(int)
        save_pair(ns, s, jr.skeleton, True, f"jrdb_{i}", d_raw_jr, d_kp_jr, d_pair,
                  f"JRDB-Pose  {s.meta['scene']}  frame {s.meta['frame']}  —  "
                  f"{len(s.persons)} people | visible {int((vis==2).sum())}, "
                  f"occluded {int((vis==1).sum())}, invisible {int((vis==0).sum())}")
        print(f"    JRDB sample {i}: {s.meta['scene']} frame {s.meta['frame']} "
              f"({len(s.persons)} people)")

    # ---- Q3: the unified set -------------------------------------------------------
    print("\nQ3     unified keypoint set")
    if cp_pick:
        draw_unified_only(ns, cp_pick[0], ns["CROWDPOSE14_TO_COMMON"], "crowdpose", d_uni,
                          "CrowdPose (native 14)")
    if jr_pick:
        draw_unified_only(ns, jr_pick[0], ns["JRDB17_TO_COMMON"], "jrdb_pose", d_uni,
                          "JRDB-Pose (native 17)")
    for f in ["keypoint_mapping.csv", "per_joint_pck.csv", "per_joint_pck.png"]:
        src = ROOT / "results" / f
        if src.is_file():
            shutil.copy2(src, d_uni / f)
            print(f"    copied {f}")

    # ---- Q4: criteria + every results table ---------------------------------------
    print("\nQ4     comparison criteria and results")
    for f in ["benchmark_summary.csv", "benchmark_comparison.png",
              "occlusion_by_visibility.csv", "occlusion_drop.csv", "occlusion_analysis.png",
              "occlusion_by_person.csv", "occlusion_by_person.png",
              "occlusion_by_crowd.csv", "accuracy_by_scale.csv", "crowd_and_scale.png",
              "occlusion_by_joint.csv", "occlusion_by_joint.png",
              "stability_summary.csv", "stability_by_joint.csv",
              "stability_comparison.png", "stability_by_joint.png",
              "per_keypoint_records.csv", "environment.csv"]:
        src = ROOT / "results" / f
        if src.is_file():
            shutil.copy2(src, d_crit / f)
            print(f"    copied {f}")

    n_files = sum(1 for _ in OUT.rglob("*") if _.is_file())
    print(f"\nWrote {n_files} files to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
