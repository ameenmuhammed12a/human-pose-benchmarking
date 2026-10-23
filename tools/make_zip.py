"""Package the deliverable into a single zip for handover.

    python tools/make_zip.py

Includes every source file, the executed notebook, the results and the small model weights.
Excludes the datasets (~4.8 GB) -- they are re-fetched with tools/setup_data.py, and
JRDB-Pose is CC BY-NC-SA with a registration request, so it should not be redistributed.
Also excludes the 209 MB OpenPose weights, which are one command away:
    python tools/setup_data.py --only openpose
"""
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "pose_benchmarking_block6.zip"

# (path relative to ROOT, description) -- files first, then whole directories.
FILES = [
    "RUN_ME.py",               # one-command setup + verification
    "pose_benchmarking.ipynb",
    "README.md",               # the first thing the recipient should open
    "requirements.txt",
]
DIRS = [
    "tools",           # setup_data, remote_zip, fetch_jrdb_*, make_zip
    "nbsrc",           # build.py, execute.py, smoke.py, cells/
    "results",         # 13 CSV tables + 8 PNG charts
    "supervisor_review",  # the four-question review pack
    "models",          # yolov8n-pose.pt + pose_landmarker_full.task
]
EXTRA = [
    ("report_assets/block6_report.html", "report/block6_progress_report.html"),
]

EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git", "data", "report_assets"}
EXCLUDE_SUFFIX = {".npy", ".part", ".pyc"}
EXCLUDE_NAMES = {"images.json", "_smoke_concat.py",
                 # 209 MB of CMU OpenPose weights. Excluded to keep the handover zip small;
                 # re-fetched in one command with:
                 #     python tools/setup_data.py --only openpose
                 "openpose_body_coco18.pth",
                 # the cached JRDB archive index; regenerated on demand, and only useful
                 # to someone re-downloading the frames
                 "archive_offsets.json"}


def keep(p: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in p.parts):
        return False
    if p.suffix in EXCLUDE_SUFFIX or p.name in EXCLUDE_NAMES:
        return False
    return True


def main():
    if OUT.exists():
        OUT.unlink()
    added = []
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in FILES:
            src = ROOT / rel
            if src.is_file():
                z.write(src, rel)
                added.append(rel)
        for d in DIRS:
            base = ROOT / d
            if not base.is_dir():
                continue
            for src in sorted(base.rglob("*")):
                if src.is_file() and keep(src.relative_to(ROOT)):
                    rel = src.relative_to(ROOT).as_posix()
                    z.write(src, rel)
                    added.append(rel)
        for rel, arc in EXTRA:
            src = ROOT / rel
            if src.is_file():
                z.write(src, arc)
                added.append(arc)

    print(f"Wrote {OUT.name}  ({OUT.stat().st_size/1e6:.1f} MB, {len(added)} files)\n")
    groups = {}
    for a in added:
        groups.setdefault(a.split("/")[0] if "/" in a else "(root)", []).append(a)
    for g in sorted(groups):
        total = sum((ROOT / a).stat().st_size for a in groups[g] if (ROOT / a).is_file())
        print(f"  {g:<14} {len(groups[g]):>3} files")
    print("\nNOT included (by design):")
    print("  data/          ~4.8 GB of CrowdPose + JRDB-Pose -- run tools/setup_data.py to fetch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
