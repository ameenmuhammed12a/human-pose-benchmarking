"""One-command setup: fetch the model weights and both datasets.

    python tools/setup_data.py                 # everything
    python tools/setup_data.py --only models   # just the model weights
    python tools/setup_data.py --skip crowdpose-images

Every stage is idempotent: anything already on disk is left alone, so the script
can be re-run safely after an interrupted download.

Stages
------
  models            YOLOv8n / YOLO11n / YOLO26n-pose weights + the MediaPipe
                    pose landmarker                                       (~30 MB)
  crowdpose-anns    CrowdPose COCO-style annotations                      (~131 MB)
  crowdpose-images  CrowdPose images, from the official Google Drive      (~2.4 GB)
  jrdb-labels       JRDB-Pose per-scene label JSONs                       (~28 MB)
  jrdb-images       JRDB-Pose frames, pulled out of a 21 GB remote zip    (~20 MB)

The two JRDB stages read only the parts they need out of the published archives
over HTTP range requests, so neither the 229 MB label zip nor the 21 GB image zip
is downloaded in full.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
CP_DIR = ROOT / "data" / "crowdpose"
JR_DIR = ROOT / "data" / "jrdb"

YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt"
# Newer YOLO generations: YOLO26 is benchmarked in the notebook, YOLO11 only in
# tools/run_yolo_generations.py. Both need ultralytics >= 8.4 to load.
YOLO26_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt"
YOLO11_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt"
MP_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
          "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
CP_ANN_URL = "https://download.openmmlab.com/mmpose/datasets/crowdpose_annotations.tar"
CP_IMAGES_GDRIVE_ID = "1VprytECcLtU4tKP32SYi_7oDRbw7yUTL"
# The original CMU OpenPose COCO-18 weights (pose_iter_440000), converted to a PyTorch
# checkpoint that keeps the Caffe layer names. Used by the notebook's PyTorch OpenPose
# backend when no compiled C++ build is available -- see Section 3.3 of the notebook.
OPENPOSE_URL = ("https://huggingface.co/lllyasviel/Annotators/resolve/main/"
                "body_pose_model.pth")

STAGES = ["models", "openpose", "crowdpose-anns", "crowdpose-images",
          "jrdb-labels", "jrdb-images", "jrdb-sequences"]


def say(msg=""):
    print(msg, flush=True)


def download(url: str, dest: Path, label: str) -> bool:
    """Stream a URL to dest with a progress line. Skips if dest already exists."""
    if dest.is_file() and dest.stat().st_size > 0:
        say(f"    already present: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    say(f"    downloading {label} ...")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r      {done/1e6:8.1f} / {total/1e6:.1f} MB "
                          f"({100*done/total:5.1f}%)", end="", flush=True)
                else:
                    print(f"\r      {done/1e6:8.1f} MB", end="", flush=True)
        print()
        tmp.replace(dest)
        return True
    except Exception as exc:
        print()
        say(f"    FAILED: {type(exc).__name__}: {exc}")
        tmp.unlink(missing_ok=True)
        return False


# --------------------------------------------------------------------------- stages
def stage_models() -> bool:
    say("[1/7] Model weights")
    ok = download(YOLO_URL, MODEL_DIR / "yolov8n-pose.pt", "yolov8n-pose.pt (6.8 MB)")
    ok &= download(YOLO26_URL, MODEL_DIR / "yolo26n-pose.pt", "yolo26n-pose.pt (7.9 MB)")
    ok &= download(YOLO11_URL, MODEL_DIR / "yolo11n-pose.pt", "yolo11n-pose.pt (6.3 MB)")
    ok &= download(MP_URL, MODEL_DIR / "pose_landmarker_full.task",
                   "pose_landmarker_full.task (9.4 MB)")
    return ok


def stage_openpose() -> bool:
    say("[2/7] OpenPose weights  (original CMU COCO-18, 209 MB)")
    return download(OPENPOSE_URL, MODEL_DIR / "openpose_body_coco18.pth",
                    "openpose_body_coco18.pth (209 MB)")


def stage_jrdb_sequences() -> bool:
    say("[7/7] JRDB-Pose consecutive-frame sequences  (for the stability analysis)")
    say("      This reads the archive index first, so it is quiet for a minute or two.")
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import fetch_jrdb_sequences
        fetch_jrdb_sequences.main()
    except Exception as exc:
        say(f"    FAILED: {type(exc).__name__}: {exc}")
        return False
    return len(list((JR_DIR / "sequences").rglob("*.jpg"))) > 0


def stage_crowdpose_anns() -> bool:
    say("[3/7] CrowdPose annotations")
    target = CP_DIR / "crowdpose_annotations" / "mmpose_crowdpose_val.json"
    if target.is_file():
        say(f"    already extracted: {target.name}")
        return True
    tar_path = CP_DIR / "crowdpose_annotations.tar"
    if not download(CP_ANN_URL, tar_path, "crowdpose_annotations.tar (131 MB)"):
        return False
    say("    extracting ...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(CP_DIR)          # archive contains only .json files
    say(f"    -> {target.parent}")
    return target.is_file()


def stage_crowdpose_images() -> bool:
    say("[4/7] CrowdPose images  (~2.4 GB - this is the slow one)")
    img_dir = CP_DIR / "images"
    if img_dir.is_dir() and len(list(img_dir.glob("*.jpg"))) > 19000:
        say(f"    already extracted: {len(list(img_dir.glob('*.jpg')))} images")
        return True
    zip_path = CP_DIR / "crowdpose_images.zip"
    if not (zip_path.is_file() and zip_path.stat().st_size > 2_000_000_000):
        try:
            import gdown  # noqa: F401
        except ImportError:
            say("    gdown is required for the Google Drive download.")
            say("    install it with:  pip install gdown")
            return False
        say("    downloading from Google Drive via gdown ...")
        CP_DIR.mkdir(parents=True, exist_ok=True)
        rc = subprocess.call([sys.executable, "-m", "gdown", CP_IMAGES_GDRIVE_ID,
                              "-O", str(zip_path)])
        if rc != 0 or not zip_path.is_file():
            say("    FAILED. Google Drive sometimes rate-limits large files; retry later,")
            say("    or download it by hand from the CrowdPose repo and place it at:")
            say(f"      {zip_path}")
            return False
    say("    extracting 20,000 images ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(CP_DIR)
    n = len(list(img_dir.glob("*.jpg")))
    say(f"    -> {img_dir}  ({n} images)")
    return n > 0


def stage_jrdb_labels() -> bool:
    say("[5/7] JRDB-Pose labels  (read out of a remote zip, no full download)")
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import fetch_jrdb_labels
        fetch_jrdb_labels.main()
    except Exception as exc:
        say(f"    FAILED: {type(exc).__name__}: {exc}")
        return False
    return (JR_DIR / "labels" / "labels_2d_pose_coco").is_dir()


def stage_jrdb_images() -> bool:
    say("[6/7] JRDB-Pose frames  (pulled selectively from a 21 GB remote zip)")
    say("      This reads the archive index first, so it is quiet for a minute or two.")
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import fetch_jrdb_images
        fetch_jrdb_images.main()
    except Exception as exc:
        say(f"    FAILED: {type(exc).__name__}: {exc}")
        return False
    return len(list((JR_DIR / "images").rglob("*.jpg"))) > 0


RUNNERS = {
    "models": stage_models,
    "openpose": stage_openpose,
    "jrdb-sequences": stage_jrdb_sequences,
    "crowdpose-anns": stage_crowdpose_anns,
    "crowdpose-images": stage_crowdpose_images,
    "jrdb-labels": stage_jrdb_labels,
    "jrdb-images": stage_jrdb_images,
}


def summary():
    say("\n" + "=" * 70)
    say("SETUP SUMMARY")
    say("=" * 70)
    checks = [
        ("YOLO weights",      MODEL_DIR / "yolov8n-pose.pt", "file"),
        ("YOLO26 weights",    MODEL_DIR / "yolo26n-pose.pt", "file"),
        ("MediaPipe model",   MODEL_DIR / "pose_landmarker_full.task", "file"),
        ("CrowdPose anns",    CP_DIR / "crowdpose_annotations" / "mmpose_crowdpose_val.json", "file"),
        ("CrowdPose images",  CP_DIR / "images", "jpgs"),
        ("JRDB labels",       JR_DIR / "labels" / "labels_2d_pose_coco", "jsons"),
        ("JRDB images",       JR_DIR / "images", "jpgs-rec"),
    ]
    all_ok = True
    for label, path, kind in checks:
        if kind == "file":
            ok, detail = path.is_file(), (f"{path.stat().st_size/1e6:.1f} MB" if path.is_file() else "not found")
        elif kind == "jpgs":
            n = len(list(path.glob("*.jpg"))) if path.is_dir() else 0
            ok, detail = n > 0, f"{n} images"
        elif kind == "jpgs-rec":
            n = len(list(path.rglob("*.jpg"))) if path.is_dir() else 0
            ok, detail = n > 0, f"{n} images"
        else:
            n = len(list(path.glob("*.json"))) if path.is_dir() else 0
            ok, detail = n > 0, f"{n} label files"
        all_ok &= ok
        say(f"  [{'OK ' if ok else 'MISSING'}] {label:<18} {detail}")
    say("=" * 70)
    if all_ok:
        say("Everything is in place. Next:\n"
            "    jupyter lab pose_benchmarking.ipynb      then Run All")
    else:
        say("Some inputs are missing. The notebook still runs end to end -- it reports\n"
            "what is absent and skips those parts. Re-run this script to retry.")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description="Fetch models and datasets for the pose benchmark.")
    ap.add_argument("--only", nargs="+", choices=STAGES, help="run only these stages")
    ap.add_argument("--skip", nargs="+", choices=STAGES, default=[], help="skip these stages")
    args = ap.parse_args()

    todo = [s for s in (args.only or STAGES) if s not in args.skip]
    say(f"Project root: {ROOT}")
    say(f"Stages: {', '.join(todo)}\n")
    for s in todo:
        RUNNERS[s]()
        say()
    return 0 if summary() else 1


if __name__ == "__main__":
    sys.exit(main())
