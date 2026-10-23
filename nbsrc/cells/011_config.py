# =============================================================================
#  CONFIGURATION  --  the only cell you need to edit to re-point the notebook
# =============================================================================
from pathlib import Path

# ---- 1. Where the project lives -------------------------------------------------
# By default the notebook uses the folder it is sitting in, so the project can be
# copied anywhere and run without editing anything. To point it at data held
# somewhere else, replace this line with an explicit path, e.g.
#     PROJECT_ROOT = Path(r"D:\pose_project")
PROJECT_ROOT = Path.cwd()

DATA_ROOT    = PROJECT_ROOT / "data"
MODEL_DIR    = PROJECT_ROOT / "models"      # local model weights (kept offline)
RESULTS_DIR  = PROJECT_ROOT / "results"     # CSV tables + saved figures

# ---- 2. CrowdPose ---------------------------------------------------------------
CROWDPOSE_IMAGE_DIR = DATA_ROOT / "crowdpose" / "images"
CROWDPOSE_ANN_FILE  = DATA_ROOT / "crowdpose" / "crowdpose_annotations" / "mmpose_crowdpose_val.json"

# ---- 3. JRDB-Pose ---------------------------------------------------------------
# Individual camera frames (not the stitched panorama), with one COCO-style
# label JSON per scene+camera.
JRDB_IMAGE_DIR = DATA_ROOT / "jrdb" / "images"
JRDB_LABEL_DIR = DATA_ROOT / "jrdb" / "labels" / "labels_2d_pose_coco"
# Runs of CONSECUTIVE annotated frames, kept apart from the scattered benchmark subset
# because only they can support the temporal-stability analysis in Section 7.
# Fetch with:  python tools/fetch_jrdb_sequences.py
JRDB_SEQUENCE_DIR = DATA_ROOT / "jrdb" / "sequences"

# ---- 4. Model weights -----------------------------------------------------------
YOLO_WEIGHTS         = MODEL_DIR / "yolov8n-pose.pt"          # YOLOv8n-pose (CPU friendly)
YOLO26_WEIGHTS       = MODEL_DIR / "yolo26n-pose.pt"          # latest YOLO generation (Sep 2025)
MEDIAPIPE_TASK_MODEL = MODEL_DIR / "pose_landmarker_full.task"  # used by the Tasks API path
OPENPOSE_ROOT        = Path(r"C:\openpose")                   # official C++ build, if you have one
# Converted original CMU COCO-18 weights, used when no C++ build is present (see Section 3.3).
# Fetch with:  python tools/setup_data.py --openpose
OPENPOSE_WEIGHTS     = MODEL_DIR / "openpose_body_coco18.pth"

# ---- 5. Benchmark subset sizes (kept small so the notebook runs on CPU) ----------
N_CROWDPOSE_IMAGES = 600    # images sampled from CrowdPose val
N_JRDB_IMAGES      = 600    # images sampled from the fetched JRDB frames
N_VIS_SAMPLES      = 4       # sample images shown per dataset in Sections 2 and 3
# Set both of the above to 150 / 220 to reproduce the smaller subset used in the interim
# report. OpenPose dominates the runtime: on CPU it costs roughly 2 s per image, so the
# full three-model run over this subset takes about an hour.

# ---- 6. Evaluation protocol constants -------------------------------------------
RANDOM_SEED        = 42      # every sampling step uses this
KPT_CONF_THRESHOLD = 0.30    # a predicted keypoint below this confidence counts as "missing"
DET_CONF_THRESHOLD = 0.25    # minimum person-detection score
MATCH_IOU_THRESHOLD = 0.30   # bbox IoU needed to accept a prediction<->GT person match
PCK_ALPHA          = 0.50    # PCK@0.5  (the headline accuracy metric)
PCK_ALPHA_STRICT   = 0.20    # PCK@0.2  (a stricter secondary reading)
MIN_GT_KEYPOINTS   = 4       # ignore GT persons with fewer labelled common keypoints

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Configuration loaded.")
print(f"  project root        : {PROJECT_ROOT}")
print(f"  results ->          : {RESULTS_DIR}")
print()

# Check every input up front and say plainly what is missing, so a wrong path is
# caught here rather than surfacing as a confusing error ten cells later.
_checks = [
    ("CrowdPose images",   CROWDPOSE_IMAGE_DIR, "dir"),
    ("CrowdPose anns",     CROWDPOSE_ANN_FILE,  "file"),
    ("JRDB images",        JRDB_IMAGE_DIR,      "dir"),
    ("JRDB labels",        JRDB_LABEL_DIR,      "dir"),
    ("YOLO weights",       YOLO_WEIGHTS,        "file"),
    ("MediaPipe model",    MEDIAPIPE_TASK_MODEL, "file"),
    ("OpenPose weights",   OPENPOSE_WEIGHTS,    "file"),
    ("JRDB sequences",     JRDB_SEQUENCE_DIR,   "dir"),
]
_missing = []
for _label, _path, _kind in _checks:
    _ok = _path.is_dir() if _kind == "dir" else _path.is_file()
    print(f"  [{'OK ' if _ok else 'MISSING'}] {_label:<18} {_path}")
    if not _ok:
        _missing.append(_label)

if _missing:
    print(f"\n  !! {len(_missing)} input(s) not found: {', '.join(_missing)}")
    print( "     Fetch everything with:   python tools/setup_data.py")
    print( "     Or set PROJECT_ROOT above to the folder that holds data/ and models/.")
    print( "     The notebook still runs: whatever is missing is reported and skipped.")
else:
    print("\n  All inputs present.")
