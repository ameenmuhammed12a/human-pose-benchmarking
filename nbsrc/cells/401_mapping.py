# =============================================================================
#  The common 12-joint set, and every skeleton -> common mapping
# =============================================================================
COMMON_KEYPOINTS = [
    "left_shoulder",  "right_shoulder",
    "left_elbow",     "right_elbow",
    "left_wrist",     "right_wrist",
    "left_hip",       "right_hip",
    "left_knee",      "right_knee",
    "left_ankle",     "right_ankle",
]
N_COMMON = len(COMMON_KEYPOINTS)
COMMON_INDEX = {nm: i for i, nm in enumerate(COMMON_KEYPOINTS)}

# Limb links on the COMMON indices, used when drawing the mapped skeletons.
COMMON_SKELETON = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5),
                   (0, 6), (1, 7), (6, 7), (6, 8), (8, 10), (7, 9), (9, 11)]

# --- Dataset skeletons -> common ---------------------------------------------------
# CrowdPose-14 order: L/R shoulder, L/R elbow, L/R wrist, L/R hip, L/R knee, L/R ankle,
#                     head, neck  -> its first 12 entries already are the common set.
CROWDPOSE14_TO_COMMON = {
    "left_shoulder": 0,  "right_shoulder": 1,
    "left_elbow":    2,  "right_elbow":    3,
    "left_wrist":    4,  "right_wrist":    5,
    "left_hip":      6,  "right_hip":      7,
    "left_knee":     8,  "right_knee":     9,
    "left_ankle":   10,  "right_ankle":   11,
}

# JRDB-17 order (read from the label files): head, right eye, left eye, right shoulder,
# center shoulder, left shoulder, right elbow, left elbow, center hip, right wrist,
# right hip, left hip, left wrist, right knee, left knee, right foot, left foot.
# NOTE: "foot" is JRDB's closest joint to an ankle (see the markdown above).
JRDB17_TO_COMMON = {
    "left_shoulder":  5,  "right_shoulder":  3,
    "left_elbow":     7,  "right_elbow":     6,
    "left_wrist":    12,  "right_wrist":     9,
    "left_hip":      11,  "right_hip":      10,
    "left_knee":     14,  "right_knee":     13,
    "left_ankle":    16,  "right_ankle":    15,   # JRDB "left foot" / "right foot"
}

# --- Model skeletons -> common ------------------------------------------------------
COCO17_TO_COMMON = {
    "left_shoulder":  5,  "right_shoulder":  6,
    "left_elbow":     7,  "right_elbow":     8,
    "left_wrist":     9,  "right_wrist":    10,
    "left_hip":      11,  "right_hip":      12,
    "left_knee":     13,  "right_knee":     14,
    "left_ankle":    15,  "right_ankle":    16,
}

MEDIAPIPE33_TO_COMMON = {
    "left_shoulder": 11,  "right_shoulder": 12,
    "left_elbow":    13,  "right_elbow":    14,
    "left_wrist":    15,  "right_wrist":    16,
    "left_hip":      23,  "right_hip":      24,
    "left_knee":     25,  "right_knee":     26,
    "left_ankle":    27,  "right_ankle":    28,
}

COCO18_TO_COMMON = {
    "left_shoulder":  5,  "right_shoulder":  2,
    "left_elbow":     6,  "right_elbow":     3,
    "left_wrist":     7,  "right_wrist":     4,
    "left_hip":      11,  "right_hip":       8,
    "left_knee":     12,  "right_knee":      9,
    "left_ankle":    13,  "right_ankle":    10,
}

BODY25_TO_COMMON = {
    "left_shoulder":  5,  "right_shoulder":  2,
    "left_elbow":     6,  "right_elbow":     3,
    "left_wrist":     7,  "right_wrist":     4,
    "left_hip":      12,  "right_hip":       9,
    "left_knee":     13,  "right_knee":     10,
    "left_ankle":    14,  "right_ankle":    11,
}

# OpenPose emits one of two skeletons depending on which backend loaded (Section 3.3), so its
# entry is selected from the wrapper rather than hard-coded. Both cover all 12 common joints,
# which is what keeps the comparison valid either way.
OPENPOSE_SKELETONS = {
    "BODY_25": ("OpenPose BODY_25", BODY25_TO_COMMON, BODY25_KEYPOINTS),
    "COCO_18": ("OpenPose COCO-18", COCO18_TO_COMMON, COCO18_KEYPOINTS),
}

# Which of the two OpenPose skeletons is in play is read off the wrapper. Tooling that runs
# only the data and mapping cells (tools/build_review_folder.py) never instantiates it, so
# this falls back to COCO-18 — the backend the committed results use.
_OPENPOSE_SKELETON = getattr(globals().get("op_model"), "skeleton_id", None) or "COCO_18"

# Which mapping belongs to which model / dataset.
MODEL_MAPPINGS = {
    "YOLOv8-pose":   ("COCO-17",      COCO17_TO_COMMON,      COCO17_KEYPOINTS),
    "MediaPipe-Pose": ("MediaPipe-33", MEDIAPIPE33_TO_COMMON, MEDIAPIPE33_KEYPOINTS),
    "OpenPose":      OPENPOSE_SKELETONS[_OPENPOSE_SKELETON],
}
MODEL_MAPPINGS["YOLO26-pose"] = MODEL_MAPPINGS["YOLOv8-pose"]   # same COCO-17 skeleton
DATASET_MAPPINGS = {
    "CrowdPose":  ("CrowdPose-14", CROWDPOSE14_TO_COMMON),
    "JRDB-Pose":  ("JRDB-17",      JRDB17_TO_COMMON),
}


def validate_mappings() -> None:
    """Fail loudly if any mapping is incomplete or points outside its skeleton."""
    checks = [
        ("CrowdPose-14", CROWDPOSE14_TO_COMMON, tuple(crowdpose.keypoint_names) or ("",) * 14),
        ("JRDB-17",      JRDB17_TO_COMMON,      tuple(jrdb.keypoint_names) or ("",) * 17),
        ("COCO-17",      COCO17_TO_COMMON,      COCO17_KEYPOINTS),
        ("MediaPipe-33", MEDIAPIPE33_TO_COMMON, MEDIAPIPE33_KEYPOINTS),
        ("BODY_25",      BODY25_TO_COMMON,      BODY25_KEYPOINTS),
        ("COCO-18",      COCO18_TO_COMMON,      COCO18_KEYPOINTS),
    ]
    problems = []
    for label, mapping, names in checks:
        missing = set(COMMON_KEYPOINTS) - set(mapping)
        if missing:
            problems.append(f"{label}: missing common joints {sorted(missing)}")
        for cn, idx in mapping.items():
            if not (0 <= idx < len(names)):
                problems.append(f"{label}: {cn} -> index {idx} is out of range "
                                f"(skeleton has {len(names)} joints)")
        if len(set(mapping.values())) != len(mapping):
            problems.append(f"{label}: two common joints map to the same native index")
    if problems:
        raise ValueError("Keypoint mapping is invalid:\n  " + "\n  ".join(problems))
    print(f"All {len(checks)} mappings validated: each covers all {N_COMMON} common joints, "
          f"with in-range, one-to-one indices.")
    _backend = getattr(globals().get("op_model"), "backend", None)
    print(f"OpenPose is mapped from its {MODEL_MAPPINGS['OpenPose'][0]} skeleton "
          f"(backend: {_backend or 'not instantiated'}).")


validate_mappings()
