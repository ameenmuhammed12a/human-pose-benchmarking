# =============================================================================
#  The common model interface
# =============================================================================

@dataclass
class PredictedPerson:
    """One detected person, in a model-independent form."""
    keypoints: Dict[str, Tuple[float, float, float]]   # native name -> (x, y, confidence)
    bbox: Tuple[float, float, float, float]            # (x, y, w, h) in full-image pixels
    score: float = 1.0                                 # person-level detection confidence

    def as_array(self, order: Sequence[str]) -> np.ndarray:
        """Return an (N, 3) array of (x, y, conf) for the requested keypoint names."""
        out = np.zeros((len(order), 3), dtype=np.float32)
        for i, nm in enumerate(order):
            if nm in self.keypoints:
                out[i] = self.keypoints[nm]
        return out


@dataclass
class PredictionResult:
    """Everything one model produced for one image, plus how long it took."""
    persons: List[PredictedPerson]
    inference_ms: float                                # wall clock, model call only
    model_name: str
    n_model_calls: int = 1                             # >1 for crop-based models
    error: Optional[str] = None                        # set instead of raising


class PoseModel(ABC):
    """Abstract base class every pose estimator wrapper implements."""
    name: str = "model"
    native_keypoints: Sequence[str] = ()

    def __init__(self):
        self.available: bool = False
        self.unavailable_reason: str = ""

    @abstractmethod
    def predict(self, image_bgr: np.ndarray,
                person_boxes: Optional[Sequence[Sequence[float]]] = None) -> PredictionResult:
        """Estimate poses in one BGR image.

        person_boxes is an optional list of (x, y, w, h) boxes, used only by
        single-person models that need to be pointed at each person in turn.
        """
        ...

    def warmup(self, image_bgr: np.ndarray) -> None:
        """Run once and discard, so lazy graph/weight loading does not pollute the timings."""
        if self.available:
            try:
                self.predict(image_bgr, person_boxes=[[0, 0, image_bgr.shape[1], image_bgr.shape[0]]])
            except Exception as exc:
                print(f"[{self.name}] warm-up failed: {type(exc).__name__}: {exc}")

    def __repr__(self) -> str:
        state = "available" if self.available else f"UNAVAILABLE ({self.unavailable_reason})"
        return f"<{self.__class__.__name__} {self.name}: {state}>"


# ---- Standard keypoint name lists for the three native skeletons -------------------
COCO17_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

MEDIAPIPE33_KEYPOINTS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]

# OpenPose's COCO release (pose_deploy_linevec) predicts 18 joints; BODY_25 is the newer
# head of the same family. Which of the two is in play depends on the OpenPose backend.
COCO18_KEYPOINTS = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
    "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar",
]

BODY25_KEYPOINTS = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
    "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye",
    "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel",
]

# Limb links, expressed on the native index of each skeleton, used only for drawing.
COCO17_SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                   (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
MEDIAPIPE33_SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23),
                        (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]
COCO18_SKELETON = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
                   (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (0, 15),
                   (14, 16), (15, 17)]
BODY25_SKELETON = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
                   (9, 10), (10, 11), (8, 12), (12, 13), (13, 14), (0, 1)]

print("PoseModel interface defined.")
