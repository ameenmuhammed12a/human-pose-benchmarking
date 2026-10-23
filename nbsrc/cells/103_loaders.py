# =============================================================================
#  Unified ground-truth structures
# =============================================================================

@dataclass
class GTPerson:
    """One annotated person, in a form shared by both datasets."""
    keypoints: np.ndarray                       # (K, 3) float array of (x, y, visibility)
    bbox: Tuple[float, float, float, float]     # (x, y, w, h) in pixels
    keypoint_names: Sequence[str]               # names for the K rows, in order
    track_id: Optional[int] = None              # JRDB only; None for CrowdPose
    ann_id: Optional[int] = None

    @property
    def num_labelled(self) -> int:
        """Keypoints with a non-zero visibility flag."""
        return int((self.keypoints[:, 2] > 0).sum())


@dataclass
class ImageSample:
    """One image plus every person annotated in it."""
    image_id: int
    image_path: Path
    width: int
    height: int
    persons: List[GTPerson] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        return self.image_path.is_file()

    def load_bgr(self) -> Optional[np.ndarray]:
        """Read the image with OpenCV; returns None instead of raising if unreadable."""
        if not self.exists:
            return None
        img = cv2.imread(str(self.image_path))
        return img if img is not None and img.size else None


def bbox_from_keypoints(kpts: np.ndarray, img_w: int, img_h: int,
                        margin: float = 0.12) -> Tuple[float, float, float, float]:
    """Derive an (x, y, w, h) box from labelled keypoints, with a margin, clipped to the image.

    Used for JRDB-Pose, whose per-camera annotations carry keypoints but no bounding box.
    """
    valid = kpts[kpts[:, 2] > 0]
    if len(valid) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    x0, y0 = float(valid[:, 0].min()), float(valid[:, 1].min())
    x1, y1 = float(valid[:, 0].max()), float(valid[:, 1].max())
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = max(0.0, x0 - mx), max(0.0, y0 - my)
    x1, y1 = min(float(img_w), x1 + mx), min(float(img_h), y1 + my)
    return (x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0))


class PoseDataset(ABC):
    """Common interface for both benchmark datasets."""
    name: str = "dataset"
    keypoint_names: Sequence[str] = ()
    skeleton: Sequence[Tuple[int, int]] = ()
    # True when v == 0 still carries a real annotated coordinate (JRDB-Pose "invisible"),
    # False when v == 0 simply means "not annotated" and (x, y) is (0, 0) (CrowdPose).
    # This distinction decides which keypoints are evaluable, so it must not be guessed.
    annotates_invisible: bool = False

    def __init__(self):
        self.samples: List[ImageSample] = []

    @abstractmethod
    def load(self, limit: Optional[int] = None, seed: int = RANDOM_SEED) -> List[ImageSample]:
        ...

    def statistics(self) -> dict:
        """Counts used for the Task 1 statistics printout."""
        n_persons = sum(len(s.persons) for s in self.samples)
        per_img = [len(s.persons) for s in self.samples]
        labelled = [p.num_labelled for s in self.samples for p in s.persons]
        vis_counts = {0: 0, 1: 0, 2: 0}
        for s in self.samples:
            for p in s.persons:
                for v in p.keypoints[:, 2].astype(int):
                    vis_counts[int(v)] = vis_counts.get(int(v), 0) + 1
        return {
            "dataset": self.name,
            "images": len(self.samples),
            "person_instances": n_persons,
            "keypoints_per_person(defined)": len(self.keypoint_names),
            "persons_per_image_mean": float(np.mean(per_img)) if per_img else 0.0,
            "persons_per_image_max": int(np.max(per_img)) if per_img else 0,
            "labelled_kpts_per_person_mean": float(np.mean(labelled)) if labelled else 0.0,
            "total_keypoint_slots": n_persons * len(self.keypoint_names),
            "visibility_counts": vis_counts,
        }
