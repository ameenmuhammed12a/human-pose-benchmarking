class JRDBPoseDataset(PoseDataset):
    """Loader for JRDB-Pose per-scene/per-camera COCO-style label files (17 keypoints).

    Every keypoint carries a visibility flag in {0, 1, 2}:
        0 = invisible   (out of frame, or especially difficult to annotate)
        1 = occluded    (hidden, but its location is reasonably inferable)
        2 = visible     (fully visible and in view of the camera)
    That flag is what makes the direct occlusion analysis in Section 6 possible.
    """
    name = "JRDB-Pose"
    annotates_invisible = True          # JRDB labels a location even for v == 0
    VISIBILITY_LABELS = {0: "invisible", 1: "occluded", 2: "visible"}

    def __init__(self, label_dir: Path, image_dir: Path):
        super().__init__()
        self.label_dir, self.image_dir = Path(label_dir), Path(image_dir)
        self.keypoint_names, self.skeleton = (), ()

    def load(self, limit: Optional[int] = None, seed: int = RANDOM_SEED) -> List[ImageSample]:
        if not self.label_dir.is_dir():
            print(f"[JRDB-Pose] label directory missing: {self.label_dir} -- returning no samples.")
            return []
        label_files = sorted(self.label_dir.glob("*.json"))
        if not label_files:
            print(f"[JRDB-Pose] no .json label files under {self.label_dir} -- returning no samples.")
            return []

        all_samples, uid = [], 0
        for lf in label_files:
            try:
                with open(lf, encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as exc:
                print(f"[JRDB-Pose] skipping unreadable {lf.name}: {type(exc).__name__}: {exc}")
                continue

            cat = (data.get("categories") or [{}])[0]
            if not self.keypoint_names:                   # schema is identical across scene files
                self.keypoint_names = tuple(cat.get("keypoints", []))
                self.skeleton = tuple(tuple(l) for l in cat.get("skeleton", []))
            n_kpt = len(self.keypoint_names)

            by_image: Dict[int, List[dict]] = {}
            for ann in data.get("annotations", []):
                if ann.get("is_crowd", ann.get("iscrowd", 0)):
                    continue
                if ann.get("keypoints"):
                    by_image.setdefault(ann["image_id"], []).append(ann)

            for im in data.get("images", []):
                anns = by_image.get(im["id"])
                if not anns:
                    continue
                img_path = self.image_dir / Path(im["file_name"])
                if not img_path.is_file():                # only the fetched subset is on disk
                    continue
                w, h = int(im["width"]), int(im["height"])
                persons = []
                for ann in anns:
                    kp = np.asarray(ann["keypoints"], dtype=np.float32)
                    if kp.size != n_kpt * 3:
                        continue
                    kp = kp.reshape(n_kpt, 3)
                    # Per-camera JRDB files carry no bbox, so it is derived from the keypoints.
                    bbox = ann.get("bbox") or bbox_from_keypoints(kp, w, h)
                    persons.append(GTPerson(
                        keypoints=kp, bbox=tuple(float(v) for v in bbox),
                        keypoint_names=self.keypoint_names,
                        track_id=ann.get("track_id"), ann_id=ann.get("id")))
                if not persons:
                    continue
                uid += 1
                parts = Path(im["file_name"]).parts        # "image_2/<scene>/000123.jpg"
                all_samples.append(ImageSample(
                    image_id=uid, image_path=img_path, width=w, height=h, persons=persons,
                    meta={"scene": parts[1] if len(parts) > 1 else lf.stem,
                          "camera": parts[0] if parts else "?",
                          "frame": Path(im["file_name"]).stem,
                          "label_file": lf.name,
                          "file_name": im["file_name"]},
                ))

        # Seeded subset, spread round-robin over scenes so no single scene dominates.
        rng = random.Random(seed)
        if limit is not None and limit < len(all_samples):
            by_scene: Dict[str, List[ImageSample]] = {}
            for s in all_samples:
                by_scene.setdefault(s.meta["scene"], []).append(s)
            scenes, picked = sorted(by_scene), []
            i = 0
            while len(picked) < limit and any(by_scene[s] for s in scenes):
                pool = by_scene[scenes[i % len(scenes)]]
                if pool:
                    picked.append(pool.pop(rng.randrange(len(pool))))
                i += 1
            all_samples = picked
        all_samples.sort(key=lambda s: (s.meta["scene"], s.meta["frame"]))
        self.samples = all_samples
        return all_samples
