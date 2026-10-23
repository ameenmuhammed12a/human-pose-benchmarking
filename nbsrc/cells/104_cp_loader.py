class CrowdPoseDataset(PoseDataset):
    """Loader for the COCO-style CrowdPose annotation file (14 keypoints per person)."""
    name = "CrowdPose"
    annotates_invisible = False         # CrowdPose v == 0 means "not annotated", (x, y) == (0, 0)

    def __init__(self, ann_file: Path, image_dir: Path):
        super().__init__()
        self.ann_file, self.image_dir = Path(ann_file), Path(image_dir)
        self.keypoint_names, self.skeleton = (), ()

    def load(self, limit: Optional[int] = None, seed: int = RANDOM_SEED) -> List[ImageSample]:
        if not self.ann_file.is_file():
            print(f"[CrowdPose] annotation file missing: {self.ann_file} -- returning no samples.")
            return []
        with open(self.ann_file, encoding="utf-8") as fh:
            data = json.load(fh)

        # Keypoint order and limb links are read from the file itself, not assumed.
        cat = (data.get("categories") or [{}])[0]
        self.keypoint_names = tuple(cat.get("keypoints", []))
        self.skeleton = tuple(tuple(link) for link in cat.get("skeleton", []))
        n_kpt = len(self.keypoint_names)

        by_image: Dict[int, List[dict]] = {}
        for ann in data.get("annotations", []):
            if ann.get("iscrowd", 0):                    # crowd regions carry no usable pose
                continue
            if not ann.get("keypoints"):
                continue
            by_image.setdefault(ann["image_id"], []).append(ann)

        # Keep only images that have annotations AND are actually present on disk.
        candidates = [im for im in data.get("images", [])
                      if by_image.get(im["id"]) and (self.image_dir / im["file_name"]).is_file()]
        missing = sum(1 for im in data.get("images", [])
                      if by_image.get(im["id"]) and not (self.image_dir / im["file_name"]).is_file())
        if missing:
            print(f"[CrowdPose] {missing} annotated images are not on disk and were skipped.")

        # Seeded, reproducible subset.
        rng = random.Random(seed)
        candidates.sort(key=lambda im: im["id"])
        if limit is not None and limit < len(candidates):
            candidates = rng.sample(candidates, limit)
            candidates.sort(key=lambda im: im["id"])

        samples = []
        for im in candidates:
            persons = []
            for ann in by_image[im["id"]]:
                kp = np.asarray(ann["keypoints"], dtype=np.float32)
                if kp.size != n_kpt * 3:                 # malformed row: skip it, do not crash
                    continue
                kp = kp.reshape(n_kpt, 3)
                bbox = ann.get("bbox") or bbox_from_keypoints(kp, im["width"], im["height"])
                persons.append(GTPerson(keypoints=kp,
                                        bbox=tuple(float(v) for v in bbox),
                                        keypoint_names=self.keypoint_names,
                                        track_id=None, ann_id=ann.get("id")))
            if not persons:
                continue
            samples.append(ImageSample(
                image_id=im["id"],
                image_path=self.image_dir / im["file_name"],
                width=int(im["width"]), height=int(im["height"]),
                persons=persons,
                meta={"crowdIndex": im.get("crowdIndex"), "file_name": im["file_name"]},
            ))
        self.samples = samples
        return samples
