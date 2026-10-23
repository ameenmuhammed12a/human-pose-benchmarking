class YOLOPoseWrapper(PoseModel):
    """Ultralytics YOLOv8-pose. Natively multi-person, outputs the COCO-17 skeleton."""
    name = "YOLOv8-pose"
    native_keypoints = COCO17_KEYPOINTS
    native_skeleton = COCO17_SKELETON

    def __init__(self, weights: Path, conf: float = DET_CONF_THRESHOLD, imgsz: int = 640):
        super().__init__()
        self.weights, self.conf, self.imgsz, self.model = Path(weights), conf, imgsz, None
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self.unavailable_reason = f"ultralytics not installed ({exc})"
            print(f"[{self.name}] UNAVAILABLE: {self.unavailable_reason}")
            print("    install with:  pip install ultralytics")
            return
        if not self.weights.is_file():
            self.unavailable_reason = f"weights not found at {self.weights}"
            print(f"[{self.name}] UNAVAILABLE: {self.unavailable_reason}")
            return
        try:
            self.model = YOLO(str(self.weights))
            self.model.to("cpu")                       # CPU-only, as required by the brief
            self.available = True
            print(f"[{self.name}] ready — weights {self.weights.name}, device CPU, imgsz {imgsz}")
        except Exception as exc:
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"
            print(f"[{self.name}] UNAVAILABLE: {self.unavailable_reason}")

    def predict(self, image_bgr, person_boxes=None) -> PredictionResult:
        if not self.available:
            return PredictionResult([], 0.0, self.name, 0, self.unavailable_reason)
        try:
            t0 = time.perf_counter()
            res = self.model.predict(image_bgr, device="cpu", imgsz=self.imgsz,
                                     conf=self.conf, verbose=False)[0]
            dt = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            return PredictionResult([], 0.0, self.name, 1, f"{type(exc).__name__}: {exc}")

        persons = []
        if res.keypoints is not None and res.keypoints.data is not None and len(res.keypoints.data):
            kdata = res.keypoints.data.cpu().numpy()          # (P, 17, 3) = x, y, conf
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else None
            scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else None
            for i in range(kdata.shape[0]):
                kp = {nm: (float(kdata[i, j, 0]), float(kdata[i, j, 1]), float(kdata[i, j, 2]))
                      for j, nm in enumerate(self.native_keypoints)}
                if boxes is not None and i < len(boxes):
                    x1, y1, x2, y2 = boxes[i]
                    bbox = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
                else:                                          # fall back to the keypoint hull
                    arr = kdata[i][kdata[i, :, 2] > 0]
                    bbox = ((float(arr[:, 0].min()), float(arr[:, 1].min()),
                             float(arr[:, 0].ptp()), float(arr[:, 1].ptp()))
                            if len(arr) else (0.0, 0.0, 0.0, 0.0))
                persons.append(PredictedPerson(
                    keypoints=kp, bbox=bbox,
                    score=float(scores[i]) if scores is not None and i < len(scores) else 1.0))
        return PredictionResult(persons, dt, self.name, 1)
