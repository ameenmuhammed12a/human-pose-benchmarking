class MediaPipeWrapper(PoseModel):
    """MediaPipe Pose (BlazePose, 33 landmarks).

    MediaPipe Pose is a SINGLE-person estimator, so this wrapper runs it once per supplied
    person box (a crop, padded for context) and maps the normalised landmark coordinates back
    into full-image pixel coordinates.

    Two backends are supported and selected automatically:
      * "solutions" - the legacy ``mp.solutions.pose`` API (mediapipe 0.10.x <= 0.10.9)
      * "tasks"     - the modern ``mediapipe.tasks`` PoseLandmarker API
    Both emit the same 33-landmark skeleton, so everything downstream is identical.
    """
    name = "MediaPipe-Pose"
    native_keypoints = MEDIAPIPE33_KEYPOINTS
    native_skeleton = MEDIAPIPE33_SKELETON

    def __init__(self, task_model: Optional[Path] = None,
                 model_complexity: int = 1, crop_pad: float = 0.25,
                 min_detection_confidence: float = 0.2):
        super().__init__()
        self.backend, self._impl, self.crop_pad = None, None, crop_pad
        try:
            import mediapipe as mp
        except ImportError as exc:
            self.unavailable_reason = f"mediapipe not installed ({exc})"
            print(f"[{self.name}] UNAVAILABLE: {self.unavailable_reason}")
            print("    install with:  pip install mediapipe")
            return
        self._mp = mp
        mp_version = _version("mediapipe")

        # --- preferred: legacy solutions API, exactly as the brief specifies ---
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            try:
                self._impl = mp.solutions.pose.Pose(
                    static_image_mode=True, model_complexity=model_complexity,
                    enable_segmentation=False,
                    min_detection_confidence=min_detection_confidence)
                self.backend, self.available = "solutions", True
                print(f"[{self.name}] ready — legacy mp.solutions.pose API "
                      f"(mediapipe {mp_version}), 33 landmarks, CPU")
                return
            except Exception as exc:
                print(f"[{self.name}] legacy solutions API present but failed to start "
                      f"({type(exc).__name__}: {exc}); trying the Tasks API.")

        # --- fallback: modern Tasks API (this is what runs on newer mediapipe builds) ---
        try:
            from mediapipe.tasks.python import vision, BaseOptions
        except Exception as exc:
            self.unavailable_reason = (f"mediapipe {mp_version} exposes neither mp.solutions.pose "
                                       f"nor mediapipe.tasks ({exc})")
            print(f"[{self.name}] UNAVAILABLE: {self.unavailable_reason}")
            return
        task_model = Path(task_model) if task_model else None
        if not task_model or not task_model.is_file():
            self.unavailable_reason = (f"mediapipe {mp_version} needs the Tasks API, but the model "
                                       f"file was not found at {task_model}")
            print(f"[{self.name}] UNAVAILABLE: {self.unavailable_reason}")
            print("    download pose_landmarker_full.task from storage.googleapis.com/mediapipe-models")
            return
        try:
            opts = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(task_model)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence)
            self._impl = vision.PoseLandmarker.create_from_options(opts)
            self.backend, self.available = "tasks", True
            print(f"[{self.name}] ready — mediapipe.tasks PoseLandmarker "
                  f"(mediapipe {mp_version}), model {task_model.name}, 33 landmarks, CPU")
            print(f"    note: mediapipe=={mp_version} publishes no mp.solutions.pose on this "
                  f"Python build, so the Tasks API is used. Same 33-landmark skeleton.")
        except Exception as exc:
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"
            print(f"[{self.name}] UNAVAILABLE: {self.unavailable_reason}")

    # -- internals ------------------------------------------------------------------
    def _landmarks_on_crop(self, crop_bgr: np.ndarray):
        """Return a list of 33 (x_norm, y_norm, confidence) tuples, or None if no pose found."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        if self.backend == "solutions":
            res = self._impl.process(rgb)
            if not res.pose_landmarks:
                return None
            return [(lm.x, lm.y, float(getattr(lm, "visibility", 1.0)))
                    for lm in res.pose_landmarks.landmark]
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._impl.detect(mp_img)
        if not res.pose_landmarks:
            return None
        return [(lm.x, lm.y, float(getattr(lm, "visibility", 1.0)))
                for lm in res.pose_landmarks[0]]

    def predict(self, image_bgr, person_boxes=None) -> PredictionResult:
        if not self.available:
            return PredictionResult([], 0.0, self.name, 0, self.unavailable_reason)
        H, W = image_bgr.shape[:2]
        # With no boxes supplied, fall back to treating the whole frame as one person.
        boxes = list(person_boxes) if person_boxes else [[0, 0, W, H]]

        persons, total_ms, calls = [], 0.0, 0
        for bx in boxes:
            x, y, w, h = [float(v) for v in bx]
            if w <= 1 or h <= 1:
                continue
            # Pad the crop: BlazePose needs some context around the person.
            px, py = w * self.crop_pad, h * self.crop_pad
            x0, y0 = int(max(0, x - px)), int(max(0, y - py))
            x1, y1 = int(min(W, x + w + px)), int(min(H, y + h + py))
            if x1 - x0 < 16 or y1 - y0 < 16:              # too small to be worth running
                continue
            crop = image_bgr[y0:y1, x0:x1]
            calls += 1
            try:
                t0 = time.perf_counter()
                lms = self._landmarks_on_crop(crop)
                total_ms += (time.perf_counter() - t0) * 1000.0
            except Exception as exc:
                print(f"[{self.name}] crop failed ({type(exc).__name__}: {exc}) — skipped")
                continue
            if lms is None:                                # no pose in this crop
                continue
            ch, cw = crop.shape[:2]
            kp = {}
            for nm, (nx, ny, conf) in zip(self.native_keypoints, lms):
                kp[nm] = (x0 + nx * cw, y0 + ny * ch, conf)   # map back to full-image pixels
            persons.append(PredictedPerson(keypoints=kp, bbox=(x, y, w, h), score=1.0))
        return PredictionResult(persons, total_ms, self.name, max(1, calls))

    def close(self):
        """Release the underlying graph (avoids noisy interpreter-shutdown warnings)."""
        try:
            if self._impl is not None:
                self._impl.close()
        except Exception:
            pass
