class OpenPoseWrapper(PoseModel):
    """CMU OpenPose, run through whichever backend this machine can offer.

    Backend 1 (`pyopenpose`, BODY_25) is the official C++ build. It is preferred when present,
    but it cannot be pip-installed, so backend 2 (`torch`, COCO-18) runs the original CMU
    weights through an equivalent PyTorch graph instead. See the markdown above for what that
    does and does not change. If neither is available the wrapper reports why and marks itself
    unavailable, so the rest of the benchmark still runs.
    """
    name = "OpenPose"

    def __init__(self, openpose_root: Optional[Path] = None,
                 torch_weights: Optional[Path] = None,
                 net_resolution: str = "-1x368"):
        super().__init__()
        self.root = Path(openpose_root) if openpose_root else None
        self.torch_weights = Path(torch_weights) if torch_weights else None
        self.net_resolution = net_resolution
        self.backend = None                 # "pyopenpose" | "torch"
        self.skeleton_id = None             # "BODY_25"    | "COCO_18"
        self._op = self._wrapper = self._net = None
        self.available, self.unavailable_reason = False, ""
        self._reasons = []                  # why each backend was passed over

        if self._try_pyopenpose() or self._try_torch():
            return
        print(f"[{self.name}] UNAVAILABLE")
        for r in self._reasons:
            print(f"    - {r}")
        print("    " + "-" * 68)
        print("    To enable OpenPose, do either of:")
        print("      (a) download the converted CMU COCO-18 weights (209 MB) with")
        print("          python tools/setup_data.py --openpose")
        print("      (b) build the official C++ OpenPose with BUILD_PYTHON=ON and point")
        print("          OPENPOSE_ROOT at it in the configuration cell.")
        print("    Until then this model is SKIPPED and every other model, dataset, metric")
        print("    and figure in this notebook is unaffected.")
        print("    " + "-" * 68)
        self.unavailable_reason = "; ".join(self._reasons)

    # ---- backend 1: the official C++ build --------------------------------------------
    def _try_pyopenpose(self) -> bool:
        op_module = self._import_pyopenpose()
        if op_module is None:
            return False
        try:
            params = {
                "model_folder": str(self.root / "models") if self.root else "models/",
                "model_pose": "BODY_25",
                "net_resolution": self.net_resolution,
                "num_gpu": 0,                 # CPU only, matching the rest of the notebook
                "num_gpu_start": 0,
                "display": 0,
                "render_pose": 0,
            }
            self._wrapper = op_module.WrapperPython()
            self._wrapper.configure(params)
            self._wrapper.start()
        except Exception as exc:
            self._reasons.append(f"pyopenpose imported but failed to start "
                                 f"({type(exc).__name__}: {exc})")
            return False
        self._op = op_module
        self.backend, self.skeleton_id = "pyopenpose", "BODY_25"
        self.native_keypoints, self.native_skeleton = BODY25_KEYPOINTS, BODY25_SKELETON
        self.available = True
        print(f"[{self.name}] ready — native C++ build, BODY_25, "
              f"model folder {params['model_folder']}, CPU")
        return True

    def _import_pyopenpose(self):
        """Try every conventional location for the compiled pyopenpose module."""
        try:                                              # already on sys.path
            import pyopenpose as op
            return op
        except ImportError:
            pass
        if self.root is None or not self.root.is_dir():
            self._reasons.append(f"pyopenpose not importable and OPENPOSE_ROOT "
                                 f"({self.root}) does not exist")
            return None
        candidates = [self.root / "build" / "python" / "openpose" / "Release",
                      self.root / "build" / "python" / "openpose",
                      self.root / "build" / "python",
                      self.root / "python" / "openpose" / "Release",
                      self.root / "python"]
        for cand in candidates:
            if cand.is_dir():
                sys.path.append(str(cand))
                # the compiled module also needs the OpenPose DLLs/SOs on the library path
                for extra in [self.root / "build" / "x64" / "Release", self.root / "build" / "bin"]:
                    if extra.is_dir():
                        os.environ["PATH"] = str(extra) + os.pathsep + os.environ.get("PATH", "")
                try:
                    import pyopenpose as op
                    return op
                except ImportError:
                    continue
        self._reasons.append(f"no compiled pyopenpose module found under {self.root}")
        return None

    # ---- backend 2: the original CMU weights in PyTorch --------------------------------
    def _try_torch(self) -> bool:
        if not _TORCH_OK:
            self._reasons.append(f"PyTorch backend unusable ({_TORCH_WHY})")
            return False
        if self.torch_weights is None or not self.torch_weights.is_file():
            self._reasons.append(f"CMU COCO-18 weights not found at {self.torch_weights}")
            return False
        try:
            self._net = OpenPoseCOCONet.load_cmu_weights(self.torch_weights)
        except Exception as exc:
            self._reasons.append(f"weights present but would not load "
                                 f"({type(exc).__name__}: {exc})")
            return False
        self.backend, self.skeleton_id = "torch", "COCO_18"
        self.native_keypoints, self.native_skeleton = COCO18_KEYPOINTS, COCO18_SKELETON
        self.available = True
        size_mb = self.torch_weights.stat().st_size / 1e6
        print(f"[{self.name}] ready — original CMU COCO-18 weights via PyTorch, CPU "
              f"({size_mb:.0f} MB, {sum(p.numel() for p in self._net.parameters())/1e6:.1f}M params)")
        print(f"    note: research-grade executor, so its timings are an upper bound "
              f"on OpenPose's cost")
        return True

    # ---- inference ---------------------------------------------------------------------
    def predict(self, image_bgr, person_boxes=None) -> PredictionResult:
        """Bottom-up: the whole image goes in once, person_boxes is deliberately ignored."""
        if not self.available:
            return PredictionResult([], 0.0, self.name, 0, self.unavailable_reason)
        try:
            t0 = time.perf_counter()
            raw = (self._predict_pyopenpose(image_bgr) if self.backend == "pyopenpose"
                   else openpose_decode(self._net, image_bgr))
            dt = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            return PredictionResult([], 0.0, self.name, 1, f"{type(exc).__name__}: {exc}")

        persons = []
        for arr in raw:
            arr = np.asarray(arr, dtype=np.float32)
            kp = {nm: (float(arr[j, 0]), float(arr[j, 1]), float(arr[j, 2]))
                  for j, nm in enumerate(self.native_keypoints) if j < arr.shape[0]}
            valid = arr[arr[:, 2] > 0]
            bbox = ((float(valid[:, 0].min()), float(valid[:, 1].min()),
                     float(valid[:, 0].max() - valid[:, 0].min()),
                     float(valid[:, 1].max() - valid[:, 1].min()))
                    if len(valid) else (0.0, 0.0, 0.0, 0.0))
            persons.append(PredictedPerson(
                keypoints=kp, bbox=bbox,
                score=float(valid[:, 2].mean()) if len(valid) else 0.0))
        return PredictionResult(persons, dt, self.name, 1)

    def _predict_pyopenpose(self, image_bgr):
        datum = self._op.Datum()
        datum.cvInputData = image_bgr
        self._wrapper.emplaceAndPop(self._op.VectorDatum([datum]))
        kps = getattr(datum, "poseKeypoints", None)        # (P, 25, 3) = x, y, confidence
        if kps is None or getattr(kps, "ndim", 0) != 3:
            return []
        return [kps[i] for i in range(kps.shape[0])]
