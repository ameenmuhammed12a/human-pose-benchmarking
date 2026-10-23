# =============================================================================
#  Instantiate the three wrappers. Any that cannot run is reported and skipped.
# =============================================================================
print("Initialising models (CPU only)\n" + "-" * 78)
yolo_model = YOLOPoseWrapper(YOLO_WEIGHTS)
print()
yolo26_model = YOLOPoseWrapper(YOLO26_WEIGHTS)
yolo26_model.name = "YOLO26-pose"
print()
mp_model   = MediaPipeWrapper(task_model=MEDIAPIPE_TASK_MODEL)
print()
op_model   = OpenPoseWrapper(OPENPOSE_ROOT, torch_weights=OPENPOSE_WEIGHTS)

ALL_MODELS   = [yolo_model, yolo26_model, mp_model, op_model]   # yolo26 added
MODELS       = [m for m in ALL_MODELS if m.available]     # only these are benchmarked
SKIPPED      = [m for m in ALL_MODELS if not m.available]

print("\n" + "=" * 78)
print("MODEL AVAILABILITY")
print("=" * 78)
for m in ALL_MODELS:
    mark = "RUN " if m.available else "SKIP"
    backend = f" via {m.backend}" if getattr(m, "backend", None) else ""
    detail = (f"{len(m.native_keypoints)} native keypoints{backend}"
              if m.available else m.unavailable_reason)
    print(f"  [{mark}] {m.name:<16} {detail}")
if not MODELS:
    print("\n!! No model is available — Sections 5 and 6 will produce empty tables.")
else:
    print(f"\n{len(MODELS)} of {len(ALL_MODELS)} models will be benchmarked: "
          f"{', '.join(m.name for m in MODELS)}")
    if SKIPPED:
        print(f"Skipped without error: {', '.join(m.name for m in SKIPPED)}")

# ---- Warm-up: the first call loads weights/graphs lazily, which would otherwise be -----
# ---- charged to the first image and distort the speed comparison in Section 5.    -----
_warm_src = (cp_samples or jr_samples)
if _warm_src and MODELS:
    _warm_img = _warm_src[0].load_bgr()
    if _warm_img is not None:
        print("\nWarming up models (timings below exclude this)...")
        for m in MODELS:
            t0 = time.perf_counter()
            m.warmup(_warm_img)
            print(f"  {m.name:<16} warm-up {1000*(time.perf_counter()-t0):7.1f} ms")
