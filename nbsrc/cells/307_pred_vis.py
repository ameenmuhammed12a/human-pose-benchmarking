def render_prediction(image_bgr: np.ndarray, result: PredictionResult,
                      model: PoseModel, conf_thr: float = KPT_CONF_THRESHOLD) -> np.ndarray:
    """Draw a model's predicted skeletons (one colour per detected person) on a copy."""
    canvas = image_bgr.copy()
    for pi, person in enumerate(result.persons):
        arr = person.as_array(model.native_keypoints)
        # Reuse the visibility slot of draw_skeleton: keep only confident keypoints.
        vis = np.where(arr[:, 2] >= conf_thr, 2.0, 0.0)
        kp = np.column_stack([arr[:, 0], arr[:, 1], vis])
        draw_skeleton(canvas, kp, getattr(model, "native_skeleton", []),
                      color=_person_color_bgr(pi), bbox=person.bbox,
                      label=f"{person.score:.2f}", draw_invisible=False)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def compare_gt_and_predictions(samples: List[ImageSample], skeleton, dataset_name: str,
                               models: List[PoseModel], n: int = 2,
                               spread_key: Optional[str] = None) -> None:
    """For n sample images, show ground truth and every available model side by side."""
    chosen = pick_samples(samples, n, seed=RANDOM_SEED, spread_key=spread_key)
    if not chosen:
        print(f"[{dataset_name}] no samples to visualise.")
        return
    for s in chosen:
        img = s.load_bgr()
        if img is None:
            print(f"[{dataset_name}] could not read {s.image_path}; skipped.")
            continue
        panels = [render_gt(s, skeleton, color_by_visibility=(dataset_name == "JRDB-Pose"))]
        titles = [f"GROUND TRUTH — {len(s.persons)} persons"]
        gt_boxes = [p.bbox for p in s.persons]
        for m in models:
            res = m.predict(img, person_boxes=gt_boxes)
            if res.error:
                print(f"[{dataset_name}] {m.name} error on {s.image_path.name}: {res.error}")
                continue
            panels.append(render_prediction(img, res, m))
            titles.append(f"{m.name} — {len(res.persons)} persons, "
                          f"{res.inference_ms:.0f} ms"
                          + (f" ({res.n_model_calls} crops)" if res.n_model_calls > 1 else ""))
        name = s.meta.get("file_name", s.image_path.name)
        show_grid(panels, titles,
                  f"{dataset_name}: ground truth vs predictions — {name}",
                  ncols=min(3, len(panels)), figsize_per=(5.6, 4.0))


if MODELS:
    if crowdpose.samples:
        compare_gt_and_predictions(cp_samples, crowdpose.skeleton, "CrowdPose", MODELS, n=2)
    if jrdb.samples:
        compare_gt_and_predictions(jr_samples, jrdb.skeleton, "JRDB-Pose", MODELS, n=2,
                                   spread_key="scene")
else:
    print("No model available — skipping the qualitative comparison.")
