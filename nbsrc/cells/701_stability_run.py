# =============================================================================
#  7.1  Run every model over the consecutive-frame sequences
# =============================================================================
seq_dataset = JRDBPoseDataset(JRDB_LABEL_DIR, JRDB_SEQUENCE_DIR)
seq_samples = seq_dataset.load(limit=None) if JRDB_SEQUENCE_DIR.is_dir() else []

# Group into per-scene runs and keep only frames that really are adjacent, so a gap in the
# fetched data can never be mistaken for a person teleporting.
sequences: Dict[str, List[ImageSample]] = {}
for s in seq_samples:
    sequences.setdefault(s.meta["scene"], []).append(s)
for scene in sequences:
    sequences[scene].sort(key=lambda s: int(s.meta["frame"]))

if not sequences:
    print("No consecutive-frame sequences found.")
    print(f"  expected under: {JRDB_SEQUENCE_DIR}")
    print("  fetch them with:  python tools/fetch_jrdb_sequences.py")
    print("  Section 7 is skipped; nothing else in the notebook depends on it.")
    seq_predictions = {}
else:
    total_frames = sum(len(v) for v in sequences.values())
    print(f"{len(sequences)} sequence(s), {total_frames} consecutive frames, "
          f"{sum(len(s.persons) for v in sequences.values() for s in v)} person annotations")
    for scene, frames in sequences.items():
        nums = [int(f.meta["frame"]) for f in frames]
        gaps = sum(1 for a, b in zip(nums, nums[1:]) if b != a + 1)
        tracks = {p.track_id for f in frames for p in f.persons}
        print(f"  {scene[:42]:44s} frames {nums[0]:06d}-{nums[-1]:06d}  "
              f"({len(frames)} fetched, {gaps} gap(s), {len(tracks)} tracks)")

    # seq_predictions[(model, scene, frame_number)] = {track_id: (12, 3) common-space array}
    # A track absent from a frame's dict was annotated there but not matched by the model.
    seq_predictions: Dict[Tuple[str, str, int], Dict[int, np.ndarray]] = {}
    # seq_truth[(scene, frame_number)] = {track_id: ((12, 3) gt array, reference size)}
    # Ground truth does not depend on the model, so it is filled once, on the first pass.
    seq_truth: Dict[Tuple[str, int], Dict[int, Tuple[np.ndarray, float]]] = {}

    print(f"\nRunning {len(MODELS)} model(s) over {total_frames} frames "
          f"({len(MODELS) * total_frames} inferences), CPU only.\n")
    t_all = time.perf_counter()
    for model in MODELS:
        _, model_map, native_names = MODEL_MAPPINGS[model.name]
        t0, done = time.perf_counter(), 0
        for scene, frames in sequences.items():
            for sample in frames:
                img = sample.load_bgr()
                if img is None:
                    continue
                fnum = int(sample.meta["frame"])
                gt_boxes = [p.bbox for p in sample.persons]
                result = model.predict(img, person_boxes=gt_boxes)
                done += 1
                if result.error:
                    continue

                pred_common = [pred_to_common(p, model_map, native_names)
                               for p in result.persons]
                matches = greedy_match([p.bbox for p in result.persons], gt_boxes)

                frame_pred = {}
                truth_of_frame = seq_truth.setdefault((scene, fnum), {})
                for gi, person in enumerate(sample.persons):
                    if person.track_id is None:
                        continue
                    if person.track_id not in truth_of_frame:
                        gtc = gt_to_common(person, JRDB17_TO_COMMON)
                        truth_of_frame[person.track_id] = (gtc, reference_size(gtc, person.bbox))
                    pi = matches.get(gi)
                    if pi is not None:
                        frame_pred[person.track_id] = pred_common[pi]
                seq_predictions[(model.name, scene, fnum)] = frame_pred

                if done % 40 == 0:
                    print(f"  {model.name:<16} {done}/{total_frames} frames "
                          f"({time.perf_counter() - t0:.0f}s)", flush=True)
        print(f"  {model.name:<16} done — {done} frames in "
              f"{time.perf_counter() - t0:.0f}s\n", flush=True)
    print(f"All sequence inference finished in {time.perf_counter() - t_all:.0f}s.")
