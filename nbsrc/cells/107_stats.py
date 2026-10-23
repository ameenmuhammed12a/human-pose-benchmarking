# ---- Instantiate and load both datasets ------------------------------------------
crowdpose = CrowdPoseDataset(CROWDPOSE_ANN_FILE, CROWDPOSE_IMAGE_DIR)
jrdb      = JRDBPoseDataset(JRDB_LABEL_DIR, JRDB_IMAGE_DIR)

t0 = time.perf_counter()
cp_samples = crowdpose.load(limit=N_CROWDPOSE_IMAGES, seed=RANDOM_SEED)
jr_samples = jrdb.load(limit=N_JRDB_IMAGES, seed=RANDOM_SEED)
print(f"Loaded in {time.perf_counter() - t0:.1f} s\n")

DATASETS = {d.name: d for d in (crowdpose, jrdb) if d.samples}
if not DATASETS:
    print("!! No dataset could be loaded. Check the paths in Section 0.1 before continuing.")


def print_dataset_report(ds: PoseDataset) -> dict:
    """Print the Task 1 statistics block for one dataset and return the stats dict."""
    st = ds.statistics()
    print("=" * 78)
    print(f"DATASET STATISTICS — {ds.name}")
    print("=" * 78)
    print(f"  images in benchmark subset  : {st['images']:,}")
    print(f"  person instances            : {st['person_instances']:,}")
    print(f"  keypoints defined per person: {st['keypoints_per_person(defined)']}")
    print(f"  persons per image           : mean {st['persons_per_image_mean']:.2f}, "
          f"max {st['persons_per_image_max']}")
    print(f"  labelled keypoints / person : mean {st['labelled_kpts_per_person_mean']:.2f} "
          f"(of {st['keypoints_per_person(defined)']})")
    print(f"  total keypoint slots        : {st['total_keypoint_slots']:,}")

    print(f"\n  keypoint order as defined in the annotation file:")
    for i, nm in enumerate(ds.keypoint_names):
        print(f"      {i:>2}  {nm}")

    total = sum(st["visibility_counts"].values()) or 1
    print(f"\n  visibility flag distribution over all {total:,} keypoint slots:")
    labels = getattr(ds, "VISIBILITY_LABELS", {0: "v=0", 1: "v=1", 2: "v=2"})
    for v in sorted(st["visibility_counts"]):
        c = st["visibility_counts"][v]
        bar = "#" * int(round(50 * c / total))
        print(f"      v={v} ({labels.get(v, '?'):<10}) {c:>8,}  {100*c/total:5.1f}%  {bar}")
    return st


cp_stats = print_dataset_report(crowdpose) if crowdpose.samples else None
print()
jr_stats = print_dataset_report(jrdb) if jrdb.samples else None

# ---- Dataset-specific extras ------------------------------------------------------
if crowdpose.samples:
    ci = [s.meta["crowdIndex"] for s in cp_samples if s.meta.get("crowdIndex") is not None]
    if ci:
        print("\n" + "=" * 78)
        print("CrowdPose crowd index (0 = uncrowded, 1 = heavily crowded)")
        print("=" * 78)
        print(f"  mean {np.mean(ci):.3f} | median {np.median(ci):.3f} | "
              f"min {np.min(ci):.3f} | max {np.max(ci):.3f}")
        # Top edge follows the data: CrowdPose crowdIndex can exceed 1.0.
        edges = [0.0, 0.2, 0.4, 0.6, 0.8, max(1.0001, float(np.max(ci)) + 1e-4)]
        hist, _ = np.histogram(ci, bins=edges)
        for lo, hi, c in zip(edges[:-1], edges[1:], hist):
            print(f"    crowdIndex [{lo:.1f}, {hi:.2f}) : {c:>4} images  "
                  + "#" * int(round(40 * c / max(1, len(ci)))))
        assert hist.sum() == len(ci), "histogram must account for every image"

if jrdb.samples:
    scenes = sorted({s.meta["scene"] for s in jr_samples})
    tracks = {(s.meta["scene"], p.track_id) for s in jr_samples for p in s.persons
              if p.track_id is not None}
    print("\n" + "=" * 78)
    print("JRDB-Pose subset composition")
    print("=" * 78)
    print(f"  scenes covered      : {len(scenes)}")
    for sc in scenes:
        n_img = sum(1 for s in jr_samples if s.meta["scene"] == sc)
        n_per = sum(len(s.persons) for s in jr_samples if s.meta["scene"] == sc)
        print(f"      {sc:<48} {n_img:>4} images, {n_per:>5} persons")
    print(f"  distinct person track_ids : {len(tracks)}  "
          f"(these enable the temporal-stability work listed in Section 7)")
