# ---- Conversion helpers: everything becomes an (12, 3) array in common order ------

def gt_to_common(person: GTPerson, mapping: Dict[str, int]) -> np.ndarray:
    """Ground-truth person -> (12, 3) array of (x, y, visibility) in common joint order."""
    out = np.zeros((N_COMMON, 3), dtype=np.float32)
    for ci, cn in enumerate(COMMON_KEYPOINTS):
        ni = mapping[cn]
        if ni < len(person.keypoints):
            out[ci] = person.keypoints[ni]
    return out


def pred_to_common(person: PredictedPerson, mapping: Dict[str, int],
                   native_names: Sequence[str]) -> np.ndarray:
    """Predicted person -> (12, 3) array of (x, y, confidence) in common joint order."""
    out = np.zeros((N_COMMON, 3), dtype=np.float32)
    for ci, cn in enumerate(COMMON_KEYPOINTS):
        ni = mapping[cn]
        if ni < len(native_names):
            nm = native_names[ni]
            if nm in person.keypoints:
                out[ci] = person.keypoints[nm]
    return out


# ---- Sanity check: mapped GT joints must sit where those joints visually are -------
# A wrong mapping is the single most damaging silent error possible here, so it is checked
# numerically: for a correct mapping, mapped shoulders sit above mapped hips, which sit above
# mapped knees, in the great majority of upright pedestrian annotations.
def check_mapping_geometry(samples: List[ImageSample], mapping: Dict[str, int],
                           label: str) -> None:
    ok = tot = 0
    for s in samples:
        for p in s.persons:
            c = gt_to_common(p, mapping)
            sh = c[[0, 1]]; hp = c[[6, 7]]; kn = c[[8, 9]]
            if (sh[:, 2] > 0).all() and (hp[:, 2] > 0).all() and (kn[:, 2] > 0).all():
                tot += 1
                if sh[:, 1].mean() < hp[:, 1].mean() < kn[:, 1].mean():
                    ok += 1
    if tot == 0:
        print(f"  {label:<12}: not enough fully-labelled persons to check")
        return
    frac = ok / tot
    verdict = "OK" if frac > 0.8 else "SUSPICIOUS - check the mapping!"
    print(f"  {label:<12}: shoulders above hips above knees in {ok}/{tot} "
          f"({100*frac:.1f}%) of persons  ->  {verdict}")


print("Geometric sanity check of the ground-truth mappings")
print("-" * 78)
if crowdpose.samples:
    check_mapping_geometry(cp_samples, CROWDPOSE14_TO_COMMON, "CrowdPose")
if jrdb.samples:
    check_mapping_geometry(jr_samples, JRDB17_TO_COMMON, "JRDB-Pose")
