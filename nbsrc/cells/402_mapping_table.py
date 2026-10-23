# ---- Print the full mapping table (the Task 4 deliverable) -----------------------
def _named(mapping: Dict[str, int], names: Sequence[str]) -> List[str]:
    """Render 'index (native name)' for each common joint, in common order."""
    out = []
    for cn in COMMON_KEYPOINTS:
        i = mapping[cn]
        nm = names[i] if i < len(names) else "?"
        out.append(f"{i:>2}  {nm}")
    return out


cp_names = tuple(crowdpose.keypoint_names) or tuple(f"kpt{i}" for i in range(14))
jr_names = tuple(jrdb.keypoint_names) or tuple(f"kpt{i}" for i in range(17))

mapping_table = pd.DataFrame({
    "common joint": COMMON_KEYPOINTS,
    "CrowdPose-14": _named(CROWDPOSE14_TO_COMMON, cp_names),
    "JRDB-17":      _named(JRDB17_TO_COMMON, jr_names),
    "COCO-17 (YOLO)": _named(COCO17_TO_COMMON, COCO17_KEYPOINTS),
    "MediaPipe-33": _named(MEDIAPIPE33_TO_COMMON, MEDIAPIPE33_KEYPOINTS),
    "BODY_25 (OpenPose)": _named(BODY25_TO_COMMON, BODY25_KEYPOINTS),
})

print("=" * 118)
print(f"COMMON KEYPOINT MAPPING — every metric in Sections 5 and 6 uses only these "
      f"{N_COMMON} joints")
print("=" * 118)
with pd.option_context("display.max_colwidth", 30, "display.width", 200):
    print(mapping_table.to_string(index=False))
print("=" * 118)

print("\nJoints present natively but EXCLUDED from the common set:")
for label, names, mapping in [
        ("CrowdPose-14", cp_names, CROWDPOSE14_TO_COMMON),
        ("JRDB-17",      jr_names, JRDB17_TO_COMMON),
        ("COCO-17",      COCO17_KEYPOINTS, COCO17_TO_COMMON),
        ("MediaPipe-33", MEDIAPIPE33_KEYPOINTS, MEDIAPIPE33_TO_COMMON),
        ("BODY_25",      BODY25_KEYPOINTS, BODY25_TO_COMMON)]:
    used = set(mapping.values())
    left = [n for i, n in enumerate(names) if i not in used]
    print(f"  {label:<14} ({len(left):>2} excluded): {', '.join(left) if left else '-'}")

mapping_table.to_csv(RESULTS_DIR / "keypoint_mapping.csv", index=False)
print(f"\nSaved -> {RESULTS_DIR / 'keypoint_mapping.csv'}")
