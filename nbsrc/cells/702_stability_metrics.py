# =============================================================================
#  7.2  Jitter, detection flicker and keypoint flicker over adjacent frame pairs
# =============================================================================
if not sequences or not MODELS:
    print("No sequence predictions — skipping the stability metrics.")
    stability_df = pd.DataFrame()
    stability_records = pd.DataFrame()
else:
    rows = []
    for model in MODELS:
        for scene, frames in sequences.items():
            nums = [int(f.meta["frame"]) for f in frames]
            for a, b in zip(nums, nums[1:]):
                if b != a + 1:                     # only genuinely adjacent frames
                    continue
                truth_a = seq_truth.get((scene, a), {})
                truth_b = seq_truth.get((scene, b), {})
                pred_a = seq_predictions.get((model.name, scene, a), {})
                pred_b = seq_predictions.get((model.name, scene, b), {})

                for track in set(truth_a) & set(truth_b):      # annotated in both frames
                    gt_a, ref_a = truth_a[track]
                    gt_b, ref_b = truth_b[track]
                    ref = 0.5 * (ref_a + ref_b)
                    in_a, in_b = track in pred_a, track in pred_b
                    rows.append({
                        "model": model.name, "scene": scene, "frame": a, "track": track,
                        "kind": "person", "joint": -1,
                        "detected_both": in_a and in_b,
                        "flicker": in_a != in_b,                # found once, lost once
                        "jitter": np.nan, "kpt_flicker": np.nan,
                    })
                    if not (in_a and in_b) or ref <= 0:
                        continue
                    pa, pb = pred_a[track], pred_b[track]
                    for j in range(N_COMMON):
                        gt_labelled = gt_a[j, 2] > 0 and gt_b[j, 2] > 0
                        emit_a = pa[j, 2] >= KPT_CONF_THRESHOLD
                        emit_b = pb[j, 2] >= KPT_CONF_THRESHOLD
                        if not gt_labelled:
                            continue
                        jit = np.nan
                        if emit_a and emit_b:
                            # spurious movement = predicted displacement minus true displacement
                            dp = pb[j, :2] - pa[j, :2]
                            dg = gt_b[j, :2] - gt_a[j, :2]
                            jit = float(np.linalg.norm(dp - dg)) / ref
                        rows.append({
                            "model": model.name, "scene": scene, "frame": a, "track": track,
                            "kind": "joint", "joint": j,
                            "detected_both": True, "flicker": np.nan,
                            "jitter": jit, "kpt_flicker": emit_a != emit_b,
                        })

    stability_records = pd.DataFrame(rows)
    if stability_records.empty:
        print("No adjacent frame pairs with a shared tracked person — nothing to measure.")
        stability_df = pd.DataFrame()
    else:
        person_rows = stability_records[stability_records["kind"] == "person"]
        joint_rows = stability_records[stability_records["kind"] == "joint"]

        summary = []
        for model in [m for m in MODEL_PALETTE if m in set(stability_records["model"])]:
            p = person_rows[person_rows["model"] == model]
            j = joint_rows[joint_rows["model"] == model]
            jit = j["jitter"].dropna()
            summary.append({
                "model": model,
                "frame pairs": int(p.shape[0]),
                "tracked both frames_%": 100.0 * float(p["detected_both"].mean()),
                "detection flicker_%": 100.0 * float(p["flicker"].mean()),
                "keypoint flicker_%": 100.0 * float(j["kpt_flicker"].mean()),
                "median jitter (x person size)": float(jit.median()) if len(jit) else np.nan,
                "mean jitter (x person size)": float(jit.mean()) if len(jit) else np.nan,
                "p90 jitter (x person size)": float(jit.quantile(0.9)) if len(jit) else np.nan,
                "jitter samples": int(len(jit)),
            })
        stability_df = pd.DataFrame(summary)

        print("=" * 112)
        print("PREDICTION STABILITY ACROSS CONSECUTIVE FRAMES — JRDB-Pose sequences")
        print("Every column is a COST: lower is steadier. Accuracy plays no part in any of them.")
        print("=" * 112)
        display(stability_df.round({
            "tracked both frames_%": 1, "detection flicker_%": 1, "keypoint flicker_%": 1,
            "median jitter (x person size)": 4, "mean jitter (x person size)": 4,
            "p90 jitter (x person size)": 4}))

        # per-joint jitter, which says where the instability actually lives
        joint_jitter = (joint_rows.dropna(subset=["jitter"])
                        .groupby(["model", "joint"])["jitter"]
                        .median().reset_index())
        joint_jitter["joint_name"] = [COMMON_KEYPOINTS[int(j)] for j in joint_jitter["joint"]]

        stability_df.to_csv(RESULTS_DIR / "stability_summary.csv", index=False)
        joint_jitter.to_csv(RESULTS_DIR / "stability_by_joint.csv", index=False)
        print(f"\nSaved -> {RESULTS_DIR / 'stability_summary.csv'}")
        print(f"Saved -> {RESULTS_DIR / 'stability_by_joint.csv'}")
