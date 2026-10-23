# =============================================================================
#  6.4b  Which joints occlusion actually costs — per-joint visible vs occluded, JRDB-Pose
# =============================================================================
if jr_records.empty:
    print("No JRDB-Pose records — skipping the per-joint occlusion breakdown.")
    joint_occl_df = pd.DataFrame()
else:
    rows = []
    for model in [m for m in MODEL_PALETTE if m in set(jr_records["model"])]:
        g = jr_records[jr_records["model"] == model]
        for j, joint in enumerate(COMMON_KEYPOINTS):
            vis = g[(g["joint"] == j) & (g["visibility"] == 2)]
            occ = g[(g["joint"] == j) & (g["visibility"] == 1)]
            if vis.empty or occ.empty:
                continue
            pck_v, pck_o = 100 * vis["correct_pck05"].mean(), 100 * occ["correct_pck05"].mean()
            rows.append({
                "model": model, "joint": joint,
                "PCK visible_%": pck_v,
                "PCK occluded_%": pck_o,
                "drop (pp)": pck_v - pck_o,
                "relative drop_%": 100.0 * (pck_v - pck_o) / pck_v if pck_v > 0 else np.nan,
                "missing visible_%": 100 * vis["missing"].mean(),
                "missing occluded_%": 100 * occ["missing"].mean(),
                "n visible": len(vis), "n occluded": len(occ),
            })
    joint_occl_df = pd.DataFrame(rows)

    print("=" * 108)
    print("PER-JOINT OCCLUSION COST — JRDB-Pose. A large drop means the joint is FRAGILE,")
    print("which is a different question from whether it is hard (Section 5's per-joint chart).")
    print("=" * 108)
    display(joint_occl_df.round(1))
    joint_occl_df.to_csv(RESULTS_DIR / "occlusion_by_joint.csv", index=False)
    print(f"\nSaved -> {RESULTS_DIR / 'occlusion_by_joint.csv'}")

    if not joint_occl_df.empty:
        models = [m for m in MODEL_PALETTE if m in set(joint_occl_df["model"])]
        joints = [j for j in COMMON_KEYPOINTS if j in set(joint_occl_df["joint"])]
        xs = np.arange(len(joints), dtype=float)
        n = max(1, len(models))
        width = min(0.28, 0.82 / n)
        fig, ax = plt.subplots(figsize=(13, 4.6))
        for mi, model in enumerate(models):
            s = (joint_occl_df[joint_occl_df["model"] == model]
                 .set_index("joint")["drop (pp)"])
            vals = [float(s.get(j, np.nan)) for j in joints]
            offs = (mi - (n - 1) / 2) * (width + 0.012)
            ax.bar(xs + offs, vals, width, label=model, color=MODEL_PALETTE[model],
                   edgecolor="#fcfcfb", linewidth=1.2)
        ax.axhline(0, color="#b9b9b4", linewidth=0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels([j.replace("_", " ") for j in joints], fontsize=8, rotation=20)
        ax.legend(frameon=False, fontsize=8.5)
        _style_axes(ax, "PCK@0.5 lost (percentage points)",
                    "How much accuracy each joint loses when the annotator marks it occluded "
                    "(JRDB-Pose; taller = more fragile)")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "occlusion_by_joint.png", bbox_inches="tight")
        plt.show()
        print(f"Saved -> {RESULTS_DIR / 'occlusion_by_joint.png'}")
