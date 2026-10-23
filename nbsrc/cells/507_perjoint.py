# =============================================================================
#  5.4  Per-joint breakdown — which joints actually cost each model its accuracy
# =============================================================================
if records_df.empty:
    print("No per-keypoint records to break down.")
else:
    joint_pck = (records_df.groupby(["dataset", "model", "joint", "joint_name"])["correct_pck05"]
                 .mean().mul(100).reset_index(name="PCK@0.5_%"))
    joint_tab = (joint_pck.pivot_table(index="joint_name", columns=["dataset", "model"],
                                       values="PCK@0.5_%")
                 .reindex(COMMON_KEYPOINTS).round(1))
    print("=" * 100)
    print("PER-JOINT PCK@0.5 (%) — common joints only")
    print("=" * 100)
    display(joint_tab)
    joint_tab.to_csv(RESULTS_DIR / "per_joint_pck.csv")
    print(f"Saved -> {RESULTS_DIR / 'per_joint_pck.csv'}")

    datasets = list(dict.fromkeys(records_df["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.4 * len(datasets), 4.4), squeeze=False)
    for ax, dname in zip(axes.ravel(), datasets):
        sub = joint_pck[joint_pck["dataset"] == dname]
        models = [m for m in MODEL_PALETTE if m in set(sub["model"])]
        xs = np.arange(N_COMMON, dtype=float)
        n = max(1, len(models))
        width = min(0.34, 0.82 / n)
        for mi, model in enumerate(models):
            s = sub[sub["model"] == model].set_index("joint")["PCK@0.5_%"]
            vals = [float(s.get(j, np.nan)) for j in range(N_COMMON)]
            offs = (mi - (n - 1) / 2) * (width + 0.015)
            ax.bar(xs + offs, vals, width, label=model, color=MODEL_PALETTE[model],
                   edgecolor="#fcfcfb", linewidth=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels([k.replace("_", "\n") for k in COMMON_KEYPOINTS], fontsize=7)
        ax.set_ylim(0, 105)
        _style_axes(ax, "PCK@0.5 (%)", f"{dname} — accuracy by joint")
        if len(models) > 1:
            ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Where the errors are: PCK@0.5 per common joint", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RESULTS_DIR / "per_joint_pck.png", bbox_inches="tight")
    plt.show()

    # Distal joints (wrists, ankles) are the classic failure case; quantify it rather
    # than asserting it.
    prox = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    dist = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]
    print("\nProximal (shoulders + hips) vs distal (wrists + ankles) PCK@0.5:")
    for (dname, mname), g in records_df.groupby(["dataset", "model"]):
        p = 100 * g[g["joint_name"].isin(prox)]["correct_pck05"].mean()
        d = 100 * g[g["joint_name"].isin(dist)]["correct_pck05"].mean()
        print(f"  {dname:<11} {mname:<16} proximal {p:5.1f}%   distal {d:5.1f}%   "
              f"drop {p - d:5.1f} pp")
