# =============================================================================
#  7.3  Charts: the three stability costs, and where the jitter lives
# =============================================================================
if stability_df.empty:
    print("No stability results to chart.")
else:
    models = [m for m in MODEL_PALETTE if m in set(stability_df["model"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    panels = [("detection flicker_%", "% of frame pairs",
               "Detection flicker\n(person found in one frame, lost in the next)", "{:.1f}"),
              ("keypoint flicker_%", "% of joint pairs",
               "Keypoint flicker\n(joint emitted in one frame, dropped in the next)", "{:.1f}"),
              ("median jitter (x person size)", "median spurious movement",
               "Jitter\n(movement the person did not actually make)", "{:.3f}")]

    for ax, (col, ylab, title, fmt) in zip(axes, panels):
        vals = [float(stability_df.set_index("model")[col].get(m, np.nan)) for m in models]
        bars = ax.bar(range(len(models)), vals,
                      width=min(0.55, 0.8 / max(1, len(models)) * 1.6),
                      color=[MODEL_PALETTE[m] for m in models],
                      edgecolor="#fcfcfb", linewidth=1.4)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 3), ha="center",
                            fontsize=8, color="#52514e")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=8.5)
        finite = [v for v in vals if np.isfinite(v)]
        ax.set_ylim(0, (max(finite) * 1.25) if finite else 1)
        _style_axes(ax, ylab, title)

    fig.suptitle("Stability across consecutive JRDB-Pose frames — every bar is a cost, "
                 "so shorter is better", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(RESULTS_DIR / "stability_comparison.png", bbox_inches="tight")
    plt.show()
    print(f"Saved -> {RESULTS_DIR / 'stability_comparison.png'}")

    # --- where the jitter lives ---------------------------------------------------------
    if not joint_jitter.empty:
        joints = [j for j in COMMON_KEYPOINTS
                  if j in set(joint_jitter["joint_name"])]
        xs = np.arange(len(joints), dtype=float)
        n = max(1, len(models))
        width = min(0.28, 0.82 / n)
        fig, ax = plt.subplots(figsize=(13, 4.4))
        for mi, model in enumerate(models):
            s = (joint_jitter[joint_jitter["model"] == model]
                 .set_index("joint_name")["jitter"])
            vals = [float(s.get(j, np.nan)) for j in joints]
            offs = (mi - (n - 1) / 2) * (width + 0.012)
            ax.bar(xs + offs, vals, width, label=model, color=MODEL_PALETTE[model],
                   edgecolor="#fcfcfb", linewidth=1.2)
        ax.set_xticks(xs)
        ax.set_xticklabels([j.replace("_", " ") for j in joints], fontsize=8, rotation=20)
        ax.legend(frameon=False, fontsize=8.5)
        _style_axes(ax, "median jitter (x person size)",
                    "Which joints wobble: median spurious frame-to-frame movement per joint")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "stability_by_joint.png", bbox_inches="tight")
        plt.show()
        print(f"Saved -> {RESULTS_DIR / 'stability_by_joint.png'}")
