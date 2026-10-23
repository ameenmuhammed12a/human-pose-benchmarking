# =============================================================================
#  6.2  Grouped bar chart: accuracy and omissions by visibility group
# =============================================================================
if jr_records.empty or occl_df.empty:
    print("No JRDB-Pose records — skipping the occlusion charts.")
else:
    models = [m for m in MODEL_PALETTE if m in set(occl_df["model"])]
    groups = [v for v in VIS_ORDER if v in set(occl_df["visibility"])]
    xs = np.arange(len(groups), dtype=float)
    n = max(1, len(models))
    width = min(0.3, 0.82 / n)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # -- panel 1: PCK by visibility group -----------------------------------------
    ax = axes[0]
    for mi, model in enumerate(models):
        s = occl_df[occl_df["model"] == model].set_index("visibility")["PCK@0.5_%"]
        vals = [float(s.get(v, np.nan)) for v in groups]
        offs = (mi - (n - 1) / 2) * (width + 0.015)
        bars = ax.bar(xs + offs, vals, width, label=model, color=MODEL_PALETTE[model],
                      edgecolor="#fcfcfb", linewidth=1.4)
        for b, val in zip(bars, vals):
            if np.isfinite(val):
                ax.annotate(f"{val:.1f}", (b.get_x() + b.get_width() / 2, val),
                            textcoords="offset points", xytext=(0, 3), ha="center",
                            fontsize=7.5, color="#52514e")
    ax.set_xticks(xs)
    ax.set_xticklabels([VIS_LABEL[v] for v in groups], fontsize=8.5)
    ax.set_ylim(0, 105)
    _style_axes(ax, "PCK@0.5 (%)", "Accuracy by visibility group\n(higher is better)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # -- panel 2: missing rate by visibility group ---------------------------------
    ax = axes[1]
    for mi, model in enumerate(models):
        s = occl_df[occl_df["model"] == model].set_index("visibility")["missing_kpt_rate_%"]
        vals = [float(s.get(v, np.nan)) for v in groups]
        offs = (mi - (n - 1) / 2) * (width + 0.015)
        bars = ax.bar(xs + offs, vals, width, label=model, color=MODEL_PALETTE[model],
                      edgecolor="#fcfcfb", linewidth=1.4)
        for b, val in zip(bars, vals):
            if np.isfinite(val):
                ax.annotate(f"{val:.1f}", (b.get_x() + b.get_width() / 2, val),
                            textcoords="offset points", xytext=(0, 3), ha="center",
                            fontsize=7.5, color="#52514e")
    ax.set_xticks(xs)
    ax.set_xticklabels([VIS_LABEL[v] for v in groups], fontsize=8.5)
    ax.set_ylim(0, 105)
    _style_axes(ax, "missing keypoints (%)", "Omissions by visibility group\n(lower is better)")

    # -- panel 3: the robustness headline — visible -> occluded drop ----------------
    ax = axes[2]
    if not drop_df.empty:
        dxs = np.arange(len(drop_df), dtype=float)
        bars = ax.bar(dxs, drop_df["PCK drop (pp)"], 0.5,
                      color=[MODEL_PALETTE[m] for m in drop_df["model"]],
                      edgecolor="#fcfcfb", linewidth=1.4)
        for b, val in zip(bars, drop_df["PCK drop (pp)"]):
            ax.annotate(f"{val:.1f} pp", (b.get_x() + b.get_width() / 2, val),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=8.5, color="#52514e")
        ax.set_xticks(dxs)
        ax.set_xticklabels(drop_df["model"], fontsize=8.5)
        ax.axhline(0, color="#b9b9b4", linewidth=0.8)
        top = float(np.nanmax(drop_df["PCK drop (pp)"].values)) if len(drop_df) else 1.0
        ax.set_ylim(min(0, top * 1.3), max(1.0, top * 1.35))
    _style_axes(ax, "PCK@0.5 drop (percentage points)",
                "Occlusion robustness: visible -> occluded\n(SMALLER bar = more robust)")

    fig.suptitle("JRDB-Pose occlusion analysis — same images, same joints, "
                 "split by annotator visibility label", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(RESULTS_DIR / "occlusion_analysis.png", bbox_inches="tight")
    plt.show()
    print(f"Saved -> {RESULTS_DIR / 'occlusion_analysis.png'}")
