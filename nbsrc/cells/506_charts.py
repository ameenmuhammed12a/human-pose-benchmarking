# =============================================================================
#  Chart styling: a fixed, colour-blind-validated categorical palette.
#  Colours are assigned to MODELS in a fixed order and never cycled, so a model
#  keeps the same colour in every figure in this notebook.
# =============================================================================
MODEL_PALETTE = {
    "YOLOv8-pose":    "#2a78d6",   # blue
    "YOLO26-pose":    "#8a4fd0",   # violet
    "MediaPipe-Pose": "#eb6834",   # orange
    "OpenPose":       "#1baf7a",   # aqua
}
GRID_KW = dict(axis="y", color="#d8d8d4", linewidth=0.7, alpha=0.9)


def _style_axes(ax, ylabel: str, title: str) -> None:
    """Recessive grid and axes; the data carries the emphasis."""
    ax.set_axisbelow(True)
    ax.grid(**GRID_KW)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b9b4")
    ax.tick_params(colors="#52514e", length=0)
    ax.set_ylabel(ylabel, color="#52514e", fontsize=9)
    ax.set_title(title, fontsize=10, color="#0b0b0b", pad=8)


def grouped_bars(ax, table: pd.DataFrame, value_col: str, ylabel: str, title: str,
                 fmt: str = "{:.1f}") -> None:
    """One grouped bar panel: x = dataset, one bar per model, value labelled on each bar."""
    datasets = list(dict.fromkeys(table["dataset"]))
    models = [m for m in MODEL_PALETTE if m in set(table["model"])]
    if not datasets or not models:
        ax.axis("off")
        return
    n = len(models)
    width = min(0.28, 0.8 / n)
    xs = np.arange(len(datasets), dtype=float)
    for mi, model in enumerate(models):
        vals = []
        for d in datasets:
            row = table[(table["dataset"] == d) & (table["model"] == model)]
            vals.append(float(row[value_col].iloc[0]) if len(row) else np.nan)
        # 2px surface gap between adjacent bars keeps the groups legible.
        offs = (mi - (n - 1) / 2) * (width + 0.015)
        bars = ax.bar(xs + offs, vals, width, label=model,
                      color=MODEL_PALETTE[model], edgecolor="#fcfcfb", linewidth=1.4)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=7.5, color="#52514e")
    ax.set_xticks(xs)
    ax.set_xticklabels(datasets, fontsize=9)
    finite = table[value_col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite):
        ax.set_ylim(0, float(finite.max()) * 1.22)
    _style_axes(ax, ylabel, title)


if results_df.empty:
    print("No results to chart.")
else:
    panels = [
        ("PCK@0.5_%",          "percent of keypoints",   "Accuracy — PCK@0.5 (higher is better)", "{:.1f}"),
        ("OKS_mean(matched)",  "OKS (0–1)",              "Accuracy — mean OKS, matched persons",  "{:.3f}"),
        ("missing_kpt_rate_%", "percent of keypoints",   "Missing keypoint rate (lower is better)", "{:.1f}"),
        ("FPS",                "frames per second",      "Speed — FPS on CPU (higher is better)", "{:.2f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    for ax, (col, ylab, title, fmt) in zip(axes.ravel(), panels):
        grouped_bars(ax, results_df, col, ylab, title, fmt)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Model comparison across datasets — common 12-joint set, CPU inference",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(RESULTS_DIR / "benchmark_comparison.png", bbox_inches="tight")
    plt.show()
    print(f"Saved -> {RESULTS_DIR / 'benchmark_comparison.png'}")

    # The speed panel needs a caveat the other three do not: the models are not all running
    # on comparably optimised executors, and saying so is the difference between a fair
    # reading and a misleading one.
    if op_model.available and op_model.backend == "torch":
        print()
        print("NOTE on the speed panel: OpenPose here is CMU's own network run through a")
        print("  research-grade PyTorch graph, not CMU's optimised C++ binary (Section 3.3).")
        print("  Its FPS is therefore an UPPER BOUND on OpenPose's true cost — a tuned build")
        print("  would be faster. The accuracy panels are unaffected: same weights, same")
        print("  algorithm. MediaPipe's cost is also not like-for-like, since it is invoked")
        print("  once per annotated person rather than once per image; the summary table")
        print("  reports its per-call time separately for exactly that reason.")
