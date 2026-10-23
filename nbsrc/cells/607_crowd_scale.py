# =============================================================================
#  6.4a  Accuracy against crowding and against person scale, on both datasets
# =============================================================================
CROWD_BANDS = [("1-2 people", 1, 3), ("3-5 people", 3, 6),
               ("6-9 people", 6, 10), ("10+ people", 10, 10 ** 6)]
SCALE_BANDS = [("small (<80 px)", 0, 80), ("medium (80-160 px)", 80, 160),
               ("large (160+ px)", 160, 10 ** 6)]


def _band_of(value: float, bands) -> Optional[str]:
    for label, lo, hi in bands:
        if lo <= value < hi:
            return label
    return None


if records_df.empty:
    print("No evaluation records — skipping the crowding and scale analysis.")
    crowd_df = scale_df = pd.DataFrame()
else:
    # Crowd size and person height are properties of the ground truth, so they are read once
    # from the datasets themselves and then applied to every model identically.
    crowd_of, height_of = {}, {}
    for ds, samples in [(crowdpose, cp_samples), (jrdb, jr_samples)]:
        if not ds.samples:
            continue
        for s in samples:
            crowd_of[(ds.name, s.image_id)] = len(s.persons)
            for gi, p in enumerate(s.persons):
                height_of[(ds.name, s.image_id, gi)] = float(p.bbox[3])

    work = records_df[records_df["visibility"] > 0].copy()      # labelled joints only
    keys = list(zip(work["dataset"], work["image_id"]))
    work["n_people"] = [crowd_of.get(k, np.nan) for k in keys]
    work["person_h"] = [height_of.get((d, i, g), np.nan) for d, i, g
                        in zip(work["dataset"], work["image_id"], work["gt_person"])]
    work["crowd_band"] = [_band_of(v, CROWD_BANDS) if np.isfinite(v) else None
                          for v in work["n_people"]]
    work["scale_band"] = [_band_of(v, SCALE_BANDS) if np.isfinite(v) else None
                          for v in work["person_h"]]

    def _breakdown(band_col, bands, label):
        rows = []
        for dataset in sorted(set(work["dataset"])):
            for model in [m for m in MODEL_PALETTE if m in set(work["model"])]:
                g = work[(work["dataset"] == dataset) & (work["model"] == model)]
                for band, _lo, _hi in bands:
                    gb = g[g[band_col] == band]
                    if gb.empty:
                        continue
                    per_person = gb.groupby(["image_id", "gt_person"])["matched"].first()
                    rows.append({
                        "dataset": dataset, "model": model, label: band,
                        "people": int(per_person.size),
                        "keypoints": len(gb),
                        "detected_%": 100.0 * float(per_person.mean()),
                        "PCK@0.5_%": 100.0 * gb["correct_pck05"].mean(),
                        "missing_kpt_rate_%": 100.0 * gb["missing"].mean(),
                    })
        return pd.DataFrame(rows)

    crowd_df = _breakdown("crowd_band", CROWD_BANDS, "crowd")
    scale_df = _breakdown("scale_band", SCALE_BANDS, "scale")

    print("=" * 104)
    print("ACCURACY AGAINST CROWDING — how each model holds up as the frame fills with people")
    print("=" * 104)
    display(crowd_df.round({"detected_%": 1, "PCK@0.5_%": 1, "missing_kpt_rate_%": 1}))

    print("\n" + "=" * 104)
    print("ACCURACY AGAINST PERSON SCALE — ground-truth box height in pixels")
    print("=" * 104)
    display(scale_df.round({"detected_%": 1, "PCK@0.5_%": 1, "missing_kpt_rate_%": 1}))

    crowd_df.to_csv(RESULTS_DIR / "occlusion_by_crowd.csv", index=False)
    scale_df.to_csv(RESULTS_DIR / "accuracy_by_scale.csv", index=False)
    print(f"\nSaved -> {RESULTS_DIR / 'occlusion_by_crowd.csv'}")
    print(f"Saved -> {RESULTS_DIR / 'accuracy_by_scale.csv'}")

    # --- chart -------------------------------------------------------------------------
    if not crowd_df.empty:
        datasets = sorted(set(crowd_df["dataset"]))
        models = [m for m in MODEL_PALETTE if m in set(crowd_df["model"])]
        fig, axes = plt.subplots(1, len(datasets) + 1,
                                 figsize=(5.0 * (len(datasets) + 1), 4.3))
        axes = np.atleast_1d(axes)
        for ax, dataset in zip(axes, datasets):
            bands = [b for b, _, _ in CROWD_BANDS
                     if b in set(crowd_df[crowd_df["dataset"] == dataset]["crowd"])]
            for model in models:
                s = (crowd_df[(crowd_df["dataset"] == dataset) & (crowd_df["model"] == model)]
                     .set_index("crowd")["PCK@0.5_%"])
                ys = [float(s.get(b, np.nan)) for b in bands]
                ax.plot(range(len(bands)), ys, marker="o", markersize=5, linewidth=2,
                        color=MODEL_PALETTE[model], label=model)
            ax.set_xticks(range(len(bands)))
            ax.set_xticklabels(bands, fontsize=8, rotation=12)
            ax.set_ylim(0, 100)
            _style_axes(ax, "PCK@0.5 (%)", f"{dataset}: accuracy vs crowding")
        # last panel: scale, pooled over datasets
        ax = axes[-1]
        bands = [b for b, _, _ in SCALE_BANDS if b in set(scale_df["scale"])]
        for model in models:
            ys = []
            for b in bands:
                g = scale_df[(scale_df["model"] == model) & (scale_df["scale"] == b)]
                ys.append(float((g["PCK@0.5_%"] * g["keypoints"]).sum() / g["keypoints"].sum())
                          if g["keypoints"].sum() else np.nan)
            ax.plot(range(len(bands)), ys, marker="s", markersize=5, linewidth=2,
                    color=MODEL_PALETTE[model], label=model)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels(bands, fontsize=8, rotation=12)
        ax.set_ylim(0, 100)
        _style_axes(ax, "PCK@0.5 (%)", "Both datasets: accuracy vs person size")
        axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
        fig.suptitle("What actually makes a frame hard: other people, and too few pixels",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(RESULTS_DIR / "crowd_and_scale.png", bbox_inches="tight")
        plt.show()
        print(f"Saved -> {RESULTS_DIR / 'crowd_and_scale.png'}")
