# =============================================================================
#  6.3  Per-person occlusion analysis
# =============================================================================
PERSON_BANDS = [("mostly visible",   0.80, 1.01),
                ("partly occluded",  0.50, 0.80),
                ("heavily occluded", -0.01, 0.50)]


def person_band(visible_fraction: float) -> str:
    for label, lo, hi in PERSON_BANDS:
        if lo <= visible_fraction < hi:
            return label
    return "heavily occluded"


if jr_records.empty:
    print("No JRDB-Pose records — skipping the per-person occlusion analysis.")
else:
    # A person's visible fraction is a property of the ground truth, so it is computed once
    # from any single model's records and then applied to every model identically.
    ref_model = jr_records["model"].iloc[0]
    lab = jr_records[(jr_records["model"] == ref_model) & (jr_records["visibility"] > 0)]
    frac = (lab.assign(is_vis=lab["visibility"] == 2)
               .groupby(["image_id", "gt_person"])["is_vis"].mean())
    band_of = {k: person_band(v) for k, v in frac.items()}

    jr_records["person_band"] = [band_of.get((i, p), "heavily occluded")
                                 for i, p in zip(jr_records["image_id"], jr_records["gt_person"])]

    n_persons = pd.Series(list(band_of.values())).value_counts()
    print("Ground-truth people in the JRDB-Pose subset, by how much of them is visible:")
    for label, _, _ in PERSON_BANDS:
        print(f"  {label:<18}: {int(n_persons.get(label, 0)):>4} people")
    print()

    rows = []
    for model in [m for m in MODEL_PALETTE if m in set(jr_records["model"])]:
        g = jr_records[(jr_records["model"] == model) & (jr_records["visibility"] > 0)]
        for label, _, _ in PERSON_BANDS:
            gb = g[g["person_band"] == label]
            if gb.empty:
                continue
            # Person-level detection recall within this band.
            per_person = gb.groupby(["image_id", "gt_person"])["matched"].first()
            rows.append({
                "model": model,
                "person group": label,
                "people": int(per_person.size),
                "detected_%": 100.0 * float(per_person.mean()),
                "PCK@0.5_%": 100.0 * gb["correct_pck05"].mean(),
                "missing_kpt_rate_%": 100.0 * gb["missing"].mean(),
            })
    person_df = pd.DataFrame(rows)

    print("=" * 100)
    print("PERFORMANCE BY HOW OCCLUDED THE PERSON IS — JRDB-Pose")
    print("=" * 100)
    display(person_df.round({"detected_%": 1, "PCK@0.5_%": 1, "missing_kpt_rate_%": 1}))
    person_df.to_csv(RESULTS_DIR / "occlusion_by_person.csv", index=False)
    print(f"Saved -> {RESULTS_DIR / 'occlusion_by_person.csv'}")

    # --- chart -------------------------------------------------------------------
    if not person_df.empty:
        bands = [b for b, _, _ in PERSON_BANDS if b in set(person_df["person group"])]
        models = [m for m in MODEL_PALETTE if m in set(person_df["model"])]
        xs = np.arange(len(bands), dtype=float)
        n = max(1, len(models))
        width = min(0.3, 0.82 / n)
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
        for ax, (col, ylab, title) in zip(axes, [
                ("PCK@0.5_%", "PCK@0.5 (%)", "Keypoint accuracy by person occlusion level"),
                ("detected_%", "persons detected (%)",
                 "Person detection rate by occlusion level")]):
            for mi, model in enumerate(models):
                s = person_df[person_df["model"] == model].set_index("person group")[col]
                vals = [float(s.get(b, np.nan)) for b in bands]
                offs = (mi - (n - 1) / 2) * (width + 0.015)
                bars = ax.bar(xs + offs, vals, width, label=model, color=MODEL_PALETTE[model],
                              edgecolor="#fcfcfb", linewidth=1.4)
                for b, val in zip(bars, vals):
                    if np.isfinite(val):
                        ax.annotate(f"{val:.1f}", (b.get_x() + b.get_width() / 2, val),
                                    textcoords="offset points", xytext=(0, 3), ha="center",
                                    fontsize=7.5, color="#52514e")
            ax.set_xticks(xs)
            ax.set_xticklabels(bands, fontsize=8.5)
            ax.set_ylim(0, 108)
            _style_axes(ax, ylab, title)
        axes[0].legend(frameon=False, fontsize=8, loc="lower left")
        fig.suptitle("JRDB-Pose: performance as whole people become more occluded", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        fig.savefig(RESULTS_DIR / "occlusion_by_person.png", bbox_inches="tight")
        plt.show()
