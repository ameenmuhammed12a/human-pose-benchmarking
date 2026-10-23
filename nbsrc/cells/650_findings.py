# =============================================================================
#  6.5  Findings, generated from the numbers actually computed above
#       (nothing here is written by hand; if the data changes, this text changes)
# =============================================================================
if jr_records.empty or drop_df.empty:
    print("Not enough JRDB-Pose data to draw an occlusion conclusion.")
else:
    print("=" * 90)
    print("AUTOMATICALLY DERIVED FINDINGS — JRDB-Pose occlusion test")
    print("=" * 90)

    d = drop_df.sort_values("PCK drop (pp)").reset_index(drop=True)
    best, worst = d.iloc[0], d.iloc[-1]

    print("\n1. Accuracy cost of occlusion (visible -> occluded, PCK@0.5):")
    for _, r in d.iterrows():
        print(f"     {r['model']:<16} {r['PCK@0.5 visible_%']:5.1f}%  ->  "
              f"{r['PCK@0.5 occluded_%']:5.1f}%   "
              f"drop {r['PCK drop (pp)']:5.1f} pp  ({r['PCK relative drop_%']:4.1f}% relative)")
    if len(d) > 1:
        print(f"\n   -> Smallest drop: {best['model']} ({best['PCK drop (pp)']:.1f} pp). "
              f"Largest: {worst['model']} ({worst['PCK drop (pp)']:.1f} pp).")
        print(f"      On this criterion alone, {best['model']} degrades least under occlusion.")
    else:
        print(f"\n   -> Only one model was available, so no cross-model comparison is possible.")

    print("\n2. Omission cost of occlusion (missing-keypoint rate):")
    for _, r in drop_df.sort_values("missing rise (pp)").iterrows():
        print(f"     {r['model']:<16} {r['missing visible_%']:5.1f}%  ->  "
              f"{r['missing occluded_%']:5.1f}%   rise {r['missing rise (pp)']:+5.1f} pp")

    print("\n3. Absolute accuracy on occluded keypoints (robustness is only useful if the")
    print("   starting accuracy is decent, so the drop must be read alongside this):")
    for _, r in drop_df.sort_values("PCK@0.5 occluded_%", ascending=False).iterrows():
        print(f"     {r['model']:<16} PCK@0.5 on occluded keypoints = "
              f"{r['PCK@0.5 occluded_%']:5.1f}%   "
              f"(over {int(r['n occluded kpts']):,} occluded keypoints)")

    if "person_df" in dir() and not person_df.empty:
        print("\n4. Whole-person occlusion (mostly visible vs heavily occluded people):")
        for model in person_df["model"].unique():
            s = person_df[person_df["model"] == model].set_index("person group")
            if "mostly visible" in s.index and "heavily occluded" in s.index:
                mv, ho = s.loc["mostly visible"], s.loc["heavily occluded"]
                print(f"     {model:<16} PCK@0.5 {mv['PCK@0.5_%']:5.1f}% -> {ho['PCK@0.5_%']:5.1f}% "
                      f"({mv['PCK@0.5_%'] - ho['PCK@0.5_%']:+5.1f} pp) | "
                      f"detected {mv['detected_%']:5.1f}% -> {ho['detected_%']:5.1f}% "
                      f"({ho['detected_%'] - mv['detected_%']:+5.1f} pp)")
            else:
                print(f"     {model:<16} not enough people in both bands to compare")

    print("\n5. Caveat carried into the conclusion:")
    print("     MediaPipe is given ground-truth person boxes, so occlusion can never cost it a")
    print("     person through mis-localisation; its 'detected' figure counts only the crops in")
    print("     which BlazePose returned no pose at all. YOLO must localise people itself, so")
    print("     only YOLO's detection numbers reflect genuine occlusion-induced detection")
    print("     failure. The two detection columns are therefore not directly comparable.")

    # A second, independent reading: precision on what each model DOES emit, versus how
    # much it omits. These can point in opposite directions, so both are reported.
    print("\n6. Precision-vs-omission split (median normalised error on emitted keypoints):")
    for model in [m for m in MODEL_PALETTE if m in set(occl_df["model"])]:
        s = occl_df[occl_df["model"] == model].set_index("visibility")
        if 2 in s.index and 1 in s.index:
            print(f"     {model:<16} median error  visible {s.loc[2, 'median_norm_dist']:.3f} -> "
                  f"occluded {s.loc[1, 'median_norm_dist']:.3f}  |  "
                  f"omitted  {s.loc[2, 'missing_kpt_rate_%']:5.1f}% -> "
                  f"{s.loc[1, 'missing_kpt_rate_%']:5.1f}%")
    print("     A model can be accurate on what it emits while omitting a great deal; PCK")
    print("     penalises both, which is why the missing-keypoint rate is reported separately.")
    print("=" * 90)
