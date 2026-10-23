# =============================================================================
#  7.4  What the stability numbers say — read straight off the tables above
# =============================================================================
if stability_df.empty:
    print("No stability results to summarise.")
else:
    tbl = stability_df.set_index("model")

    def _best(col, low_is_good=True):
        s = tbl[col].dropna()
        if s.empty:
            return None, np.nan
        m = s.idxmin() if low_is_good else s.idxmax()
        return m, float(s[m])

    pairs = int(tbl["frame pairs"].max())
    n_scenes = len(sequences)
    n_frames = sum(len(v) for v in sequences.values())

    print("=" * 100)
    print("SECTION 7 FINDINGS — prediction stability")
    print("=" * 100)
    print(f"Measured over {n_frames} consecutive frames from {n_scenes} JRDB-Pose scene(s), "
          f"giving {pairs} tracked person-pairs on adjacent frames.\n")

    steadiest_det, det_v = _best("detection flicker_%")
    steadiest_kpt, kpt_v = _best("keypoint flicker_%")
    steadiest_jit, jit_v = _best("median jitter (x person size)")

    print(f"1. Least detection flicker : {steadiest_det} ({det_v:.1f}% of frame pairs lose "
          f"and regain the person)")
    for m in tbl.index:
        if m != steadiest_det:
            print(f"     vs {m}: {tbl.loc[m, 'detection flicker_%']:.1f}%")
    print()
    print(f"2. Least keypoint flicker  : {steadiest_kpt} ({kpt_v:.2f}% of joints appear or "
          f"vanish between adjacent frames)")
    for m in tbl.index:
        if m != steadiest_kpt:
            print(f"     vs {m}: {tbl.loc[m, 'keypoint flicker_%']:.2f}%")
    # This measure came out near zero for every model, and that near-zero IS the finding:
    # it separates two failure modes the static missing-keypoint rate of Section 6 merges.
    if tbl["keypoint flicker_%"].max() < 1.0:
        print("   -> Every model scores under 1% here, which is worth more than the ranking.")
        print("      Section 6 found missing-keypoint rates of tens of percent, but those")
        print("      omissions almost never flicker: a joint a model cannot produce in one")
        print("      frame it also cannot produce in the next. The omissions are STRUCTURAL")
        print("      (this joint, on this person, is genuinely not recoverable) rather than")
        print("      intermittent noise. That is the better news of the two for a downstream")
        print("      tracker, which can hold a persistent gap far more easily than a joint")
        print("      that blinks on and off.")
    print()
    print(f"3. Least jitter            : {steadiest_jit} (median {jit_v:.4f} of person size "
          f"of movement the person did not make)")
    for m in tbl.index:
        if m != steadiest_jit:
            print(f"     vs {m}: {tbl.loc[m, 'median jitter (x person size)']:.4f}")
    print()

    # Does the steadiest model also win on accuracy? That is the question the section exists
    # to answer, and it is only interesting when the answer is no.
    if not results_df.empty:
        jr = results_df[results_df["dataset"] == "JRDB-Pose"]
        if not jr.empty:
            most_accurate = jr.set_index("model")["PCK@0.5_%"].idxmax()
            winners = {steadiest_det, steadiest_kpt, steadiest_jit}
            print(f"Most accurate on JRDB-Pose (Section 5): {most_accurate}")
            if len(winners) == 1 and most_accurate in winners:
                print(f"  -> {most_accurate} is both the most accurate and the steadiest model "
                      f"on all three measures, so on this evidence accuracy and stability do "
                      f"not trade off against each other here.")
            else:
                print(f"  -> the steadiest model is not the same on every measure "
                      f"({', '.join(sorted(winners))}), so single-image accuracy alone does "
                      f"not predict how usable a model's output is over time. This is exactly "
                      f"what Section 5 could not have shown.")
    print("\nA caveat worth stating: jitter is measured only on joints the model emits in BOTH "
          "frames.\nA model that omits a difficult joint is therefore never charged jitter for "
          "it — it is charged\nkeypoint flicker instead. The three measures have to be read "
          "together, not ranked separately.")
