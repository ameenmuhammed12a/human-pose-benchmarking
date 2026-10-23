# =============================================================================
#  6.1  PCK and missing-keypoint rate per visibility group
# =============================================================================
VIS_ORDER  = [2, 1, 0]
VIS_LABEL  = {2: "visible (v=2)", 1: "occluded (v=1)", 0: "invisible (v=0)"}

jr_records = (records_df[records_df["dataset"] == "JRDB-Pose"].copy()
              if not records_df.empty else pd.DataFrame())

occl_rows, drop_rows = [], []
if jr_records.empty:
    print("No JRDB-Pose records available — the occlusion analysis needs JRDB-Pose.")
    print("Check JRDB_IMAGE_DIR / JRDB_LABEL_DIR in the configuration cell.")
else:
    for model in [m for m in MODEL_PALETTE if m in set(jr_records["model"])]:
        g = jr_records[jr_records["model"] == model]
        for v in VIS_ORDER:
            gv = g[g["visibility"] == v]
            if gv.empty:
                continue
            occl_rows.append({
                "model": model,
                "visibility": v,
                "group": VIS_LABEL[v],
                "keypoints": len(gv),
                "PCK@0.5_%": 100.0 * gv["correct_pck05"].mean(),
                "PCK@0.2_%": 100.0 * gv["correct_pck02"].mean(),
                "missing_kpt_rate_%": 100.0 * gv["missing"].mean(),
                "median_norm_dist": float(gv["norm_dist"].median(skipna=True)),
            })
        vis, occ = g[g["visibility"] == 2], g[g["visibility"] == 1]
        if not vis.empty and not occ.empty:
            pck_v, pck_o = 100 * vis["correct_pck05"].mean(), 100 * occ["correct_pck05"].mean()
            mis_v, mis_o = 100 * vis["missing"].mean(), 100 * occ["missing"].mean()
            drop_rows.append({
                "model": model,
                "PCK@0.5 visible_%": pck_v,
                "PCK@0.5 occluded_%": pck_o,
                "PCK drop (pp)": pck_v - pck_o,
                "PCK relative drop_%": 100.0 * (pck_v - pck_o) / pck_v if pck_v > 0 else np.nan,
                "missing visible_%": mis_v,
                "missing occluded_%": mis_o,
                "missing rise (pp)": mis_o - mis_v,
                "n visible kpts": len(vis),
                "n occluded kpts": len(occ),
            })

    occl_df = pd.DataFrame(occl_rows)
    drop_df = pd.DataFrame(drop_rows)

    print("=" * 100)
    print("PERFORMANCE BY GROUND-TRUTH VISIBILITY GROUP — JRDB-Pose, common 12 joints")
    print("=" * 100)
    display(occl_df.round({"PCK@0.5_%": 1, "PCK@0.2_%": 1,
                           "missing_kpt_rate_%": 1, "median_norm_dist": 3}))

    print("\n" + "=" * 100)
    print("OCCLUSION ROBUSTNESS — the drop from visible to occluded (SMALLER = MORE ROBUST)")
    print("=" * 100)
    display(drop_df.round({"PCK@0.5 visible_%": 1, "PCK@0.5 occluded_%": 1,
                           "PCK drop (pp)": 1, "PCK relative drop_%": 1,
                           "missing visible_%": 1, "missing occluded_%": 1,
                           "missing rise (pp)": 1}))

    occl_df.to_csv(RESULTS_DIR / "occlusion_by_visibility.csv", index=False)
    drop_df.to_csv(RESULTS_DIR / "occlusion_drop.csv", index=False)
    print(f"\nSaved -> {RESULTS_DIR / 'occlusion_by_visibility.csv'}")
    print(f"Saved -> {RESULTS_DIR / 'occlusion_drop.csv'}")
