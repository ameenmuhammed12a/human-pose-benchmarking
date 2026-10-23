# ---- 2.2  JRDB-Pose ground truth: keypoints colour-coded by visibility -----------
if jrdb.samples:
    jr_vis = pick_samples(jr_samples, N_VIS_SAMPLES, spread_key="scene")
    imgs, titles = [], []
    for s in jr_vis:
        rgb = render_gt(s, jrdb.skeleton, color_by_visibility=True)
        if rgb is None:
            print(f"[JRDB-Pose] could not read {s.image_path}")
            continue
        vis = np.concatenate([p.keypoints[:, 2] for p in s.persons]).astype(int)
        imgs.append(rgb)
        titles.append(f"{s.meta['scene']}  frame {s.meta['frame']}  ({s.meta['camera']})\n"
                      f"{len(s.persons)} persons | visible {int((vis==2).sum())}, "
                      f"occluded {int((vis==1).sum())}, invisible {int((vis==0).sum())}")

    legend = [Line2D([0], [0], marker="o", color="none", label=f"{lab} (v={v})",
                     markerfacecolor=np.array(VIS_COLORS_RGB[v]) / 255.0,
                     markeredgecolor="white", markersize=9)
              for v, lab in [(2, "visible"), (1, "occluded"), (0, "invisible")]]

    show_grid(imgs, titles,
              "JRDB-Pose ground truth — 17 keypoints, coloured by visibility flag "
              "(limb lines keep the per-person colour)",
              ncols=2, legend_handles=legend)

    # Quantify what the colours show, so the visual and the numbers agree.
    allv = np.concatenate([p.keypoints[:, 2] for s in jr_samples for p in s.persons]).astype(int)
    print("Across the whole JRDB-Pose benchmark subset:")
    for v, lab in [(2, "visible"), (1, "occluded"), (0, "invisible")]:
        print(f"   v={v} {lab:<10}: {int((allv==v).sum()):>7,}  ({100*(allv==v).mean():5.1f}% "
              f"of all annotated keypoint slots)")
    print(f"\n{100*(allv==1).mean():.1f}% of ground-truth keypoints are explicitly labelled as "
          f"occluded -- this is the signal Section 6 exploits.")
else:
    print("JRDB-Pose not loaded; skipping its ground-truth visualisation.")
