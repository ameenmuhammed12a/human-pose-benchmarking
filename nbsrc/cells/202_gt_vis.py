# ---- 2.1  CrowdPose ground truth: one colour per person --------------------------
if crowdpose.samples:
    cp_vis = pick_samples(cp_samples, N_VIS_SAMPLES)
    imgs, titles = [], []
    for s in cp_vis:
        rgb = render_gt(s, crowdpose.skeleton, color_by_visibility=False)
        if rgb is None:
            print(f"[CrowdPose] could not read {s.image_path}")
            continue
        imgs.append(rgb)
        ci = s.meta.get("crowdIndex")
        titles.append(f"{s.meta['file_name']}  |  {len(s.persons)} persons"
                      + (f"  |  crowdIndex {ci:.2f}" if ci is not None else ""))
    show_grid(imgs, titles,
              "CrowdPose ground truth — 14 keypoints, one colour per person",
              ncols=2)
else:
    print("CrowdPose not loaded; skipping its ground-truth visualisation.")
