# =============================================================================
#  The evaluation loop: run one model over one dataset and record every keypoint
# =============================================================================

def evaluate(model: PoseModel, dataset: PoseDataset, samples: List[ImageSample],
             gt_mapping: Dict[str, int], use_gt_boxes_for_crops: bool = True,
             progress_every: int = 25) -> Tuple[dict, pd.DataFrame]:
    """Run `model` over `samples` and return (summary dict, per-keypoint records DataFrame).

    The per-keypoint DataFrame is the raw material for both the Section 5 tables and the
    Section 6 occlusion analysis, so no metric is ever computed twice from different data.
    """
    model_map, native_names = None, None
    if model.name in MODEL_MAPPINGS:
        _, model_map, native_names = MODEL_MAPPINGS[model.name]
    if model_map is None:
        raise KeyError(f"No common-set mapping registered for model {model.name!r}")

    records, per_image_ms, per_call_ms = [], [], []
    n_gt = n_pred = n_matched = 0
    n_unreadable = n_errors = 0
    oks_values = []

    t_start = time.perf_counter()
    for si, sample in enumerate(samples, 1):
        img = sample.load_bgr()
        if img is None:                                   # unreadable file: skip, do not crash
            n_unreadable += 1
            continue

        gt_boxes = [p.bbox for p in sample.persons]
        # Single-person models are pointed at each annotated person in turn (see Section 3.1).
        boxes_arg = gt_boxes if use_gt_boxes_for_crops else None
        result = model.predict(img, person_boxes=boxes_arg)
        if result.error:
            n_errors += 1
            continue

        per_image_ms.append(result.inference_ms)
        if result.n_model_calls:
            per_call_ms.append(result.inference_ms / result.n_model_calls)

        pred_common = [pred_to_common(p, model_map, native_names) for p in result.persons]
        pred_boxes = [p.bbox for p in result.persons]
        n_pred += len(result.persons)

        matches = greedy_match(pred_boxes, gt_boxes)

        for gi, person in enumerate(sample.persons):
            gtc = gt_to_common(person, gt_mapping)
            if int((gtc[:, 2] > 0).sum()) < MIN_GT_KEYPOINTS:
                continue                                   # too sparsely annotated to score
            n_gt += 1
            pi = matches.get(gi)
            is_matched = pi is not None
            if is_matched:
                n_matched += 1
                pc = pred_common[pi]
                oks = person_oks(gtc, pc, person.bbox)
                if oks is not None:
                    oks_values.append(oks)
            else:
                pc, oks = None, None

            ref = reference_size(gtc, person.bbox)
            for j in range(N_COMMON):
                v = int(gtc[j, 2])
                # A joint is evaluable if it is labelled (v > 0), or if the dataset annotates
                # a real location even for v == 0 (JRDB-Pose "invisible"). CrowdPose v == 0
                # rows are placeholders at (0, 0) and must not be scored.
                if v <= 0 and not (dataset.annotates_invisible
                                   and (gtc[j, 0] != 0 or gtc[j, 1] != 0)):
                    continue
                if is_matched and pc[j, 2] >= KPT_CONF_THRESHOLD:
                    dist = float(math.hypot(pc[j, 0] - gtc[j, 0], pc[j, 1] - gtc[j, 1]))
                    missing = False
                else:
                    dist, missing = float("nan"), True
                records.append({
                    "dataset": dataset.name, "model": model.name,
                    "image_id": sample.image_id, "gt_person": gi,
                    "track_id": person.track_id,
                    "joint": j, "joint_name": COMMON_KEYPOINTS[j],
                    "visibility": v, "matched": is_matched, "missing": missing,
                    "dist": dist, "ref": ref,
                    "norm_dist": dist / ref if (ref > 0 and not missing) else float("nan"),
                    "correct_pck05": (not missing) and dist <= PCK_ALPHA * ref,
                    "correct_pck02": (not missing) and dist <= PCK_ALPHA_STRICT * ref,
                    "person_oks": oks,
                })
        if progress_every and si % progress_every == 0:
            print(f"    {si}/{len(samples)} images  "
                  f"({time.perf_counter() - t_start:.0f}s elapsed)", flush=True)

    df = pd.DataFrame(records)
    # Section 5 headline metrics use LABELLED keypoints only (visibility > 0), so CrowdPose
    # and JRDB-Pose are treated identically. The v == 0 rows JRDB provides are kept in `df`
    # for the occlusion analysis in Section 6, but excluded from the summary below.
    scored = df[df["visibility"] > 0] if not df.empty else df
    n_eval = len(scored)
    mean_ms = float(np.mean(per_image_ms)) if per_image_ms else float("nan")
    summary = {
        "dataset": dataset.name,
        "model": model.name,
        "images_evaluated": len(per_image_ms),
        "gt_persons": n_gt,
        "predicted_persons": n_pred,
        "matched_persons": n_matched,
        "detection_recall_%": 100.0 * n_matched / n_gt if n_gt else float("nan"),
        "evaluable_keypoints": n_eval,
        "OKS_mean(matched)": float(np.mean(oks_values)) if oks_values else float("nan"),
        "PCK@0.5_%": 100.0 * scored["correct_pck05"].mean() if n_eval else float("nan"),
        "PCK@0.2_%": 100.0 * scored["correct_pck02"].mean() if n_eval else float("nan"),
        "missing_kpt_rate_%": 100.0 * scored["missing"].mean() if n_eval else float("nan"),
        "mean_ms_per_image": mean_ms,
        "FPS": 1000.0 / mean_ms if mean_ms and mean_ms > 0 else float("nan"),
        "mean_ms_per_model_call": float(np.mean(per_call_ms)) if per_call_ms else float("nan"),
        "unreadable_images": n_unreadable,
        "inference_errors": n_errors,
    }
    return summary, df


print("Evaluation loop defined.")
