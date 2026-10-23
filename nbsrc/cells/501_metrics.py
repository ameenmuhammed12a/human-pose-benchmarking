# =============================================================================
#  Metric primitives
# =============================================================================

# COCO per-keypoint sigmas, restricted to the 12 common joints and kept in common order.
COMMON_SIGMAS = np.array([
    0.079, 0.079,   # shoulders
    0.072, 0.072,   # elbows
    0.062, 0.062,   # wrists
    0.107, 0.107,   # hips
    0.087, 0.087,   # knees
    0.089, 0.089,   # ankles
], dtype=np.float64)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two (x, y, w, h) boxes."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def greedy_match(pred_boxes: List[Sequence[float]], gt_boxes: List[Sequence[float]],
                 iou_thr: float = MATCH_IOU_THRESHOLD) -> Dict[int, int]:
    """Greedy highest-IoU-first matching. Returns {gt_index: pred_index}."""
    pairs = []
    for gi, gb in enumerate(gt_boxes):
        for pi, pb in enumerate(pred_boxes):
            iou = bbox_iou(pb, gb)
            if iou >= iou_thr:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True)
    matched, used_gt, used_pred = {}, set(), set()
    for iou, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        matched[gi] = pi
        used_gt.add(gi)
        used_pred.add(pi)
    return matched


def reference_size(gt_common: np.ndarray, bbox: Sequence[float]) -> float:
    """Person-relative length used to normalise PCK distances.

    Primary: torso length (shoulder midpoint to hip midpoint) from the ground truth.
    Fallback: 0.5 * sqrt(bbox area), which approximates torso length for a person-shaped box.
    """
    sh, hp = gt_common[[0, 1]], gt_common[[6, 7]]
    if (sh[:, 2] > 0).all() and (hp[:, 2] > 0).all():
        torso = float(np.linalg.norm(sh[:, :2].mean(axis=0) - hp[:, :2].mean(axis=0)))
        if torso > 1.0:
            return torso
    area = max(1.0, float(bbox[2]) * float(bbox[3]))
    return 0.5 * math.sqrt(area)


def person_oks(gt_common: np.ndarray, pred_common: np.ndarray,
               bbox: Sequence[float], conf_thr: float = KPT_CONF_THRESHOLD) -> Optional[float]:
    """COCO OKS between one GT person and one predicted person, on the common joints."""
    valid = gt_common[:, 2] > 0
    if not valid.any():
        return None
    area = max(1.0, float(bbox[2]) * float(bbox[3]))
    k = 2.0 * COMMON_SIGMAS
    d2 = ((pred_common[:, 0] - gt_common[:, 0]) ** 2 +
          (pred_common[:, 1] - gt_common[:, 1]) ** 2)
    e = d2 / (2.0 * area * (k ** 2) + np.finfo(float).eps)
    sim = np.exp(-e)
    # A joint the model did not emit (or emitted below threshold) contributes zero similarity.
    sim = np.where(pred_common[:, 2] >= conf_thr, sim, 0.0)
    return float(sim[valid].mean())


print("Metric primitives ready (IoU matching, OKS, PCK reference size).")
