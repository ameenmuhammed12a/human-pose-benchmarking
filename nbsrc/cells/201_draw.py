# =============================================================================
#  Drawing helpers (shared by Sections 2, 3 and 6)
# =============================================================================

# Colour-blind-safe qualitative palette, used to give each person a distinct colour.
PERSON_COLORS = [
    (230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66),
    (0, 114, 178), (213, 94, 0), (204, 121, 167), (150, 150, 150),
]
# Visibility colours, in BGR for OpenCV: 0 invisible = red, 1 occluded = orange, 2 visible = green
VIS_COLORS_BGR = {0: (0, 0, 208), 1: (0, 112, 224), 2: (0, 160, 0)}
VIS_COLORS_RGB = {k: (v[2], v[1], v[0]) for k, v in VIS_COLORS_BGR.items()}


def _person_color_bgr(idx: int) -> Tuple[int, int, int]:
    r, g, b = PERSON_COLORS[idx % len(PERSON_COLORS)]
    return (b, g, r)


def draw_skeleton(img: np.ndarray,
                  keypoints: np.ndarray,
                  skeleton: Sequence[Tuple[int, int]],
                  color: Tuple[int, int, int] = (0, 200, 0),
                  color_by_visibility: bool = False,
                  bbox: Optional[Sequence[float]] = None,
                  label: Optional[str] = None,
                  radius: int = 3,
                  thickness: int = 2,
                  draw_invisible: bool = True) -> np.ndarray:
    """Draw one person's skeleton onto a BGR image (modified in place, and returned).

    keypoints : (K, 3) array of (x, y, v). v == 0 means invisible/unlabelled.
    skeleton  : sequence of 0-based (i, j) index pairs to join with a line.
    """
    h, w = img.shape[:2]

    def _ok(i: int) -> bool:
        """A keypoint we are willing to draw: in range and with usable coordinates."""
        if not (0 <= i < len(keypoints)):
            return False
        x, y, v = keypoints[i]
        if not draw_invisible and v <= 0:
            return False
        # CrowdPose (and COCO) store an unannotated keypoint as the placeholder (0, 0, 0).
        # Drawing those joins limbs to the image origin, so they are skipped. JRDB-Pose
        # annotates a real location even at v == 0, and those are still drawn (in red).
        if v <= 0 and x == 0 and y == 0:
            return False
        return np.isfinite(x) and np.isfinite(y) and -w <= x <= 2 * w and -h <= y <= 2 * h

    # --- limbs ---
    for i, j in skeleton:
        if _ok(i) and _ok(j):
            # A limb is only solid if both ends are at least occluded (v >= 1).
            faint = bool(keypoints[i][2] == 0 or keypoints[j][2] == 0)
            cv2.line(img,
                     (int(round(keypoints[i][0])), int(round(keypoints[i][1]))),
                     (int(round(keypoints[j][0])), int(round(keypoints[j][1]))),
                     color, max(1, thickness - 1) if faint else thickness,
                     lineType=cv2.LINE_AA)

    # --- joints ---
    for i in range(len(keypoints)):
        if not _ok(i):
            continue
        x, y, v = keypoints[i]
        c = VIS_COLORS_BGR.get(int(v), color) if color_by_visibility else color
        cv2.circle(img, (int(round(x)), int(round(y))), radius, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(round(x)), int(round(y))), radius, (255, 255, 255), 1,
                   lineType=cv2.LINE_AA)

    # --- box + label ---
    if bbox is not None and bbox[2] > 0 and bbox[3] > 0:
        x, y, bw, bh = [int(round(v)) for v in bbox]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 1, lineType=cv2.LINE_AA)
        if label:
            cv2.putText(img, label, (x, max(11, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1, cv2.LINE_AA)
    return img


def render_gt(sample: ImageSample,
              skeleton: Sequence[Tuple[int, int]],
              color_by_visibility: bool = False) -> Optional[np.ndarray]:
    """Return an RGB copy of the sample image with every ground-truth person drawn on it."""
    img = sample.load_bgr()
    if img is None:
        return None
    canvas = img.copy()
    for pi, person in enumerate(sample.persons):
        col = _person_color_bgr(pi)
        draw_skeleton(canvas, person.keypoints, skeleton,
                      color=col, color_by_visibility=color_by_visibility,
                      bbox=person.bbox,
                      label=(f"id{person.track_id}" if person.track_id is not None else f"p{pi}"))
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def show_grid(images: List[np.ndarray], titles: List[str], suptitle: str,
              ncols: int = 2, legend_handles: Optional[list] = None,
              figsize_per: Tuple[float, float] = (6.4, 4.4)) -> None:
    """Display a grid of already-rendered RGB images inline."""
    images = [im for im in images if im is not None]
    if not images:
        print(f"[{suptitle}] nothing to display (no readable images).")
        return
    nrows = int(math.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, im, ti in zip(axes, images, titles):
        ax.imshow(im)
        ax.set_title(ti, fontsize=8)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(suptitle, fontsize=11)
    # h_pad keeps each row's title clear of the image above it.
    fig.tight_layout(rect=(0, 0.02, 1, 0.96), h_pad=2.2)
    plt.show()


def pick_samples(samples: List[ImageSample], n: int, seed: int = RANDOM_SEED,
                 prefer_busy: bool = True, spread_key: Optional[str] = None) -> List[ImageSample]:
    """Seeded choice of n sample images, biased toward images with several people.

    spread_key names a `meta` field (e.g. "scene"); when given, the picks are spread
    round-robin across its distinct values so one busy scene cannot supply every sample.
    """
    if not samples:
        return []
    rng = random.Random(seed)
    if spread_key:
        groups: Dict[str, List[ImageSample]] = {}
        for s in samples:
            groups.setdefault(str(s.meta.get(spread_key, "?")), []).append(s)
        for key in groups:
            groups[key].sort(key=lambda s: -len(s.persons))
            groups[key] = groups[key][: max(3, n)]         # busiest few frames per group
        keys = sorted(groups)
        rng.shuffle(keys)
        picked, i = [], 0
        while len(picked) < n and any(groups[k] for k in keys):
            pool = groups[keys[i % len(keys)]]
            if pool:
                picked.append(pool.pop(rng.randrange(len(pool))))
            i += 1
        return picked
    pool = sorted(samples, key=lambda s: -len(s.persons))[: max(n * 3, n)] if prefer_busy else list(samples)
    return rng.sample(pool, min(n, len(pool)))


print("Drawing helpers ready.")
