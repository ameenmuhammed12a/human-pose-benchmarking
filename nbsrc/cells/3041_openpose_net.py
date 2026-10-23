# =============================================================================
#  OpenPose, PyTorch backend: the CMU COCO-18 network and its PAF decoder
# =============================================================================
# Everything in this cell is inert unless the PyTorch backend is actually used: it defines
# a graph and some array maths, and touches no files until the wrapper loads the weights.

from collections import OrderedDict

try:
    import torch
    import torch.nn as nn
    from scipy.ndimage import gaussian_filter
    _TORCH_OK, _TORCH_WHY = True, ""
except Exception as _exc:                       # torch/scipy absent -> backend simply unavailable
    _TORCH_OK, _TORCH_WHY = False, f"{type(_exc).__name__}: {_exc}"
    torch = nn = gaussian_filter = None


# ---- 1. The network -------------------------------------------------------------------
# pose_deploy_linevec: VGG-19 truncated to conv4_2 (output stride 8), then two CPM branches
# refined over six stages. Branch L1 regresses 38 part-affinity-field channels (19 limbs x 2
# components), branch L2 regresses 19 confidence maps (18 joints + background).

_VGG_CFG = [("conv1_1", 3, 64), ("conv1_2", 64, 64), ("pool1", 0, 0),
            ("conv2_1", 64, 128), ("conv2_2", 128, 128), ("pool2", 0, 0),
            ("conv3_1", 128, 256), ("conv3_2", 256, 256), ("conv3_3", 256, 256),
            ("conv3_4", 256, 256), ("pool3", 0, 0),
            ("conv4_1", 256, 512), ("conv4_2", 512, 512),
            ("conv4_3_CPM", 512, 256), ("conv4_4_CPM", 256, 128)]


def _named(layers):
    return nn.Sequential(OrderedDict(layers))


def _vgg():
    out = []
    for name, cin, cout in _VGG_CFG:
        if name.startswith("pool"):
            out.append((name, nn.MaxPool2d(2, 2, 0)))
        else:
            out.append((name, nn.Conv2d(cin, cout, 3, 1, 1)))
            out.append((name + "_relu", nn.ReLU(inplace=True)))
    return out


def _stage1(tail, cout):
    """Three 3x3 convs, a 1x1 widening to 512, then the 1x1 output layer."""
    out = []
    for i in (1, 2, 3):
        out.append((f"conv5_{i}_CPM_{tail}", nn.Conv2d(128, 128, 3, 1, 1)))
        out.append((f"conv5_{i}_CPM_{tail}_relu", nn.ReLU(inplace=True)))
    out.append((f"conv5_4_CPM_{tail}", nn.Conv2d(128, 512, 1, 1, 0)))
    out.append((f"conv5_4_CPM_{tail}_relu", nn.ReLU(inplace=True)))
    out.append((f"conv5_5_CPM_{tail}", nn.Conv2d(512, cout, 1, 1, 0)))
    return out


def _stage_n(stage, tail, cout):
    """Refinement stage: five 7x7 convs over [PAFs | heatmaps | features] = 185 channels."""
    out = []
    for i in (1, 2, 3, 4, 5):
        out.append((f"Mconv{i}_stage{stage}_{tail}",
                    nn.Conv2d(185 if i == 1 else 128, 128, 7, 1, 3)))
        out.append((f"Mconv{i}_stage{stage}_{tail}_relu", nn.ReLU(inplace=True)))
    out.append((f"Mconv6_stage{stage}_{tail}", nn.Conv2d(128, 128, 1, 1, 0)))
    out.append((f"Mconv6_stage{stage}_{tail}_relu", nn.ReLU(inplace=True)))
    out.append((f"Mconv7_stage{stage}_{tail}", nn.Conv2d(128, cout, 1, 1, 0)))
    return out


if _TORCH_OK:
    class OpenPoseCOCONet(nn.Module):
        """CMU OpenPose COCO-18 body network (pose_deploy_linevec.prototxt)."""

        N_PAF, N_HEAT, N_STAGES = 38, 19, 6

        def __init__(self):
            super().__init__()
            self.model0 = _named(_vgg())
            self.model1_1 = _named(_stage1("L1", self.N_PAF))
            self.model1_2 = _named(_stage1("L2", self.N_HEAT))
            for s in range(2, self.N_STAGES + 1):
                setattr(self, f"model{s}_1", _named(_stage_n(s, "L1", self.N_PAF)))
                setattr(self, f"model{s}_2", _named(_stage_n(s, "L2", self.N_HEAT)))

        def forward(self, x):
            feat = self.model0(x)
            paf, heat = self.model1_1(feat), self.model1_2(feat)
            for s in range(2, self.N_STAGES + 1):
                joined = torch.cat([paf, heat, feat], 1)
                paf = getattr(self, f"model{s}_1")(joined)
                heat = getattr(self, f"model{s}_2")(joined)
            return paf, heat

        @staticmethod
        def load_cmu_weights(path):
            """Load the converted CMU checkpoint, which keeps the original Caffe layer names."""
            net = OpenPoseCOCONet()
            raw = torch.load(str(path), map_location="cpu", weights_only=True)
            own = net.state_dict()
            # "model3_1.Mconv2_stage3_L1.weight" -> "Mconv2_stage3_L1.weight"
            mapped = {k: raw[k.split(".", 1)[1]] for k in own if k.split(".", 1)[1] in raw}
            if len(mapped) != len(own):
                missing = sorted(set(own) - set(mapped))[:4]
                raise KeyError(f"checkpoint matches {len(mapped)}/{len(own)} tensors; "
                               f"first unmatched: {missing}")
            net.load_state_dict(mapped, strict=True)
            net.eval()
            return net
else:
    OpenPoseCOCONet = None


# ---- 2. The decoder -------------------------------------------------------------------
# The network emits dense maps; turning them into people is the part-affinity-field parse
# described in Cao et al. (2017): find heatmap peaks, score every candidate limb by the line
# integral of the PAF along it, keep a one-to-one assignment per limb type, then merge limbs
# that share an endpoint into whole people.

# The 19 limbs the COCO model scores, on 0-based COCO-18 joint indices.
COCO18_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
                (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
                (2, 16), (5, 17)]
# Which pair of the 38 PAF channels carries each of those limbs.
COCO18_PAF_CHANNELS = [(12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1),
                       (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (28, 29), (30, 31),
                       (34, 35), (32, 33), (36, 37), (18, 19), (26, 27)]

OP_STRIDE      = 8       # network output stride
OP_PAD_VALUE   = 128     # grey padding, as in the reference implementation
OP_HEAT_THRESH = 0.10    # a heatmap local maximum below this is not a candidate joint
OP_PAF_THRESH  = 0.05    # a sample point along a limb counts as supporting it above this
OP_PAF_SAMPLES = 10      # points sampled along each candidate limb
OP_MIN_PARTS   = 4       # a person assembled from fewer joints than this is discarded


def _op_forward_maps(net, image_bgr, boxsize=368, max_side=1100):
    """One forward pass at a single scale; returns (heatmaps, PAFs) at full image resolution."""
    h, w = image_bgr.shape[:2]
    scale = boxsize / h
    if max(h, w) * scale > max_side:            # keep very wide frames affordable on CPU
        scale = max_side / max(h, w)
    small = cv2.resize(image_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ph = (OP_STRIDE - small.shape[0] % OP_STRIDE) % OP_STRIDE
    pw = (OP_STRIDE - small.shape[1] % OP_STRIDE) % OP_STRIDE
    padded = cv2.copyMakeBorder(small, 0, ph, 0, pw, cv2.BORDER_CONSTANT,
                                value=(OP_PAD_VALUE,) * 3)

    x = np.transpose(np.float32(padded), (2, 0, 1)) / 256.0 - 0.5
    with torch.no_grad():
        paf_t, heat_t = net(torch.from_numpy(x[None]))

    def to_image(t):
        a = t[0].numpy().transpose(1, 2, 0)
        a = cv2.resize(a, (0, 0), fx=OP_STRIDE, fy=OP_STRIDE, interpolation=cv2.INTER_CUBIC)
        a = a[: padded.shape[0] - ph, : padded.shape[1] - pw, :]
        return cv2.resize(a, (w, h), interpolation=cv2.INTER_CUBIC)

    return to_image(heat_t), to_image(paf_t)


def _op_find_peaks(heatmap):
    """Local maxima per joint channel -> [(x, y, score, running_id), ...] for each of 18 joints."""
    peaks_per_joint, next_id = [], 0
    for j in range(18):
        m = gaussian_filter(heatmap[:, :, j], sigma=3)
        up, down = np.zeros_like(m), np.zeros_like(m)
        left, right = np.zeros_like(m), np.zeros_like(m)
        up[1:, :], down[:-1, :] = m[:-1, :], m[1:, :]
        left[:, 1:], right[:, :-1] = m[:, :-1], m[:, 1:]
        is_peak = ((m >= up) & (m >= down) & (m >= left) & (m >= right) & (m > OP_HEAT_THRESH))
        ys, xs = np.nonzero(is_peak)
        found = [(int(x), int(y), float(m[y, x]), next_id + i)
                 for i, (x, y) in enumerate(zip(xs, ys))]
        next_id += len(found)
        peaks_per_joint.append(found)
    return peaks_per_joint


def _op_connect(peaks_per_joint, paf, image_shape):
    """Score candidate limbs by the PAF line integral and keep a one-to-one matching."""
    h, w = image_shape[:2]
    per_limb = []
    for (a, b), (ch_x, ch_y) in zip(COCO18_LIMBS, COCO18_PAF_CHANNELS):
        peaks_a, peaks_b = peaks_per_joint[a], peaks_per_joint[b]
        if not peaks_a or not peaks_b:
            per_limb.append([])
            continue
        field = paf[:, :, [ch_x, ch_y]]
        candidates = []
        for i, pa in enumerate(peaks_a):
            for k, pb in enumerate(peaks_b):
                vec = np.array([pb[0] - pa[0], pb[1] - pa[1]], dtype=np.float32)
                length = float(np.linalg.norm(vec))
                if length < 1e-4:
                    continue
                xs = np.clip(np.round(np.linspace(pa[0], pb[0], OP_PAF_SAMPLES)), 0, w - 1)
                ys = np.clip(np.round(np.linspace(pa[1], pb[1], OP_PAF_SAMPLES)), 0, h - 1)
                along = field[ys.astype(int), xs.astype(int), :] @ (vec / length)
                # the reference implementation penalises limbs that are long for the frame,
                # which suppresses links between different people
                prior = min(0.5 * h / length - 1.0, 0.0)
                scored = float(along.mean() + prior)
                if (along > OP_PAF_THRESH).sum() > 0.8 * OP_PAF_SAMPLES and scored > 0:
                    candidates.append((i, k, float(along.mean()),
                                       scored + pa[2] + pb[2]))
        candidates.sort(key=lambda c: -c[3])
        used_a, used_b, kept = set(), set(), []
        for i, k, score, _total in candidates:
            if i in used_a or k in used_b:
                continue
            used_a.add(i)
            used_b.add(k)
            kept.append((peaks_a[i][3], peaks_b[k][3], score))
            if len(kept) >= min(len(peaks_a), len(peaks_b)):
                break
        per_limb.append(kept)
    return per_limb


def _op_group(peaks_per_joint, per_limb):
    """Merge limbs sharing an endpoint into people: rows of 18 peak ids + [score, n_parts]."""
    flat = [p for joint in peaks_per_joint for p in joint]
    people = np.full((0, 20), -1.0)
    for li, (a, b) in enumerate(COCO18_LIMBS):
        for id_a, id_b, score in per_limb[li]:
            owners = [r for r in range(len(people))
                      if people[r][a] == id_a or people[r][b] == id_b]
            if len(owners) == 1:
                r = owners[0]
                if people[r][b] != id_b:
                    people[r][b] = id_b
                    people[r][-1] += 1
                    people[r][-2] += flat[int(id_b)][2] + score
            elif len(owners) == 2:
                r1, r2 = owners
                shared = (((people[r1][:18] >= 0).astype(int)
                           + (people[r2][:18] >= 0).astype(int)) == 2)
                if not shared.any():                     # disjoint halves of one person: merge
                    people[r1][:18] += people[r2][:18] + 1
                    people[r1][-2:] += people[r2][-2:]
                    people[r1][-2] += score
                    people = np.delete(people, r2, axis=0)
                else:                                    # contradictory: extend the first only
                    people[r1][b] = id_b
                    people[r1][-1] += 1
                    people[r1][-2] += flat[int(id_b)][2] + score
            elif li < 17:                                # the two ear-shoulder limbs never start
                row = np.full(20, -1.0)                  # a new person, as in the reference code
                row[a], row[b] = id_a, id_b
                row[-1] = 2
                row[-2] = flat[int(id_a)][2] + flat[int(id_b)][2] + score
                people = np.vstack([people, row])
    keep = [r for r in range(len(people))
            if people[r][-1] >= OP_MIN_PARTS and people[r][-2] / people[r][-1] >= 0.4]
    return people[keep], flat


def openpose_decode(net, image_bgr):
    """Full OpenPose parse of one image -> list of (18, 3) arrays of (x, y, confidence)."""
    heat, paf = _op_forward_maps(net, image_bgr)
    peaks = _op_find_peaks(heat)
    limbs = _op_connect(peaks, paf, image_bgr.shape)
    rows, flat = _op_group(peaks, limbs)

    out = []
    for row in rows:
        kp = np.zeros((18, 3), dtype=np.float32)
        for j in range(18):
            pid = int(row[j])
            if pid >= 0:
                x, y, score, _ = flat[pid]
                kp[j] = (x, y, score)
        out.append(kp)
    return out


print("OpenPose COCO-18 network and PAF decoder defined"
      f"{'' if _TORCH_OK else f' (inactive: {_TORCH_WHY})'}.")
