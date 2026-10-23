### 3.3  OpenPose — and how it is run here

OpenPose is the bottom-up baseline this benchmark is built around, and it is the one model of
the three that is not a Python package. It ships as a C++ project that must be compiled against
Caffe, and exposes Python only through the `pyopenpose` module that build produces. There is no
`pip install openpose`.

That leaves two ways to put OpenPose in a benchmark, and this notebook supports both:

| backend | what it is | skeleton | when it is used |
|---|---|---|---|
| `pyopenpose` | the official CMU C++ build, driven through its Python API | BODY_25 | if a build is found at `OPENPOSE_ROOT` |
| `torch` | the **original CMU COCO-18 weights** (`pose_iter_440000.caffemodel`), run through an equivalent PyTorch graph | COCO-18 | otherwise — this is the path used for the results below |

The second backend is not a different model. `models/openpose_body_coco18.pth` is a direct,
name-preserving conversion of the weights CMU released for `pose_deploy_linevec.prototxt`: all
184 tensors load into the architecture defined in the next cell with no renaming and no
retraining. The network, the part affinity fields, the peak detection and the greedy limb
grouping below are the same algorithm the C++ binary runs — what differs is the executor
(PyTorch on CPU instead of Caffe) and the head, because CMU's COCO release predicts 18 joints
and 19 limbs where BODY_25 predicts 25 and 26.

**What that means for the comparison.** All three models are reduced to the same 12 common
joints in Section 4, and all 12 exist in both OpenPose skeletons, so the accuracy numbers are
directly comparable either way. The only figure that carries a caveat is speed: the PyTorch
backend is a research-grade executor and is *slower* than CMU's optimised C++ binary, so the
OpenPose timings in Section 5 should be read as an upper bound on its cost, not as the speed of
a tuned deployment. This is stated again next to the speed chart.

A second, deliberate difference from the other two models: OpenPose is **bottom-up**. It is
given the whole image once and finds every person itself, with no detector and no ground-truth
boxes — exactly like YOLOv8-pose and unlike MediaPipe, which is pointed at one person at a time.
