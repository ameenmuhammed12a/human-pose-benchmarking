# Benchmarking Human Pose Estimation Models

**MSc Artificial Intelligence — Project Block 6 (AY 2025/26)**
University of Bedfordshire

---

## Aim

This notebook benchmarks three human pose estimation systems — **OpenPose**, **MediaPipe Pose**
and **YOLO Pose (Ultralytics YOLOv8-pose)** — on two public multi-person datasets,
**CrowdPose** and **JRDB-Pose**, under a *single standardised evaluation protocol*.

The particular focus of the project is **occlusion handling**. JRDB-Pose was chosen precisely
because every ground-truth keypoint carries an explicit visibility label
(`0 = invisible`, `1 = occluded`, `2 = visible`). That lets occlusion robustness be
**measured directly** rather than inferred, which is the core novelty of the evaluation.

## What this notebook contains

| Section | Block 6 task | Content |
|---|---|---|
| 1 | Task 1 | Dataset loaders returning one unified structure, plus dataset statistics |
| 2 | Task 2 | Ground-truth skeleton visualisation (JRDB colour-coded by visibility) |
| 3 | Task 3 | `PoseModel` abstract interface + YOLO / MediaPipe / OpenPose wrappers |
| 4 | Task 4 | The common 12-joint keypoint set and all five explicit mapping tables |
| 5 | Task 5 | Matching, OKS / PCK / speed / missing-keypoint-rate benchmark |
| 6 | Task 6 | Occlusion analysis on JRDB-Pose using the visibility labels |
| 7 | — | Summary and next steps |

## Design rules followed throughout

1. **Everything runs on CPU.** No GPU is required at any point.
2. **All comparisons use one common 12-joint set** (Section 4), so no model is
   advantaged by having a richer native skeleton.
3. **Nothing is hard-coded from the literature.** Every number in every table below is
   produced by code executed in this notebook.
4. **Reproducible.** All sampling is seeded and the exact software/hardware
   configuration is recorded in Section 0.2.
