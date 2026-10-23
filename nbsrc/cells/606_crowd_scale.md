### 6.4  Beyond the visibility flag: crowding, scale, and which joints occlusion costs

Sections 6.1–6.3 use JRDB-Pose's per-keypoint visibility flag, which is the most direct
evidence available but also the narrowest: it says whether the annotator judged a joint hidden,
and nothing about *why*. Three further cuts of the same per-keypoint records are made below,
and together they separate causes the visibility flag conflates.

1. **Crowding.** How many annotated people share the frame. This is occlusion by *other
   people* — the failure mode both benchmark datasets were built to expose, and the one that
   separates bottom-up models (OpenPose, YOLOv8-pose), which see the whole scene at once, from
   MediaPipe, which is handed one person at a time and cannot reason about who is in front.
   Run on both datasets, since it needs no visibility flag.

2. **Person scale.** The height of the ground-truth box, in pixels. A small person is hard for
   reasons that have nothing to do with being hidden — too few pixels to localise a wrist —
   and scale correlates with crowding, so reading the two apart matters before attributing a
   drop to occlusion.

3. **Which joints occlusion costs.** The per-joint accuracy drop from visible to occluded, on
   JRDB-Pose. Section 5's per-joint chart shows which joints are hard overall; this shows which
   are *fragile*, which is a different and more actionable question.
