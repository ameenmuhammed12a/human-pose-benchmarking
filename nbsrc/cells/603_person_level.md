## 6.3 Per-person comparison: mostly-visible people vs heavily-occluded people

Section 6.1 splits *individual keypoints*. But occlusion in a real scene usually affects a
whole person at once — someone standing behind a table or another pedestrian has many joints
hidden together, and a pose estimator can fail on that person as a whole rather than joint by
joint. So the same data is regrouped **by person**.

Each ground-truth person is scored by the fraction of their annotated common joints that are
fully visible, and sorted into:

* **mostly visible** — at least 80% of their labelled common joints are `v=2`
* **partly occluded** — between 50% and 80%
* **heavily occluded** — under 50%

Every model is then evaluated on the *same* three groups of people. This also exposes a
failure mode that the keypoint-level split cannot see: whether a heavily occluded person is
detected as a person at all.
