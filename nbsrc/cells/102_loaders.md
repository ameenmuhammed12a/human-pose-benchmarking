## 1.2 Unified data structures and dataset loaders

Both datasets are now parsed into **one common in-memory structure**, so that every later
section (visualisation, inference, matching, metrics) is written once and works for both:

* `GTPerson` — one annotated person: a `(K, 3)` array of `(x, y, visibility)` keypoints,
  a bounding box in `(x, y, w, h)` form, an optional `track_id`, and the dataset's keypoint names.
* `ImageSample` — one image: its path, id, size, its list of `GTPerson`s, and a free-form
  `meta` dict for dataset-specific extras (CrowdPose's `crowdIndex`, JRDB's scene and camera).

Two dataset-specific details had to be handled, both discovered from the printout above:

1. **CrowdPose** stores 14 keypoints in the order
   `left/right shoulder, left/right elbow, left/right wrist, left/right hip, left/right knee,
   left/right ankle, head, neck`, and supplies a `bbox` per annotation plus a `crowdIndex`
   per image.
2. **JRDB-Pose** stores 17 keypoints in the order printed above
   (`head, right eye, left eye, right shoulder, center shoulder, left shoulder, ...`) — note
   this differs from the ordering given on the dataset website, which is why the loader reads
   the order out of the file's own `categories` block rather than hard-coding it.
   The per-camera JRDB files carry **no `bbox` field**, so the loader derives each person's box
   from the extent of their labelled keypoints plus a small margin, clipped to the image.
   This is stated explicitly because the derived box is what person-matching uses in Section 5.
