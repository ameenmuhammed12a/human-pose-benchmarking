## 1.3 Load the benchmark subsets and print dataset statistics

Both loaders are now run over a **seeded subset** of each dataset (sizes set by
`N_CROWDPOSE_IMAGES` / `N_JRDB_IMAGES` in the configuration cell), so that the whole notebook
completes in a reasonable time on CPU while remaining exactly reproducible.

The statistics printed below are the Task 1 deliverable: number of images, number of person
instances, keypoints per person, and — for JRDB-Pose — the **distribution of the 0/1/2
visibility labels**, which is the raw material for the occlusion analysis in Section 6.
