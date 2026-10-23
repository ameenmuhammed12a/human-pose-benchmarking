---

# Section 0 — Configuration and Environment

## 0.1 Configuration cell

**Every path and tunable parameter lives in this one cell.** To point the notebook at a
different copy of the datasets, change only the values here — nothing below needs editing.

The expected on-disk layout is:

```
DATA_ROOT/
├── crowdpose/
│   ├── images/                              # 20,000 CrowdPose .jpg files
│   └── crowdpose_annotations/
│       └── mmpose_crowdpose_val.json        # COCO-style annotation file
└── jrdb/
    ├── images/
    │   └── image_2/<scene>/000123.jpg       # individual camera frames
    └── labels/
        └── labels_2d_pose_coco/
            └── <scene>_image2.json          # one COCO-style JSON per scene+camera
```
