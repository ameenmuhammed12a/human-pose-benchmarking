## 3.2 Run the available models and compare predictions against ground truth

Each available model is now run on the same sample images from both datasets, and its output
is drawn beside the ground truth. This is a *qualitative* check that the wrappers are wired up
correctly — that coordinates are in full-image pixel space, that MediaPipe's crop coordinates
are mapped back properly, and that nothing is mirrored or transposed. The quantitative
comparison follows in Section 5.

Note when reading the MediaPipe panels: it is given the ground-truth person boxes, so it
returns exactly one skeleton per annotated person by construction. YOLO, by contrast, has to
find the people itself, so missed or spurious detections are visible in its panels.
