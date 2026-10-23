## 6.6 Conclusion — which model handles occlusion best

*(Every figure below is read from the tables produced by the cells above. All three models are
present: OpenPose runs here via CMU's own COCO-18 weights, as Section 3.3 sets out.)*

**YOLOv8-pose handles occlusion best, on every occlusion measure used here. The more
interesting result is that the ranking on occlusion is not the ranking on precision: OpenPose
is the most *accurate* model on the keypoints it chooses to emit, and simultaneously the most
*fragile* when a joint is hidden.**

**1. YOLO loses least accuracy when a joint is occluded; OpenPose loses most.** Moving from the
visible group to the occluded group, PCK@0.5 falls:

| model | visible | occluded | drop | relative drop |
|---|---|---|---|---|
| YOLOv8-pose | 82.6% | 69.1% | **13.5 pp** | **16.4%** |
| MediaPipe-Pose | 62.5% | 42.3% | 20.2 pp | 32.4% |
| OpenPose | 74.4% | 45.4% | **29.1 pp** | **39.1%** |

All three are scored on the *same* 1,600 occluded keypoints in the *same* images, so these
differences are attributable to occlusion handling rather than to scene difficulty. Note that
OpenPose starts *ahead* of MediaPipe on visible joints (74.4% vs 62.5%) and ends *behind* it on
occluded ones (45.4% vs 42.3% — within noise of each other). Occlusion costs OpenPose more than
it costs either of the others.

**2. That fragility is a refusal to guess, not a loss of precision.** On the keypoints each
model does emit, OpenPose is the *most precise of the three*: median normalised error 0.089 on
the visible group, against YOLO's 0.124 and MediaPipe's 0.118 — and it stays the most precise
under occlusion (0.142 against 0.193 and 0.206). Its weakness is that it declines to output the
joint at all: its missing-keypoint rate rises from **22.0% to 51.8%** under occlusion (+29.7 pp),
the largest rise of the three, against YOLO's 11.2% → 22.3% (+11.1 pp). This is a direct
consequence of the architecture. OpenPose is bottom-up: a joint exists only if a peak survives
in its confidence map, so a hidden joint simply produces no peak. YOLO regresses a fixed
17-joint skeleton per detected person, so it *always* proposes a location and lets the
confidence score express its doubt — and on this evidence its guesses under occlusion are more
often right than not.

**For the intended use case this matters more than the headline PCK.** A downstream tracker can
smooth a slightly misplaced joint; it cannot recover a joint that was never emitted. If the
requirement is a complete skeleton every frame, YOLO is the right choice. If the requirement is
that every emitted joint be trustworthy, OpenPose's reticence is a feature rather than a fault.

**3. Whole people, not just joints, tell the same story.** As people become more occluded
(Section 6.3), person-level detection falls from 91.7% → 85.0% for YOLO, 85.7% → 75.0% for
OpenPose, and 75.5% → 55.0% for MediaPipe. On the heavily-occluded group, keypoint accuracy is
60.8% (YOLO), 38.1% (OpenPose) and 21.1% (MediaPipe): the gap between the models *widens* as
the task gets harder, which is the clearest single argument for testing on occluded data rather
than on clean benchmark images.

**4. Crowding separates the two architectures cleanly.** On CrowdPose (Section 6.4), going from
frames with 1–2 people to frames with 10 or more, YOLO holds at **85.7% → 82.7%** and OpenPose
at **73.7% → 73.3%** — essentially flat. MediaPipe falls from **69.1% → 52.7%**. Both bottom-up
and top-down-with-detector models see the whole scene and can use it; MediaPipe is handed one
crop at a time and cannot tell which limb in that crop belongs to the person it was asked
about. This is the failure mode CrowdPose was built to expose, and it is visible here.

**5. A confound worth stating, which the scale analysis catches.** On JRDB-Pose the crowding
trend *inverts* — sparse frames score worse than busy ones (YOLO 63.9% at 1–2 people against
88.9% at 6–9). That is not a reversal of the crowding effect: JRDB's sparse frames are mostly
corridors and open spaces where the few people present are far from the robot and therefore
small. The scale breakdown confirms it — accuracy tracks person height closely, and on
CrowdPose every model degrades sharply on people under 80 px (YOLO 87.9% → 52.9%). Person
*size*, not person *count*, is the dominant difficulty factor on JRDB-Pose, and crowding
analysis on that dataset should not be read without it.

**6. Which joints are fragile is not which joints are hard.** The per-joint breakdown shows
shoulders and hips barely move under occlusion for any model (some even improve slightly, since
the occluded subset of those joints is a different and not necessarily harder sample). The cost
is concentrated entirely in the extremities: for OpenPose, elbows and wrists lose 43–46 pp,
against 6–19 pp for YOLO. Torso joints are anchored by the rest of the body and can be inferred;
a hidden wrist has no such support.

**Overall.** On this subset, for occluded multi-person scenes, the ranking is YOLOv8-pose,
then OpenPose, then MediaPipe Pose — but the three fail differently enough that the ranking
alone is misleading. YOLO is the most robust and the most complete. OpenPose is the most
precise and the most honest about what it does not know, at the cost of large gaps. MediaPipe,
built for single-person capture, is the wrong tool for this setting and its numbers say so.
Section 7 adds a dimension none of this can see: whether the output is *steady* from frame to
frame, where the ordering changes again.
