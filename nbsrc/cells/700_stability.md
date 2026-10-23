---

# Section 7 — Prediction stability across consecutive frames

Everything up to here scores each image on its own. That is the right way to measure accuracy,
but it is blind to a property that matters just as much for the intended use — a robot tracking
people as it moves through a building — namely whether a model's output is *steady*. A model
that is 80% accurate but whose wrists twitch several centimetres between adjacent frames, and
whose people wink in and out of existence, is harder to build on than a model that is 75%
accurate and calm. Single-image PCK cannot tell those apart, because it never looks at two
frames together.

This section measures that directly. `tools/fetch_jrdb_sequences.py` fetches unbroken runs of
**consecutive annotated frames** from the busiest JRDB-Pose scenes — deliberately separate from
the scattered benchmark subset used above, which samples frames from across each scene and so
has no temporal structure at all. JRDB-Pose is the only one of the two datasets that can support
this: it carries a persistent `track_id` per person, so the same individual can be followed from
one frame to the next, and CrowdPose is a collection of unrelated still images.

Three measures are reported, each computed only over pairs of adjacent frames:

**1. Jitter.** For a joint the model emits in both frames of a pair, compare how far the
*prediction* moved with how far the *ground truth* moved, and take the difference, normalised by
person size. Real motion cancels out, so what is left is spurious movement. Reported as the
median over all joints and pairs, in units of person size — 0.02 means a typical joint wobbles
by 2% of the person's scale beyond whatever the person actually did.

**2. Detection flicker.** The fraction of adjacent frame pairs in which a tracked person is
found in one frame but lost in the other, though annotated in both. This is the model losing
somebody it had a moment ago.

**3. Keypoint flicker.** The same idea one level down: for people detected in *both* frames, the
fraction of joints emitted in one frame but dropped in the other. This is the intermittency that
Section 6 saw as a static missing-keypoint rate, now measured as instability over time.

All three are *cost* measures: lower is better, and unlike PCK they are independent of whether
the model is right, only of whether it is consistent.
