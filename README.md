# Benchmarking Human Pose Estimation Models

MSc Artificial Intelligence — Project Block 6, University of Bedfordshire, AY 2025/26.

Benchmarks four pose estimators — **YOLOv8-pose**, **YOLO26-pose**, **MediaPipe Pose** and
**OpenPose** — on **CrowdPose** and **JRDB-Pose** under one standardised, CPU-only evaluation
protocol. The focus is **occlusion robustness**, measured directly from JRDB-Pose's
per-keypoint visibility labels, plus **temporal stability** on consecutive video frames.

> **You may not need to run anything.** `pose_benchmarking.ipynb` is saved **with all outputs
> already in it** — every table, chart and annotated image. Open it in Jupyter, VS Code or
> GitHub and read it straight through. The same results are in `results/` (CSV + PNG), and
> the supervisor's four questions are answered in `supervisor_review/README.md`.
> Read on only if you want to re-run it.

---

## 1. Quick start

Run everything from **inside this folder**.

### Option A — one command (recommended)

```bash
python RUN_ME.py
```

It checks your Python version, offers to install missing packages, downloads the model
weights and datasets, and then verifies that the sample selection reproduces exactly. It asks
before installing or downloading anything, skips whatever is already there, and is safe to
interrupt and re-run. When it finishes it tells you how to open the notebook.

### Option B — by hand

```bash
pip install -r requirements.txt       # 1. dependencies                (~5 min)
python tools/setup_data.py            # 2. model weights + datasets    (~30-60 min, ~5 GB)
jupyter lab pose_benchmarking.ipynb   # 3. Run > Run All Cells         (~55 min on CPU)
```

Or run step 3 without opening Jupyter (outputs are saved back into the notebook):

```bash
python nbsrc/execute.py
```

Everything runs on **CPU** — no GPU, no CUDA, no cloud account.

---

## 2. Requirements

| | |
|---|---|
| **Python** | 3.9 – 3.13 |
| **ultralytics** | **8.4 or newer** — needed for YOLO26 (see below) |
| **Disk** | ~5.5 GB (mostly CrowdPose images) |
| **RAM** | 4 GB is enough |
| **GPU** | Not required, not used |
| **Internet** | For setup only; the notebook itself runs fully offline |

### Check this first: ultralytics version

YOLO26 needs **ultralytics 8.4+**. With an older version the YOLO26 weights fail to load
(`Can't get attribute 'Pose26'`); the notebook does not crash, it simply skips YOLO26 and
benchmarks the other three models. Check and upgrade with:

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
pip install -U "ultralytics>=8.4"
```

### Python 3.13 and MediaPipe

The brief specifies `mediapipe==0.10.14` (legacy `mp.solutions.pose` API), which only has
wheels for Python 3.8–3.12; mediapipe 0.10.30+ removed `mp.solutions` entirely. The notebook
handles both automatically:

| Your Python | Install | API the notebook uses |
|---|---|---|
| **3.9 – 3.12** | `mediapipe==0.10.14` (uncomment that line in `requirements.txt`) | legacy `mp.solutions.pose` |
| **3.13+** | `mediapipe>=0.10` (the default) | `mediapipe.tasks.PoseLandmarker` |

Both give the same 33-landmark BlazePose skeleton, so results are identical either way.

### Smaller PyTorch download (optional)

`pip install -r requirements.txt` pulls in PyTorch. To avoid a ~2.5 GB CUDA build, install the
CPU-only wheels first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 3. Getting the models and data

```bash
python tools/setup_data.py
```

Idempotent: anything already downloaded is left alone, so just re-run it if interrupted.

| Stage | Size | What it is |
|---|---|---|
| `models` | ~30 MB | `yolov8n-pose.pt`, `yolo26n-pose.pt`, `yolo11n-pose.pt`, MediaPipe landmarker |
| `openpose` | 209 MB | CMU's original OpenPose COCO-18 weights (PyTorch checkpoint) |
| `crowdpose-anns` | 131 MB | CrowdPose annotations (OpenMMLab mirror) |
| `crowdpose-images` | 2.4 GB | 20,000 CrowdPose images (Google Drive) — **the slow part** |
| `jrdb-labels` | 28 MB | JRDB-Pose per-scene label files |
| `jrdb-images` | 20 MB | the JRDB frames the benchmark uses |
| `jrdb-sequences` | 25 MB | consecutive JRDB frames for the stability analysis — slow server, can take ~1 h |

Run stages individually with `--only` / `--skip`:

```bash
python tools/setup_data.py --only models openpose
python tools/setup_data.py --skip crowdpose-images
```

The JRDB stages read only the frames they need out of a 21 GB remote zip using HTTP range
requests. They are quiet for a minute or two while reading the archive index — that is
normal, not a hang.

---

## 4. Running the notebook

```bash
jupyter lab pose_benchmarking.ipynb      # then Run > Run All Cells
```

**The first code cell checks every input** and prints `[OK ]` or `[MISSING]` for each one. If
something is missing the notebook still runs end to end: it reports what is absent and skips
those parts instead of crashing.

**Section 3.1 lists which models will run**, e.g.

```
  [RUN ] YOLOv8-pose      17 native keypoints
  [RUN ] YOLO26-pose      17 native keypoints
  [RUN ] MediaPipe-Pose   33 native keypoints via tasks
  [RUN ] OpenPose         18 native keypoints via torch
```

If YOLO26 shows `SKIP`, upgrade ultralytics (Section 2).

**Runtime:** about **55 minutes** on a 12-core CPU (reference run: 53 min). OpenPose costs
roughly 2 s per image and dominates; the other three models take a few minutes in total.
Nothing needs supervising.

### What you get

| Where | What |
|---|---|
| the notebook | 56 cells, 14 inline figures, all tables and section-by-section commentary |
| `results/*.csv` | benchmark summary, per-joint, occlusion (by visibility / person / crowd / joint), accuracy by scale, stability, keypoint mapping, environment, raw per-keypoint records |
| `results/*.png` | the charts at full resolution |

### Headline results (reference run)

| Dataset | Model | PCK@0.5 | OKS | Missing kpts |
|---|---|---|---|---|
| CrowdPose | YOLOv8-pose | 85.1% | 0.823 | 9.2% |
| CrowdPose | YOLO26-pose | 84.2% | 0.846 | 11.6% |
| CrowdPose | MediaPipe Pose | 60.2% | 0.584 | 19.9% |
| CrowdPose | OpenPose | 70.3% | 0.772 | 27.1% |
| JRDB-Pose | YOLOv8-pose | 78.9% | 0.756 | 14.2% |
| JRDB-Pose | YOLO26-pose | 76.3% | 0.746 | 17.4% |
| JRDB-Pose | MediaPipe Pose | 57.0% | 0.666 | 34.5% |
| JRDB-Pose | OpenPose | 66.6% | 0.720 | 30.0% |

All sampling is seeded (`RANDOM_SEED = 42`), so accuracy figures reproduce exactly on the same
software versions. Only the timing / FPS columns vary with machine load.

---

## 5. YOLO generations comparison (optional)

A separate script compares three YOLO generations — **YOLOv8, YOLO11 and YOLO26** — using the
notebook's own loaders, mapping and metrics:

```bash
python tools/run_yolo_generations.py      # ~5 min, no OpenPose involved
```

It writes `results/yolo_generations_summary.csv`, `yolo_generations_occlusion.csv`,
`yolo_generations_occlusion_drop.csv` and two charts. It needs ultralytics 8.4+ and all three
YOLO weight files in `models/` (`python tools/setup_data.py --only models`).

---

## 6. Changing the settings

Everything tunable is in **one cell** — Section 0.1, the first code cell:

```python
PROJECT_ROOT = Path.cwd()      # defaults to this folder; set an explicit path to override

N_CROWDPOSE_IMAGES = 600       # CrowdPose images to benchmark
N_JRDB_IMAGES      = 600       # JRDB frames to benchmark (198 are available)
N_VIS_SAMPLES      = 4         # sample images shown per dataset

RANDOM_SEED         = 42
KPT_CONF_THRESHOLD  = 0.30     # below this a predicted keypoint counts as "missing"
MATCH_IOU_THRESHOLD = 0.30     # bbox IoU needed to match a prediction to a person
PCK_ALPHA           = 0.50     # PCK@0.5, the headline accuracy metric
```

To keep the data on another drive: `PROJECT_ROOT = Path(r"D:\pose_project")`.
Lower `N_CROWDPOSE_IMAGES` for a quicker run (runtime scales roughly linearly).

---

## 7. Editing the notebook

The notebook is **generated** from `nbsrc/cells/` — one file per cell, sorted by name
(`.md` → markdown cell, `.py` → code cell). To change it:

```bash
# edit the file in nbsrc/cells/
python nbsrc/build.py       # reassemble pose_benchmarking.ipynb  (clears all outputs!)
python nbsrc/execute.py     # re-run it and save the outputs back in
python tools/build_review_folder.py   # refresh supervisor_review/ from results/
```

Editing `pose_benchmarking.ipynb` directly in Jupyter also works, but the next `build.py`
would overwrite it from `nbsrc/cells/` — so make the same change in both places.

For a fast error check without Jupyter: `python nbsrc/smoke.py 403` runs every code cell up to
cell `403` headlessly.

---

## 8. How the sample images were chosen

Answered with runnable code (needs the datasets):

```bash
python supervisor_review/00_sample_selection/select_samples.py
```

It re-runs the seeded selection, re-renders the images and compares them byte-for-byte with
the ones in `supervisor_review/`, ending in `PASS` or `FAIL`. The written explanation is in
`supervisor_review/00_sample_selection/HOW_SAMPLES_WERE_SELECTED.md`.

---

## 9. Troubleshooting

**YOLO26 shows `SKIP` / `Can't get attribute 'Pose26'`.**
ultralytics is older than 8.4: `pip install -U "ultralytics>=8.4"`.

**The first cell says `MISSING` for some inputs.**
Run `python tools/setup_data.py`, or set `PROJECT_ROOT` to the folder that contains `data/`
and `models/`.

**`pip install mediapipe==0.10.14` fails with "no matching distribution".**
You are on Python 3.13+. Keep the default unpinned `mediapipe>=0.10` line.

**The CrowdPose download fails or stalls.**
Google Drive rate-limits large files. Wait and re-run, or download the images zip by hand from
the [CrowdPose repo](https://github.com/Jeff-sjtu/CrowdPose) to
`data/crowdpose/crowdpose_images.zip` — the script finds and extracts it.

**The JRDB fetch looks frozen.**
It is reading the index of a 21 GB remote archive. Give it two or three minutes.

**OpenPose is reported as unavailable.**
Fetch its weights: `python tools/setup_data.py --only openpose` (209 MB). The other models and
all tables still run without it.

**Jupyter says the kernel is missing.**
`pip install ipykernel` then `python -m ipykernel install --user --name python3`.

**Windows: `torch` / `ultralytics` import errors.**
Install the CPU-only PyTorch wheels first (Section 2), then reinstall the requirements.

---

## 10. Notes on the models

* **YOLOv8-pose and YOLO26-pose** both output the COCO-17 skeleton, so they share one mapping
  to the 12 common joints. YOLO26 (released Sep 2025) is Ultralytics' newest generation.
* **MediaPipe Pose** is a single-person model, so it degrades sharply in crowded scenes.
* **OpenPose** is a C++ project and cannot be pip-installed. By default the notebook runs
  **CMU's own released COCO-18 weights** (`pose_iter_440000`) through an equivalent PyTorch
  graph — same network, part affinity fields and limb grouping, different executor. Its
  *accuracy* is representative; its *speed* is an upper bound, not a tuned deployment figure.
  If you have a `CPU_ONLY` C++ build with `BUILD_PYTHON=ON`, set `OPENPOSE_ROOT` in the
  configuration cell and it is preferred automatically.
* All models are reduced to the **same 12 common joints** (shoulders, elbows, wrists, hips,
  knees, ankles), which is what makes them comparable.
* Results come from a **subset** (600 CrowdPose images, 198 scattered JRDB-Pose frames, plus
  consecutive JRDB-Pose frames for stability), sized to run on a CPU. Rankings hold across
  both datasets, but absolute numbers are subset figures, not full-dataset scores.

---

## 11. Project layout

```
README.md                    this file — start here
RUN_ME.py                    one-command setup + verification
pose_benchmarking.ipynb      the deliverable, saved with all outputs
requirements.txt             dependencies (with MediaPipe / OpenPose / YOLO26 notes)

supervisor_review/           the four supervisor questions, answered
  00_sample_selection/         how the samples were chosen -- runnable code
  01_dataset_samples/          raw frames from each dataset
  02_keypoints/                the same frames with ground-truth keypoints
  03_unified_keypoint_set/     the 12-joint unified set and its mappings
  04_comparison_criteria/      what is compared, on what criteria, and the results

results/                     CSV tables + PNG charts written by the notebook
models/                      yolov8n-pose.pt, yolo26n-pose.pt, yolo11n-pose.pt,
                             pose_landmarker_full.task, openpose_body_coco18.pth
data/                        datasets (created by tools/setup_data.py, ~4.9 GB)

tools/
  setup_data.py              one-command fetch of models + both datasets
  run_yolo_generations.py    YOLOv8 vs YOLO11 vs YOLO26 comparison
  build_review_folder.py     rebuilds supervisor_review/ from results/
  fetch_jrdb_labels.py       JRDB-Pose label files     } used by setup_data.py
  fetch_jrdb_images.py       JRDB-Pose frames          }
  fetch_jrdb_sequences.py    consecutive JRDB frames   }
  remote_zip.py              reads single files out of a remote zip over HTTP
  make_zip.py                packages a handover zip (excludes data/ and OpenPose weights)

nbsrc/
  cells/                     the notebook's cell sources, one file per cell
  build.py                   assembles cells/ into pose_benchmarking.ipynb
  execute.py                 runs the notebook headlessly and saves outputs
  smoke.py                   quick headless error check of the cells
```

To share the project without the 4.9 GB of data: `python tools/make_zip.py` creates
`pose_benchmarking_block6.zip`; the recipient re-fetches data with `tools/setup_data.py`.

---

## 12. Data sources and licensing

**CrowdPose** — images from the official Google Drive link in the
[CrowdPose repo](https://github.com/Jeff-sjtu/CrowdPose); annotations from the OpenMMLab
mirror. Cite Li et al., *CrowdPose*, CVPR 2019.

**JRDB-Pose** — from `jrdb.erc.monash.edu`, licensed **CC BY-NC-SA 3.0** (non-commercial). The
JRDB site asks users to [register](https://jrdb.erc.monash.edu/register/) before downloading.
Cite Vendrow et al., *JRDB-Pose*, CVPR 2023.

Both datasets are used here for non-commercial academic research only.
