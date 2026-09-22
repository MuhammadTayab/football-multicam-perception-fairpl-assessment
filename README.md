# Football Multi-Camera Perception — Technical Assessment

Player and ball detection, tracking, and 3D localization from multi-camera football
footage. This repository contains the code, the fine-tuned ball model, tracker configs,
homography files, and the training logs for a four-part assessment.

The annotated result videos and the writeup are shared privately (they contain footage
of minors, kept off any public service by design). See **Results and footage** below.

---

## Tasks

- **Task 1 — Player detection and tracking** on the two fixed cameras (YOLO11m + Deep OC-SORT), with a tracker comparison (ByteTrack, BoT-SORT, Deep OC-SORT).
- **Task 2 — Ball tracking**: a fine-tuned single-class ball detector (YOLO11s) feeding a single-object Kalman trajectory with prediction-gating, coasting, interpolation, and smoothing.
- **Task 3 — 3D mapping**: per-camera homography from pitch geometry, projecting players and the ball onto one common top-down pitch from the stereo pair.
- **Task 4 — Broadcast (moving camera)**: the same detection/tracking pipelines applied to the moving broadcast footage, with analysis of what changes.

---

## Environment

- **Machine used:** MacBook Air M4, 16 GB RAM (ran primarily on CPU; Apple MPS gave no meaningful speedup for this workload)
- **Python:** 3.12.7
- **Key libraries:** Ultralytics YOLO 8.4.x, OpenCV 4.x, NumPy, SciPy

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python numpy scipy filterpy lap
```

---

## Repository structure

```
.
├── README.md
├── task1_tracking.py                # Task 1: player detection + tracking (tracker switchable)
├── task2_ball_trajectory.py         # Task 2: fine-tuned ball detection + Kalman trajectory pipeline
├── task3_topdown_map.py             # Task 3: project players + ball onto a common top-down pitch
├── task4_tracking_broadcast.py      # Task 4: detection + tracking on the moving broadcast footage
├── extract_frames-task2_1.py        # Uniformly-spaced frame extraction for ball annotation
├── annotator_task2_2.py             # Single-class, single-box ball annotator (crosshair + zoom)
├── homography_picker_task3_1.py     # Interactive per-camera pitch-point selection + homography
│
├── trackers/                        # Tracker configs
│   ├── deepocsort_tuned.yaml        #   primary tracker (tuned)
│   ├── botsort_tuned.yaml
│   └── botsort_orig.yaml
│
├── homography/                      # Saved per-camera homographies (matrices only, no footage)
│   ├── homography_left.npy
│   └── homography_right.npy
│
├── ball_finetune/                   # Ball-detector training
│   ├── data.yaml                    #   dataset config
│   ├── train.py                     #   training script
│   └── runs/detect/ball_finetune/train-2/
│       ├── args.yaml                #   training arguments
│       ├── results.csv              #   per-epoch metrics
│       ├── results.png              #   training curves
│       ├── BoxP_curve.png / BoxR_curve.png / BoxF1_curve.png / BoxPR_curve.png
│       ├── confusion_matrix.png / confusion_matrix_normalized.png
│       ├── labels.jpg
│       └── weights/best.pt          #   fine-tuned ball model (~19 MB)
│
├── FootageResults/                  # placeholder only — result videos shared privately
└── videos/                          # placeholder only — source footage kept local (not committed)
```

---

## How to run

Each script has a small config block at the top (input video, model, tracker, resolution,
frame range). Run directly, for example:

```bash
python task1_tracking.py            # players + tracking on a fixed camera
python task2_ball_trajectory.py     # ball detection + trajectory (uses the fine-tuned model)
python task3_topdown_map.py         # top-down localization from both cameras (needs both homographies)
python task4_tracking_broadcast.py  # tracking on the broadcast clip
```

Task 3 reads `homography/homography_left.npy` and `homography/homography_right.npy`.
To regenerate a homography for a camera, run `homography_picker_task3_1.py` and click the
pitch reference points in the printed order.

The fine-tuned ball model is at `ball_finetune/runs/detect/ball_finetune/train-2/weights/best.pt`.

---
