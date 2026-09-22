# Player detection + tracking. Set TRACKER to compare configs.
# Trackers tried: bytetrack.yaml | botsort.yaml | deepocsort.yaml

import cv2, os, time
from ultralytics import YOLO

# ---------------- CONFIG ----------------
VIDEO      = "videos/synced/right_sync_clip.mp4" # Change for other video left/right
MODEL      = "yolo11m.pt"           # yolo11m accepted. yolo11x was also tried.
TRACKER    = "trackers/deepocsort_tuned.yaml"     # We have tried three trackers, from trackers folder in delivery.

#MAX_FRAMES = 1000                  # None for full video
MAX_FRAMES = None

INFER_W    = 1920                   # 1920 to iterate fast; 3840 for full-res test

CONF       = 0.45
CLASSES    = [0]                    # person
OUT_DIR    = "FootageResults"
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
tag = os.path.splitext(os.path.basename(VIDEO))[0]
trk = TRACKER.split(".")[0]
out_path = os.path.join(OUT_DIR, f"{tag}_highres_task1.mp4")

model = YOLO(MODEL)

cap = cv2.VideoCapture(VIDEO)
src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps   = cap.get(cv2.CAP_PROP_FPS)
scale = INFER_W / src_w
infer_h = int(src_h * scale)
print(out_path)
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (INFER_W, infer_h))

frame_idx = 0
t_start = time.time()
per_frame_ms = []
print(f"Processing {VIDEO} with {TRACKER} at {INFER_W}x{infer_h} ...")

while True:
    ok, frame = cap.read()
    if not ok or (MAX_FRAMES and frame_idx >= MAX_FRAMES):
        break

    small = cv2.resize(frame, (INFER_W, infer_h))

    t0 = time.time()
    results = model.track(
        small, persist=True, tracker=TRACKER,
        conf=CONF, classes=CLASSES, verbose=False,
    )
    per_frame_ms.append((time.time() - t0) * 1000)

    # compact drawing: "id-conf"
    annotated = small.copy()
    r = results[0]
    if r.boxes is not None and r.boxes.id is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        ids   = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), tid, cf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(annotated, f"{tid} - {cf:.2f}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    writer.write(annotated)
    frame_idx += 1
    if frame_idx % 50 == 0:
        avg = sum(per_frame_ms[-50:]) / len(per_frame_ms[-50:])
        print(f"  {frame_idx} frames | last-50 avg inference {avg:.0f} ms")

cap.release()
writer.release()

total = time.time() - t_start
avg_inf = sum(per_frame_ms) / len(per_frame_ms)
print(f"Done. {frame_idx} frames in {total:.1f}s "
      f"({frame_idx/total:.2f} fps end-to-end) | avg inference {avg_inf:.0f} ms")
print(f"Saved: {out_path}")
