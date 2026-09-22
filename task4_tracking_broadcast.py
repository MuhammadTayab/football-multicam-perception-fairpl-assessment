# task4_broadcast.py — player tracking (Deep OC-SORT) + ball detection on the MOVING broadcast camera
import os
#os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import cv2, time, numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
VIDEO        = "videos/broadcast_clip.mp4"
PLAYER_MODEL = "yolo11m.pt"
BALL_MODEL   = "yolo11m.pt"
TRACKER      = "trackers/deepocsort_tuned.yaml"      # your agreed best tracker
MAX_FRAMES   = None                          # None for full video
WORK_W       = 1920                         # output width
PLAYER_CONF  = 0.35
BALL_CONF    = 0.25
BALL_IMGSZ   = 960
OUT_DIR      = "FootageResults"
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
tag = os.path.splitext(os.path.basename(VIDEO))[0]
out_path = os.path.join(OUT_DIR, f"{tag}_task4.mp4")

player_model = YOLO(PLAYER_MODEL)
ball_model   = YOLO(BALL_MODEL)

cap = cv2.VideoCapture(VIDEO)
src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps   = cap.get(cv2.CAP_PROP_FPS)
out_h = int(src_h * WORK_W / src_w)

writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (WORK_W, out_h))
if not writer.isOpened():
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (WORK_W, out_h))

frame_idx = 0
per_frame_ms = []
t0 = time.time()

while True:
    ok, frame = cap.read()
    if not ok or (MAX_FRAMES and frame_idx >= MAX_FRAMES):
        break
    small = cv2.resize(frame, (WORK_W, out_h))
    annotated = small.copy()

    ti = time.time()

    # --- players: Deep OC-SORT tracking ---
    pres = player_model.track(small, persist=True, tracker=TRACKER,
                              conf=PLAYER_CONF, classes=[0], verbose=False)
    r = pres[0]
    if r.boxes is not None and r.boxes.id is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        ids   = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for (x1,y1,x2,y2), tid, cf in zip(boxes, ids, confs):
            x1,y1,x2,y2 = map(int,(x1,y1,x2,y2))
            cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(annotated,f"{tid}-{cf:.2f}",(x1,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2,cv2.LINE_AA)

    # --- ball: fine-tuned detector, highest-confidence detection ---
    bres = ball_model.predict(small, conf=BALL_CONF, classes=[32], imgsz=BALL_IMGSZ, verbose=False)
    br = bres[0]
    if br.boxes is not None and len(br.boxes) > 0:
        best=None; bc=0
        for box, cf in zip(br.boxes.xyxy.cpu().numpy(), br.boxes.conf.cpu().numpy()):
            if cf>bc: best=box; bc=cf
        if best is not None:
            x1,y1,x2,y2 = map(int, best)
            cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(annotated,f"ball {bc:.2f}",(x1,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2,cv2.LINE_AA)

    per_frame_ms.append((time.time()-ti)*1000)

    cv2.putText(annotated,f"frame {frame_idx}",(20,35),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2,cv2.LINE_AA)
    writer.write(annotated)

    frame_idx += 1
    if frame_idx % 25 == 0:
        avg = sum(per_frame_ms[-25:])/len(per_frame_ms[-25:])
        print(f"  {frame_idx} frames | last-25 avg {avg:.0f} ms")

cap.release(); writer.release()
total = time.time()-t0
print(f"Done {frame_idx} frames in {total:.1f}s ({frame_idx/total:.2f} fps end-to-end). Saved {out_path}")
