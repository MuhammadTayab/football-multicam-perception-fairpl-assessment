# task2_ball_trajectory.py
# Ball trajectory: fine-tuned detection -> confidence+prediction gating -> Kalman coast
#                   -> interpolate -> quick detection-anchored smoothing. Single ball.
import cv2, os, time, csv, numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
VIDEO      = "videos/synced/left_sync_clip.mp4"
# --- fine-tuned model ---
MODEL      = 'ball_finetune/runs/detect/ball_finetune/train-2/weights/best.pt'
BALL_CLASS = [0]          # fine-tuned model is single-class ball -> class 0
IMGSZ      = 960          # detection at 960.
CONF       = 0.25         # fine-tuned model detection confidence
# (for the generic model instead: MODEL="yolo11x.pt", BALL_CLASS=[32], IMGSZ=960, CONF=0.25)
MAX_FRAMES = None
INFER_W    = 1920
OUT_DIR    = "FootageResults"
# gating / filtering priors
MAX_AREA_FRAC   = 0.0008
MIN_ASPECT, MAX_ASPECT = 0.5, 2.0
GATE_BASE_PX    = 80
GATE_GROWTH_PX  = 40
MAX_COAST       = 25
# confidence-aware gating: distance penalty per pixel vs confidence reward
DIST_WEIGHT     = 1.0     # cost per pixel of distance from prediction
CONF_WEIGHT     = 300.0   # reward for confidence (higher = trust confidence more over proximity)
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
tag = os.path.splitext(os.path.basename(VIDEO))[0]
out_path = os.path.join(OUT_DIR, f"{tag}_finetuned_ball_trajectory_smooth_task2.mp4")

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)
src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps   = cap.get(cv2.CAP_PROP_FPS)
out_w, out_h = (src_w, src_h) if INFER_W is None else (INFER_W, int(src_h*INFER_W/src_w))
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (out_w, out_h))
if not writer.isOpened():
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

def make_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 5.0
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0
    kf.errorCovPost        = np.eye(4, dtype=np.float32) * 100.0
    return kf

kf = make_kalman()
initialized = False
misses = 0
pred_pt = None
trajectory = []
per_frame_ms = []

def detect_ball(frame, frame_area):
    res = model.predict(frame, conf=CONF, classes=BALL_CLASS, imgsz=IMGSZ, verbose=False)
    out = []
    r = res[0]
    if r.boxes is not None and len(r.boxes) > 0:
        for box, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x1,y1,x2,y2 = box
            w,h = x2-x1, y2-y1
            if w*h > MAX_AREA_FRAC*frame_area:
                print('continued on size')
                continue
            asp = w/h if h>0 else 0
            if not (MIN_ASPECT <= asp <= MAX_ASPECT):
                print('continued on AspRat')
                continue
            out.append(((x1+x2)/2,(y1+y2)/2,x1,y1,x2,y2,float(cf)))
    return out

frame_idx = 0
t0 = time.time()
while True:
    ok, frame = cap.read()
    if not ok or (MAX_FRAMES and frame_idx >= MAX_FRAMES): break
    small = frame if INFER_W is None else cv2.resize(frame,(out_w,out_h))
    frame_area = out_w*out_h
    annotated = small.copy()

    if initialized:
        pred = kf.predict()
        pred_pt = (float(pred[0][0]), float(pred[1][0]))

    _t = time.time()
    cands = detect_ball(small, frame_area)
    #print(type(cands),cands)
    per_frame_ms.append((time.time()-_t)*1000)

    # choose candidate: highest-conf when seeding; confidence-aware gating once tracking
    chosen = None
    if cands:
        if not initialized:
            chosen = max(cands, key=lambda c: c[6])
        else:
            gate = GATE_BASE_PX + GATE_GROWTH_PX*misses
            best, best_score = None, -1e9
            for c in cands:
                dx,dy = c[0]-pred_pt[0], c[1]-pred_pt[1]
                dist = (dx*dx+dy*dy)**0.5
                if dist > gate:  # outside search gate -> reject as distractor
                    continue
                score = CONF_WEIGHT*c[6] - DIST_WEIGHT*dist   # near AND confident wins
                if score > best_score:
                    best, best_score = c, score
            chosen = best

    if chosen is not None:
        cx,cy = chosen[0], chosen[1]
        meas = np.array([[np.float32(cx)],[np.float32(cy)]])
        if not initialized:
            kf.statePost = np.array([[cx],[cy],[0],[0]], np.float32); initialized = True
        else:
            kf.correct(meas)
        misses = 0
        trajectory.append((frame_idx, cx, cy, "det"))
        x1,y1,x2,y2 = map(int, chosen[2:6])
        cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(annotated,f"ball {chosen[6]:.2f}",(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2,cv2.LINE_AA)
    else:
        if initialized and misses < MAX_COAST:
            misses += 1
            cx,cy = pred_pt
            trajectory.append((frame_idx, cx, cy, "pred"))
            cv2.circle(annotated,(int(cx),int(cy)),8,(0,165,255),2)
            cv2.putText(annotated,"pred",(int(cx)+10,int(cy)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,165,255),2,cv2.LINE_AA)
        else:
            trajectory.append((frame_idx, None, None, "lost"))
            if misses >= MAX_COAST:
                initialized = False; kf = make_kalman()
            cv2.putText(annotated,"ball lost",(20,80),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2,cv2.LINE_AA)

    tail = [t for t in trajectory[-40:] if t[1] is not None]
    for i in range(1,len(tail)):
        cv2.line(annotated,(int(tail[i-1][1]),int(tail[i-1][2])),(int(tail[i][1]),int(tail[i][2])),(255,0,0),2)

    cv2.putText(annotated,f"frame {frame_idx}  misses:{misses}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2,cv2.LINE_AA)
    writer.write(annotated)
    frame_idx += 1
    if frame_idx % 10 == 0:
        avg = sum(per_frame_ms[-10:])/len(per_frame_ms[-10:])
        print(f"  {frame_idx} frames | last-10 avg inference {avg:.0f} ms")

cap.release(); writer.release()

# ---- POST: interpolate short gaps, then quick detection-anchored smoothing ----
def build_series(traj):
    d = {}
    for (f,x,y,s) in traj:
        if x is not None: d[f] = [x,y,s]
    return d

series = build_series(trajectory)
known = sorted(series.keys())

# 1) interpolate short gaps between real detections (replaces wrong coasts with straight anchor line)
det_frames = sorted(f for f in series if series[f][2]=="det")
for a,b in zip(det_frames, det_frames[1:]):
    gap = b-a
    if 1 < gap <= MAX_COAST:
        (xa,ya),(xb,yb) = series[a][:2], series[b][:2]
        for g in range(1,gap):
            t = g/gap
            series[a+g] = [xa+(xb-xa)*t, ya+(yb-ya)*t, "interp"]

# 2) quick smoothing: moving-average over the position series to remove sharp corners
sm_frames = sorted(series.keys())
xs = np.array([series[f][0] for f in sm_frames])
ys = np.array([series[f][1] for f in sm_frames])
def moving_avg(a, k=5):
    if len(a) < k: return a
    kernel = np.ones(k)/k
    return np.convolve(a, kernel, mode="same")
xs_s, ys_s = moving_avg(xs), moving_avg(ys)
smoothed = {f:(float(xs_s[i]), float(ys_s[i])) for i,f in enumerate(sm_frames)}

det_n  = sum(1 for (f,x,y,s) in trajectory if s=="det")
pred_n = sum(1 for (f,x,y,s) in trajectory if s=="pred")
lost_n = sum(1 for (f,x,y,s) in trajectory if s=="lost")
print(f"Done {frame_idx} frames in {time.time()-t0:.1f}s")
print(f"  detected:{det_n}  coasted:{pred_n}  lost:{lost_n}  final trajectory points:{len(smoothed)}")
print(f"Saved {out_path}")

with open(os.path.join(OUT_DIR, f"{tag}_ball_trajectory_task2.csv"),"w",newline="") as f:
    w = csv.writer(f); w.writerow(["frame","x","y"])
    for fr in sorted(smoothed): w.writerow([fr, f"{smoothed[fr][0]:.1f}", f"{smoothed[fr][1]:.1f}"])
