# task3_topdown.py — project players + ball from both cameras onto one top-down pitch
import cv2, os, numpy as np
from ultralytics import YOLO
import time

# ---------------- CONFIG ----------------
LEFT_VIDEO   = "videos/synced/left_sync_clip.mp4"
RIGHT_VIDEO  = "videos/synced/right_sync_clip.mp4"
PLAYER_MODEL = "yolo11m.pt"
BALL_MODEL   = "ball_finetune/runs/detect/ball_finetune/train-2/weights/best.pt"
TRACKER      = "trackers/deepocsort_tuned.yaml"      # your tuned tracker file
WORK_W       = 1920
MAX_FRAMES   = None                      # None for full video
PLAYER_CONF  = 0.35
BALL_CONF    = 0.25
OVERLAP_LO, OVERLAP_HI = 19.5, 20.5     # center band (m) where both cameras contribute
OUT_DIR      = "FootageResults"
# ---- pitch model (must match homography_*.npy) ----
PITCH_L, PITCH_W = 40.0, 20.0
SCALE = 25                              # px per meter on the minimap
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# load homographies
HL = np.array(np.load("homography/homography_left.npy",  allow_pickle=True).item()["H"], dtype=np.float64)
HR = np.array(np.load("homography/homography_right.npy", allow_pickle=True).item()["H"], dtype=np.float64)

player_model_L = YOLO(PLAYER_MODEL)
player_model_R = YOLO(PLAYER_MODEL)

ball_model   = YOLO(BALL_MODEL)

capL = cv2.VideoCapture(LEFT_VIDEO); capR = cv2.VideoCapture(RIGHT_VIDEO)
srcw = int(capL.get(cv2.CAP_PROP_FRAME_WIDTH)); srch = int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps  = capL.get(cv2.CAP_PROP_FPS)
work_h = int(srch * WORK_W / srcw)


CW, CH = int(PITCH_L*SCALE), int(PITCH_W*SCALE)
cam_out_h = int(work_h * CW / WORK_W)     # both cameras resized to width CW
combined_w = CW
combined_h = CH + cam_out_h + cam_out_h    # minimap + left cam + right cam

writer = cv2.VideoWriter(os.path.join(OUT_DIR,"Topdown MiniMap_Left_Right_combined.mp4"),
                         cv2.VideoWriter_fourcc(*"avc1"), fps, (combined_w, combined_h))
if not writer.isOpened():
    writer = cv2.VideoWriter(os.path.join(OUT_DIR,"topdown_combined.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (combined_w, combined_h))

'''
# minimap canvas size
CW, CH = int(PITCH_L*SCALE), int(PITCH_W*SCALE)
writer = cv2.VideoWriter(os.path.join(OUT_DIR,"topdown_minimap.mp4"),
                         cv2.VideoWriter_fourcc(*"avc1"), fps, (CW, CH))
if not writer.isOpened():
    writer = cv2.VideoWriter(os.path.join(OUT_DIR,"topdown_minimap.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (CW, CH))
'''

def draw_pitch():
    c = np.full((CH, CW, 3), (40,110,40), np.uint8)   # green
    cv2.rectangle(c,(0,0),(CW-1,CH-1),(255,255,255),2)
    cv2.line(c,(int(20*SCALE),0),(int(20*SCALE),CH),(255,255,255),2)          # halfway
    cv2.circle(c,(int(20*SCALE),int((PITCH_W-10)*SCALE)),int(3*SCALE),(255,255,255),2)  # center circle
    return c

def meters_to_px(x, y):
    # world (x,y) meters -> minimap pixels (flip y so y=0 is bottom)
    return int(x*SCALE), int((PITCH_W - y)*SCALE)

def project(H, px, py):
    p = np.array([px, py, 1.0], dtype=np.float64)
    w = H @ p
    return w[0]/w[2], w[1]/w[2]     # world x,y in meters

def players_in_frame(model, frame):
    res = model.track(frame, persist=True, tracker=TRACKER, conf=PLAYER_CONF,
                      classes=[0], verbose=False)
    out = []
    r = res[0]

    annotated = frame.copy()
    
    if r.boxes is not None and r.boxes.id is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        ids   = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), tid, cf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{tid}-{cf:.2f}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
    if r.boxes is not None and r.boxes.id is not None:
        for box in r.boxes.xyxy.cpu().numpy():
            x1,y1,x2,y2 = box
            foot = ((x1+x2)/2, y2)     # bottom-center = feet
            out.append(foot)
    return out,annotated

def ball_in_frame(frame):
    res = ball_model.predict(frame, conf=BALL_CONF, classes=[0], imgsz=960, verbose=False)
    r = res[0]
    (x1,y1,x2,y2) = (0,0,0,0)
    best = None; bc = 0
    if r.boxes is not None and len(r.boxes) > 0:
        for box, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if cf > bc:
                x1,y1,x2,y2 = box; best = ((x1+x2)/2, (y1+y2)/2); bc = cf
    return best,(x1,y1,x2,y2)

t1 = time.time()
frame_idx = 0
while True:
    okL, fL = capL.read(); okR, fR = capR.read()
    if not okL or not okR or (MAX_FRAMES and frame_idx >= MAX_FRAMES):
        break
    fL = cv2.resize(fL, (WORK_W, work_h)); fR = cv2.resize(fR, (WORK_W, work_h))
    
    mini = draw_pitch()

    # players: left camera owns x<=OVERLAP_HI, right owns x>=OVERLAP_LO
    plyr_in_frm, annl = players_in_frame(player_model_L, fL)
    for foot in plyr_in_frm:
        wx, wy = project(HL, *foot)
        #print('Left',wx,wy)
        if wx <= OVERLAP_HI and 0 <= wy <= PITCH_W:
            cv2.circle(mini, meters_to_px(wx,wy), 8, (255,120,0), -1)   # blue-ish = left cam
    plyr_in_frm, annr = players_in_frame(player_model_R, fR)
    for foot in plyr_in_frm:
        wx, wy = project(HR, *foot)
        if wx >= OVERLAP_LO and 0 <= wy <= PITCH_W:
            cv2.circle(mini, meters_to_px(wx,wy), 8, (0,120,255), -1)   # orange = right cam

    '''cv2.imshow('left',annl)
    cv2.imshow('right',annr)
    key = cv2.waitKey(32)
    if key == ord('q'):
        break'''

    
    '''# ball: project from both cameras; disagreement = relative height
    bL = ball_in_frame(fL); bR = ball_in_frame(fR)
    if bL is not None:
        wxL, wyL = project(HL, *bL)
        cv2.circle(mini, meters_to_px(wxL,wyL), 6, (255,255,0), 2)      # cyan ring = left proj
    if bR is not None:
        wxR, wyR = project(HR, *bR)
        cv2.circle(mini, meters_to_px(wxR,wyR), 6, (0,255,255), 2)      # yellow ring = right proj
    if bL is not None and bR is not None:
        disagree = ((wxL-wxR)**2 + (wyL-wyR)**2) ** 0.5   # meters of ground-projection separation
        midx, midy = (wxL+wxR)/2, (wyL+wyR)/2
        p = meters_to_px(midx, midy)
        cv2.putText(mini, f"ball dh~{disagree:.1f}m", (p[0]+10, p[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)'''


    # ball: ground position from the owning camera; height only in central overlap
    bL,bbox = ball_in_frame(fL)
    p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(annl, p1, p2, (0, 0, 255), 2)
    cv2.putText(annl, "ball", (int(bbox[0]), int(bbox[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    #print('ball',bbox)

    bR,bbox = ball_in_frame(fR)
    p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(annr, p1, p2, (0, 0, 255), 2)
    cv2.putText(annr, "ball", (int(bbox[0]), int(bbox[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    
    #print('ball R',bbox)

    
    
    wxL = wyL = wxR = wyR = None
    if bL is not None: wxL, wyL = project(HL, *bL)
    if bR is not None: wxR, wyR = project(HR, *bR)

    # choose ground position by which half the ball is in (like players)
    ball_pos = None
    if wxL is not None and wxL <= OVERLAP_HI and 0 <= wyL <= PITCH_W:
        ball_pos = (wxL, wyL)
    if wxR is not None and wxR >= OVERLAP_LO and 0 <= wyR <= PITCH_W:
        # if both valid in overlap, prefer the one nearer its own camera side later; for now take right
        ball_pos = (wxR, wyR) if ball_pos is None else ((ball_pos[0]+wxR)/2,(ball_pos[1]+wyR)/2)

    if ball_pos is not None:
        cv2.circle(mini, meters_to_px(*ball_pos), 7, (0,0,255), -1)   # red dot = ball ground pos

    # height ONLY when ball is in the central overlap band in BOTH cameras
    if (wxL is not None and wxR is not None
            and OVERLAP_LO <= wxL <= OVERLAP_HI and OVERLAP_LO <= wxR <= OVERLAP_HI):
        disagree = ((wxL-wxR)**2 + (wyL-wyR)**2) ** 0.5
        midx, midy = (wxL+wxR)/2, (wyL+wyR)/2
        pp = meters_to_px(midx, midy)
        cv2.putText(mini, f"height~{disagree:.1f}", (pp[0]+10, pp[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

    cv2.putText(mini, f"frame {frame_idx}   blue=left cam  orange=right cam  red=ball",
                (15,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    '''writer.write(mini)'''

    # resize the two camera frames to the minimap width, preserving aspect
    left_resized  = cv2.resize(annl, (CW, cam_out_h))
    right_resized = cv2.resize(annr, (CW, cam_out_h))

    # optional labels so the stack is self-explanatory
    cv2.putText(mini,          "TOP-DOWN MINIMAP", (15, CH-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(left_resized,  "LEFT CAMERA",  (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(right_resized, "RIGHT CAMERA", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    # vertical stack: minimap on top, left cam, right cam
    combined = cv2.vconcat([mini, left_resized, right_resized])

    '''cv2.imshow('combined',combined)
    key = cv2.waitKey(32)
    if key == ord('q'):
        break'''
    writer.write(combined)

    
    frame_idx += 1
    if frame_idx % 25 == 0:
        print(f"  {frame_idx} frames | avg time per frame: {int(1000*(time.time()-t1)/25)} ms")
        t1 =time.time()

capL.release(); capR.release(); writer.release()
print(f"Done {frame_idx} frames. Saved {os.path.join(OUT_DIR,'topdown_minimap.mp4')}")



