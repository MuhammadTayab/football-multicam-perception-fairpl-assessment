# annotator.py — single-class, single-box YOLO annotator with crosshair and keyboard zoom/pan
# Skipped frames are NOT saved. Only frames where you draw a box get a label file.
import cv2, os, glob

# ---------------- CONFIG ----------------
IMG_DIR    = "annotation_frames _selected"
LBL_DIR    = os.path.join(IMG_DIR, "labels")
CLASS_ID   = 0
CLASS_NAME = "ball"
WIN_W, WIN_H = 1600, 900
# ----------------------------------------

os.makedirs(LBL_DIR, exist_ok=True)
images = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
if not images:
    raise SystemExit(f"No images in {IMG_DIR}")

idx = 0
zoom = 1.0
view_x0, view_y0 = 0, 0
box_pts = []
mouse_win = (0, 0)
img = None; H = W = 0

def load(i):
    global img, H, W, zoom, view_x0, view_y0, box_pts
    im = cv2.imread(images[i]); H, W = im.shape[:2]
    zoom, view_x0, view_y0, box_pts = 1.0, 0, 0, []
    base = os.path.splitext(os.path.basename(images[i]))[0]
    lp = os.path.join(LBL_DIR, base + ".txt")
    if os.path.exists(lp):
        with open(lp) as f:
            line = f.readline().split()
        if len(line) == 5:
            _, xc, yc, bw, bh = map(float, line)
            x1 = (xc - bw/2)*W; y1 = (yc - bh/2)*H
            x2 = (xc + bw/2)*W; y2 = (yc + bh/2)*H
            box_pts[:] = [(x1, y1), (x2, y2)]
    return im

def view():
    vw = int(W / zoom); vh = int(H / zoom)
    x0 = max(0, min(view_x0, W - vw)); y0 = max(0, min(view_y0, H - vh))
    crop = img[y0:y0+vh, x0:x0+vw]
    disp = cv2.resize(crop, (WIN_W, WIN_H), interpolation=cv2.INTER_LINEAR)
    return disp, x0, y0, vw, vh

def win_to_full(wx, wy, x0, y0, vw, vh):
    return x0 + wx*(vw/WIN_W), y0 + wy*(vh/WIN_H)

def full_to_win(fx, fy, x0, y0, vw, vh):
    return int((fx-x0)*(WIN_W/vw)), int((fy-y0)*(WIN_H/vh))

def on_mouse(event, x, y, flags, param):
    global box_pts, mouse_win
    mouse_win = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        _, x0, y0, vw, vh = view()
        fx, fy = win_to_full(x, y, x0, y0, vw, vh)
        if len(box_pts) >= 2: box_pts.clear()
        box_pts.append((fx, fy))

def zoom_at(mx, my, factor):
    global zoom, view_x0, view_y0
    _, x0, y0, vw, vh = view()
    fx, fy = win_to_full(mx, my, x0, y0, vw, vh)
    zoom = max(1.0, min(zoom * factor, 30))
    nvw = int(W/zoom); nvh = int(H/zoom)
    view_x0 = int(fx - (mx/WIN_W)*nvw); view_y0 = int(fy - (my/WIN_H)*nvh)

def save_label(i):
    if len(box_pts) != 2:
        return
    base = os.path.splitext(os.path.basename(images[i]))[0]
    lp = os.path.join(LBL_DIR, base + ".txt")
    (x1,y1),(x2,y2) = box_pts
    x1,x2 = sorted((x1,x2)); y1,y2 = sorted((y1,y2))
    xc = ((x1+x2)/2)/W; yc = ((y1+y2)/2)/H
    bw = (x2-x1)/W; bh = (y2-y1)/H
    with open(lp,"w") as f:
        f.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

cv2.namedWindow("annot", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("annot", on_mouse)
img = load(idx)

while True:
    disp, x0, y0, vw, vh = view()

    if len(box_pts) == 2:
        p1 = full_to_win(*box_pts[0], x0,y0,vw,vh)
        p2 = full_to_win(*box_pts[1], x0,y0,vw,vh)
        cv2.rectangle(disp, p1, p2, (0,255,0), 1)
    elif len(box_pts) == 1:
        p1 = full_to_win(*box_pts[0], x0,y0,vw,vh)
        cv2.circle(disp, p1, 3, (0,255,0), -1)

    mx, my = mouse_win
    cv2.line(disp, (mx,0), (mx,WIN_H), (0,255,255), 1)
    cv2.line(disp, (0,my), (WIN_W,my), (0,255,255), 1)

    status = f"{idx+1}/{len(images)}  zoom {zoom:.1f}x  box:{len(box_pts)}/2  {os.path.basename(images[idx])}"
    cv2.putText(disp, status, (15,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow("annot", disp)

    k = cv2.waitKey(20) & 0xFF
    if k == ord('q'):
        save_label(idx); break
    elif k in (ord('d'), 83):                  # next frame
        save_label(idx); idx = min(idx+1, len(images)-1); img = load(idx)
    elif k in (ord('a'), 81):                  # previous frame
        save_label(idx); idx = max(idx-1, 0); img = load(idx)
    elif k in (ord('='), ord('+')):            # zoom in at mouse
        zoom_at(mouse_win[0], mouse_win[1], 1.25)
    elif k == ord('-'):                        # zoom out at mouse
        zoom_at(mouse_win[0], mouse_win[1], 1/1.25)
    elif k == ord('i'):  view_y0 -= int((H/zoom)*0.15)   # pan up
    elif k == ord('k'):  view_y0 += int((H/zoom)*0.15)   # pan down
    elif k == ord('j'):  view_x0 -= int((W/zoom)*0.15)   # pan left
    elif k == ord('l'):  view_x0 += int((W/zoom)*0.15)   # pan right
    elif k == ord('c'):  box_pts.clear()                 # clear box
    elif k == ord('r'):  zoom, view_x0, view_y0 = 1.0, 0, 0   # reset zoom

cv2.destroyAllWindows()
print("Done. Labels in:", LBL_DIR)
