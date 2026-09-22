# homography_picker.py — click pitch points in order, compute homography, preview top-down
import cv2, os, numpy as np

# ---------------- CONFIG ----------------
VIDEO   = "videos/synced/right_sync.mp4"
FRAME   = 0
WORK_W  = 1920
CAM_TAG = "right"
# ----------------------------------------


PITCH_L, PITCH_W = 40.0, 20.0

# points IN CLICK ORDER with exact pitch coords (meters). LEFT camera.
POINTS = [
    ("1: PITCH TOP-LEFT corner",              (0.0,  20.0)),
    ("2: LEFT goal TOP post base",            (0.0,  11.5)),
    ("3: LEFT goal BOTTOM post base",         (0.0,   8.5)),
    ("4: LEFT box TOP-RIGHT corner (to center)",(6.0, 15.0)),
    ("5: CENTER circle TOP point",            (20.0, 13.0)),
    ("6: CENTER circle CENTER point",         (20.0, 10.0)),
    ("7: CENTER circle BOTTOM point",         (20.0,  7.0)),
    ("8: HALFWAY line TOP end",               (20.0, 20.0)),
    
]
'''0
#Right
POINTS = [
    ("1: PITCH TOP-RIGHT corner",              (40.0, 20.0)),
    ("2: RIGHT goal TOP post base",            (40.0, 11.5)),
    ("3: RIGHT goal BOTTOM post base",         (40.0,  8.5)),
    ("4: RIGHT box TOP-LEFT corner (to center)",(34.0, 15.0)),
    ("5: CENTER circle TOP point",             (20.0, 13.0)),
    ("6: CENTER circle CENTER point",          (20.0, 10.0)),
    ("7: CENTER circle BOTTOM point",          (20.0,  7.0)),
    ("8: HALFWAY line TOP end",                (20.0, 20.0)),
]

'''

N = len(POINTS)

cap = cv2.VideoCapture(VIDEO)
src_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); src_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
work_h=int(src_h*WORK_W/src_w)
cap.set(cv2.CAP_PROP_POS_FRAMES,FRAME); ok,frame=cap.read(); cap.release()
if not ok: raise SystemExit("Could not read frame")
img=cv2.resize(frame,(WORK_W,work_h)); H_IMG,W_IMG=img.shape[:2]

clicked=[]; skipped=[]; zoom=1.0; vx0=vy0=0; mouse_win=(0,0)
WIN_W,WIN_H=1600,int(1600*H_IMG/W_IMG)

def view():
    vw=int(W_IMG/zoom); vh=int(H_IMG/zoom)
    x0=max(0,min(vx0,W_IMG-vw)); y0=max(0,min(vy0,H_IMG-vh))
    return cv2.resize(img[y0:y0+vh,x0:x0+vw],(WIN_W,WIN_H)),x0,y0,vw,vh
def win_to_full(wx,wy,x0,y0,vw,vh): return x0+wx*(vw/WIN_W), y0+wy*(vh/WIN_H)
def full_to_win(fx,fy,x0,y0,vw,vh): return int((fx-x0)*(WIN_W/vw)), int((fy-y0)*(WIN_H/vh))

def cur_index():
    return len(clicked) + len(skipped)

def on_mouse(event, x, y, flags, param):
    global mouse_win
    mouse_win = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN and cur_index() < N:
        _, x0, y0, vw, vh = view()
        fx, fy = win_to_full(x, y, x0, y0, vw, vh)
        this_index = cur_index()            # index BEFORE appending  (0-based)
        clicked.append((this_index, (fx, fy)))
        print(f"  clicked -> {POINTS[this_index][0]} at ({fx:.0f},{fy:.0f})")
        
def zoom_at(mx,my,f):
    global zoom,vx0,vy0
    _,x0,y0,vw,vh=view(); fx,fy=win_to_full(mx,my,x0,y0,vw,vh)
    zoom=max(1.0,min(zoom*f,30)); nvw=int(W_IMG/zoom); nvh=int(H_IMG/zoom)
    vx0=int(fx-(mx/WIN_W)*nvw); vy0=int(fy-(my/WIN_H)*nvh)

cv2.namedWindow("pick",cv2.WINDOW_AUTOSIZE); cv2.setMouseCallback("pick",on_mouse)
print("Click points IN ORDER. Keys: +/- zoom, i/j/k/l pan, u=undo, s=skip point, r=reset, Enter=compute, q=quit")
for lbl,_ in POINTS: print("   ",lbl)

while True:
    disp,x0,y0,vw,vh=view()
    for idx,(fx,fy) in clicked:
        p=full_to_win(fx,fy,x0,y0,vw,vh)
        cv2.circle(disp,p,5,(0,0,255),-1)
        cv2.putText(disp,str(idx),(p[0]+6,p[1]-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2,cv2.LINE_AA)
    mx,my=mouse_win
    cv2.line(disp,(mx,0),(mx,WIN_H),(0,255,255),1); cv2.line(disp,(0,my),(WIN_W,my),(0,255,255),1)
    ci=cur_index()
    nxt = POINTS[ci][0] if ci<N else "DONE - press Enter"
    cv2.putText(disp,f"NEXT -> {nxt}",(15,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(disp,f"zoom {zoom:.1f}x  placed {len(clicked)}  skipped {len(skipped)}",(15,55),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow("pick",disp)

    k=cv2.waitKey(20)&0xFF
    if k==ord('q'): break
    elif k in (ord('='),ord('+')): zoom_at(*mouse_win,1.25)
    elif k==ord('-'): zoom_at(*mouse_win,1/1.25)
    elif k==ord('i'): vy0-=int((H_IMG/zoom)*0.15)
    elif k==ord('k'): vy0+=int((H_IMG/zoom)*0.15)
    elif k==ord('j'): vx0-=int((W_IMG/zoom)*0.15)
    elif k==ord('l'): vx0+=int((W_IMG/zoom)*0.15)
    elif k==ord('s') and ci<N: skipped.append(ci); print(f"  SKIPPED {POINTS[ci][0]}")
    elif k==ord('u'):
        if clicked and (not skipped or clicked[-1][0] > skipped[-1]): clicked.pop(); print("  undo click")
        elif skipped: skipped.pop(); print("  undo skip")
    elif k==ord('r'): zoom,vx0,vy0=1.0,0,0
    elif k in (13,10):
        if len(clicked)<4:
            print(f"  need at least 4 points, have {len(clicked)}"); continue
        src = np.array([p for _, p in clicked], dtype=np.float64)
        dst = np.array([POINTS[i][1] for i, _ in clicked], dtype=np.float64)  # i is 0-based now
        print(src,dst)
        Hmat,mask=cv2.findHomography(src,dst,0)
        print("\nHomography:\n",Hmat,"\ninliers:",int(mask.sum()),"/",len(clicked))
        np.save(f"homography_{CAM_TAG}.npy",{"H":Hmat.tolist(),
                "img_pts":[list(p) for _,p in clicked],
                "world_pts":[POINTS[i][1] for i,_ in clicked],   # i is 0-based
                "work_w":WORK_W,"work_h":work_h,"pitch_l":PITCH_L,"pitch_w":PITCH_W})
        print(f"SAVED homography_{CAM_TAG}.npy")
        SCALE=25; cw=int(PITCH_L*SCALE); ch=int(PITCH_W*SCALE)
        M=np.array([[SCALE,0,0],[0,-SCALE,ch],[0,0,1]],dtype=np.float64)
        topdown=cv2.warpPerspective(img,M@Hmat,(cw,ch))
        # draw the pitch outline + center circle on the preview for sanity
        cv2.rectangle(topdown,(0,0),(cw-1,ch-1),(255,255,255),2)
        cv2.circle(topdown,(int(20*SCALE),int((PITCH_W-10)*SCALE)),int(3*SCALE),(0,255,0),2)
        cv2.imshow("topdown",topdown); cv2.imwrite(f"topdown_{CAM_TAG}.jpg",topdown)
        print(f"Saved topdown_{CAM_TAG}.jpg")

cv2.destroyAllWindows()
