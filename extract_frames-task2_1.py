# extract_frames.py — 500 uniformly-spaced frames from each video, saved at 1920, no duplicates
import cv2, os

# ---------------- CONFIG ----------------
VIDEOS      = ["videos/left.mp4", "videos/right.mp4"]
OUT_DIR     = "annotation_frames"
PER_VIDEO   = 500
DISP_W      = 1920
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

for vpath in VIDEOS:
    if not os.path.exists(vpath):
        print(f"Missing: {vpath}"); continue

    tag = os.path.splitext(os.path.basename(vpath))[0]
    cap = cv2.VideoCapture(vpath)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    disp_h = int(src_h * DISP_W / src_w)

    # uniformly spaced frame indices across the whole video, evenly distributed, unique
    # e.g. if total=13677 and PER_VIDEO=500, step ~27 frames apart, spread start-to-end
    step = total / PER_VIDEO
    indices = sorted(set(int(i * step) for i in range(PER_VIDEO)))
    # guard against edge duplicates near the end
    indices = [i for i in indices if i < total]

    print(f"\n{tag}: {total} frames -> extracting {len(indices)} uniformly-spaced frames at {DISP_W}x{disp_h}")

    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_1920 = cv2.resize(frame, (DISP_W, disp_h))
        out = os.path.join(OUT_DIR, f"{tag}_f{idx:06d}.jpg")
        cv2.imwrite(out, frame_1920)
        saved += 1
        if saved % 50 == 0:
            print(f"  {tag}: {saved}/{len(indices)}")

    cap.release()
    print(f"  {tag}: done, {saved} frames saved")

print(f"\nAll done. Frames in: {OUT_DIR}")
