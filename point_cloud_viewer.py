import cv2
import numpy as np
from pupil_apriltags import Detector
import time

# ====================
# 参数配置区域
# ====================
CAMERA_INDEX       = 0           # camera index for cv2.VideoCapture
WINDOW_WIDTH       = 1920        # cv window width
WINDOW_HEIGHT      = 1080         # cv window height

# apriltag board settings
TAG_FAMILY         = 'tag25h9'
TAG_SIZE           = 0.105        # meters
TAG_SPACING        = 0.117        # meters between tag centers
ROWS, COLS         = 2, 2
BOARD_IDS          = [0, 1, 2, 3]

# calibration capture settings
TARGET_FRAMES      = 10
MIN_TAGS_CAPTURE   = 1

# live display settings
AXIS_LENGTH        = 0.10        # meters
NORM_LINE_LENGTH   = 0.20        # meters (length of normal vector)
PANEL_WIDTH        = 300         # pixels, increased to fit full Z text
GUIDE_RATIO        = 0.25        # 1/4 margin for guide rectangle

# ====================
# utility functions
# ====================
def draw_progress(frame, count, target):
    h, w = frame.shape[:2]
    bar_w = int(w * (count / target))
    cv2.rectangle(frame, (0, h - 20), (bar_w, h), (0, 200, 0), -1)

def draw_text_panel(frame, lines):
    h, w = frame.shape[:2]
    x0, y0 = w - PANEL_WIDTH + 10, 10
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (x0, y0 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# compute board-centers in meter
centers = [(c * TAG_SPACING, r * TAG_SPACING)
           for r in range(ROWS) for c in range(COLS)]

# --------------------
# Phase 1: Camera Calibration
# --------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
det = Detector(families=TAG_FAMILY)
objpoints, imgpoints = [], []
frame_count = 0

print("Phase 1: Camera Calibration — press [c] to capture, [q] to finish")

cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Calibration", WINDOW_WIDTH, WINDOW_HEIGHT)

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = det.detect(gray)

    # draw detected tags
    for d in dets:
        pts = np.int32(d.corners).reshape(-1,2)
        cv2.polylines(frame, [pts], True, (0,255,0), 2)
        cv2.putText(frame, f"ID:{d.tag_id}", tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # translucent guide box
    h, w = frame.shape[:2]
    guide = frame.copy()
    cv2.rectangle(guide,
                  (int(w*GUIDE_RATIO), int(h*GUIDE_RATIO)),
                  (int(w*(1-GUIDE_RATIO)), int(h*(1-GUIDE_RATIO))),
                  (255,255,255), 2)
    frame = cv2.addWeighted(guide, 0.3, frame, 0.7, 0)

    # UI text (English)
    cv2.putText(frame,
                f"Phase 1: Calibrating ({frame_count}/{TARGET_FRAMES})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,200), 2)
    cv2.putText(frame,
                f"Press [c] to capture (>= {MIN_TAGS_CAPTURE} tags), [q] to finish",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    draw_progress(frame, frame_count, TARGET_FRAMES)

    cv2.imshow("Calibration", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and len(dets) >= MIN_TAGS_CAPTURE:
        obj_pts, img_pts = [], []
        for d in dets:
            if d.tag_id not in BOARD_IDS: continue
            idx = BOARD_IDS.index(d.tag_id)
            cx, cy = centers[idx]
            half = TAG_SIZE/2
            obj_pts.append(np.array([
                [cx-half, cy+half, 0],
                [cx+half, cy+half, 0],
                [cx+half, cy-half, 0],
                [cx-half, cy-half, 0]
            ], dtype=np.float32))
            img_pts.append(np.array(d.corners, dtype=np.float32))
        objpoints += obj_pts
        imgpoints += img_pts
        frame_count += 1
    elif key == ord('q') or frame_count >= TARGET_FRAMES:
        break

cap.release()
cv2.destroyWindow("Calibration")

if frame_count < 3:
    print("Not enough frames (need >=3). Calibration failed.")
    exit(1)

# perform calibration
ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)
print(f"Calibration done. Reprojection error: {ret:.4f}")
print("Camera matrix K:\n", K)
print("Distortion coeffs D:\n", D.ravel())

# --------------------
# Phase 2: Live Pose Estimation
# --------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
normal_vec = np.array([0, 0, 1], dtype=np.float32) * NORM_LINE_LENGTH
prev_time = time.time()

print("Phase 2: Live Pose Estimation — press [q] to exit")

cv2.namedWindow("Live Pose", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Pose", WINDOW_WIDTH, WINDOW_HEIGHT)

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = det.detect(gray)

    poses = []
    for d in dets:
        if d.tag_id not in BOARD_IDS: continue
        idx = BOARD_IDS.index(d.tag_id)
        cx, cy = centers[idx]
        half = TAG_SIZE / 2
        obj_pts = np.array([
            [cx-half, cy+half, 0],
            [cx+half, cy+half, 0],
            [cx+half, cy-half, 0],
            [cx-half, cy-half, 0]
        ], dtype=np.float32)
        img_pts = np.array(d.corners, dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: continue

        # Calculate the start and end points for the normal vector (at tag center)
        R, _ = cv2.Rodrigues(rvec)
        start_pt_tag = np.array([cx, cy, 0], dtype=np.float32)  # Center of the tag in world coordinates
        end_pt_tag = start_pt_tag - (R @ normal_vec)  # Transform normal vector from tag frame to world frame
        pts_norm, _ = cv2.projectPoints(
            np.vstack([start_pt_tag, end_pt_tag]),
            rvec, tvec, K, D
        )

        # Check if normal vector projection is valid
        if np.any(np.isnan(pts_norm)):
            continue  # Skip invalid normal projection

        p0 = tuple(pts_norm[0].ravel().astype(int))
        p1 = tuple(pts_norm[1].ravel().astype(int))

        # Draw normal vector with a more noticeable appearance
        cv2.line(frame, p0, p1, (0, 255, 255), 3)

        # Draw concentric circles at the end of the normal vector to create a 3D effect
        radius_increment = 5  # Each circle will grow by this amount
        num_circles = 3  # Number of concentric circles to draw
        for i in range(1, num_circles + 1):
            radius = i * radius_increment
            cv2.circle(frame, p1, radius, (0, 255, 255), 2)  # Yellow circles

        poses.append((tvec.ravel(), rvec, d.tag_id))

        # Mark tag center
        center_pt = tuple(np.mean(d.corners, axis=0).astype(int))
        cv2.circle(frame, center_pt, 5, (255, 255, 0), -1)
        cv2.putText(frame, f"ID:{d.tag_id}", (center_pt[0]+5, center_pt[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Compute FPS
    now = time.time()
    fps = 1.0 / (now - prev_time)
    prev_time = now

    import math

    # Draw info panel with each value on a separate line
    panel = []
    if poses:
        tvec, rvec, tid = poses[0]
        # —— 正确：先 Rodrigues -> R，再从 R 提取 ZYX 欧拉角 —— 
        R, _ = cv2.Rodrigues(rvec)
        # sy = √(R00² + R10²)，用来判断是否接近奇异
        sy = math.hypot(R[0,0], R[1,0])
        if sy < 1e-6:
            # 奇异情形：pitch 约 ±90°
            pitch  = math.degrees(math.atan2(-R[1,2],  R[1,1]))
            yaw = math.degrees(math.atan2(-R[2,0], sy))
            roll   = 0.0
        else:
            pitch  = math.degrees(math.atan2( R[2,1],  R[2,2]))
            yaw = math.degrees(math.atan2(-R[2,0],  sy))
            roll   = math.degrees(math.atan2( R[1,0],  R[0,0]))

        panel += [
            f"Tag ID: {tid}",
            f"X={tvec[0]:.2f}m",
            f"Y={tvec[1]:.2f}m",
            f"Z={tvec[2]:.2f}m",
            f"Yaw={yaw:.1f} degree",
            f"Pitch={pitch:.1f} degree",
            f"Roll={roll:.1f} degree"
        ]

    # Display the panel
    draw_text_panel(frame, panel)

    cv2.imshow("Live Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
