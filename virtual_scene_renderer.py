import cv2
import numpy as np
from pupil_apriltags import Detector
import time

# ====================
# 参数配置区域
# ====================
CAMERA_INDEX       = 0           # camera index for cv2.VideoCapture
WINDOW_WIDTH       = 1280        # cv window width
WINDOW_HEIGHT      = 720         # cv window height

# apriltag board settings
TAG_FAMILY         = 'tag36h11'
TAG_SIZE           = 0.10        # meters
TAG_SPACING        = 0.15        # meters between tag centers
ROWS, COLS         = 1, 1
BOARD_IDS          = [0]

# calibration capture settings
TARGET_FRAMES      = 10
MIN_TAGS_CAPTURE   = 1

# live display settings
AXIS_LENGTH        = 0.02        # meters
NORM_LINE_LENGTH   = 0.20        # meters (length of normal vector)
PANEL_WIDTH        = 300         # pixels, increased to fit full Z text
GUIDE_RATIO        = 0.25        # 1/4 margin for guide rectangle

# 3D visualization settings
SCENE_WIDTH        = 800         # 3D场景窗口宽度
SCENE_HEIGHT       = 600         # 3D场景窗口高度
CUBE_SIZE          = 0.1         # 场景中立方体大小 (meters)
GROUND_SIZE        = 2.0         # 地面大小 (meters)
FOV                = 60          # 视角 (度)

# ====================
# 全局变量
# ====================
camera_matrix = None
dist_coeffs = None
tag_pose = None  # 存储最新的tag位姿

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

def project_points_3d(points_3d, camera_matrix, rvec, tvec):
    """将3D点投影到2D平面"""
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, None)
    return points_2d.reshape(-1, 2).astype(int)

def create_3d_scene_camera():
    """创建更精确的透视相机矩阵"""
    focal_length = SCENE_WIDTH / (2 * np.tan(np.radians(FOV/2)))
    center = (SCENE_WIDTH//2, SCENE_HEIGHT//2)
    return np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

def draw_3d_scene(frame, rvec, tvec, scene_camera_matrix):
    """优化后的3D场景绘制"""
    scene = np.zeros((SCENE_HEIGHT, SCENE_WIDTH, 3), dtype=np.uint8)
    
    # 绘制渐变背景
    cv2.rectangle(scene, (0, 0), (SCENE_WIDTH, SCENE_HEIGHT//2), (80, 120, 200), -1)  # 天空
    cv2.rectangle(scene, (0, SCENE_HEIGHT//2), (SCENE_WIDTH, SCENE_HEIGHT), (40, 80, 40), -1)  # 地面
    
    # 绘制地面网格（增强深度感）
    grid_step = GROUND_SIZE / 10
    for i in range(-10, 11):
        x = i * grid_step
        z = -GROUND_SIZE/2
        color = (180, 180, 180) if i % 2 == 0 else (160, 160, 160)
        start = project_points_3d(np.array([[x, 0, z]]), scene_camera_matrix, rvec, tvec)[0]
        end = project_points_3d(np.array([[x, 0, GROUND_SIZE/2]]), scene_camera_matrix, rvec, tvec)[0]
        cv2.line(scene, tuple(start), tuple(end), color, 1)
        start = project_points_3d(np.array([[z, 0, x]]), scene_camera_matrix, rvec, tvec)[0]
        end = project_points_3d(np.array([[GROUND_SIZE/2, 0, x]]), scene_camera_matrix, rvec, tvec)[0]
        cv2.line(scene, tuple(start), tuple(end), color, 1)
    
    # 定义更丰富的3D物体（金字塔+立方体群）
    # 金字塔（蓝色） - 放置在原点附近
    pyramid_points = np.float32([
        [-CUBE_SIZE/3, -CUBE_SIZE/3, 0],
        [CUBE_SIZE/3, -CUBE_SIZE/3, 0],
        [CUBE_SIZE/3, CUBE_SIZE/3, 0],
        [-CUBE_SIZE/3, CUBE_SIZE/3, 0],
        [0, 0, CUBE_SIZE/2]  # 顶点
    ])
    
    # 立方体（红色） - 放置在金字塔外侧，远离原点
    cube_offset = np.float32([0.1, 0.1, -0.5])  # 调整立方体位置，远离原点
    cube_points = np.float32([
        [-CUBE_SIZE/2, -CUBE_SIZE/2, -CUBE_SIZE/2],
        [CUBE_SIZE/2, -CUBE_SIZE/2, -CUBE_SIZE/2],
        [CUBE_SIZE/2, CUBE_SIZE/2, -CUBE_SIZE/2],
        [-CUBE_SIZE/2, CUBE_SIZE/2, -CUBE_SIZE/2],
        [-CUBE_SIZE/2, -CUBE_SIZE/2, CUBE_SIZE/2],
        [CUBE_SIZE/2, -CUBE_SIZE/2, CUBE_SIZE/2],
        [CUBE_SIZE/2, CUBE_SIZE/2, CUBE_SIZE/2],
        [-CUBE_SIZE/2, CUBE_SIZE/2, CUBE_SIZE/2]
    ])
    cube_points += cube_offset  # 应用位置偏移
    
    # 投影物体
    cube_projected = project_points_3d(cube_points, scene_camera_matrix, rvec, tvec)
    pyramid_projected = project_points_3d(pyramid_points, scene_camera_matrix, rvec, tvec)
    
    # 绘制立方体（带深度排序）
    cube_faces = [
        [0, 1, 2, 3], [4, 5, 6, 7],  # 前后
        [0, 1, 5, 4], [2, 3, 7, 6],  # 上下
        [1, 2, 6, 5], [0, 3, 7, 4]   # 左右
    ]
    for face in cube_faces:
        points = cube_projected[face].astype(np.int32)
        cv2.polylines(scene, [points], True, (0, 0, 255), 2)
    
    # 绘制金字塔
    for i in range(4):
        cv2.line(scene, tuple(pyramid_projected[i].astype(np.int32)), 
                 tuple(pyramid_projected[4].astype(np.int32)), (255, 0, 0), 2)
        cv2.line(scene, tuple(pyramid_projected[i].astype(np.int32)), 
                 tuple(pyramid_projected[(i+1)%4].astype(np.int32)), (255, 0, 0), 2)
    
    # 绘制坐标轴（更大更清晰）
    axis_points = np.float32([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]) * AXIS_LENGTH * 2
    axis_projected = project_points_3d(axis_points, scene_camera_matrix, rvec, tvec)
    cv2.arrowedLine(scene, tuple(axis_projected[0].astype(np.int32)), 
                    tuple(axis_projected[1].astype(np.int32)), (0,0,255), 3, tipLength=0.02)  # X轴
    cv2.arrowedLine(scene, tuple(axis_projected[0].astype(np.int32)), 
                    tuple(axis_projected[2].astype(np.int32)), (0,255,0), 3, tipLength=0.02)  # Y轴
    cv2.arrowedLine(scene, tuple(axis_projected[0].astype(np.int32)), 
                    tuple(axis_projected[3].astype(np.int32)), (255,0,0), 3, tipLength=0.02)  # Z轴
    
    return scene

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
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)
print(f"Calibration done. Reprojection error: {ret:.4f}")
print("Camera matrix K:\n", camera_matrix)
print("Distortion coeffs D:\n", dist_coeffs.ravel())

# --------------------
# Phase 2: Live Pose Estimation with Virtual Camera
# --------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
axis = np.float32([[AXIS_LENGTH,0,0],
                   [0,AXIS_LENGTH,0],
                   [0,0,AXIS_LENGTH]])
prev_time = time.time()

# 创建3D场景相机矩阵
scene_camera_matrix = create_3d_scene_camera()

print("Phase 2: Live Pose Estimation with Virtual Camera — press [q] to exit")

cv2.namedWindow("Live Pose", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Pose", WINDOW_WIDTH, WINDOW_HEIGHT)

cv2.namedWindow("3D Scene", cv2.WINDOW_NORMAL)
cv2.resizeWindow("3D Scene", SCENE_WIDTH, SCENE_HEIGHT)

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = det.detect(gray)

    current_pose = None
    
    for d in dets:
        if d.tag_id not in BOARD_IDS: continue
        idx = BOARD_IDS.index(d.tag_id)
        cx, cy = centers[idx]
        half = TAG_SIZE/2
        obj_pts = np.array([
            [cx-half, cy+half, 0],
            [cx+half, cy+half, 0],
            [cx+half, cy-half, 0],
            [cx-half, cy-half, 0]
        ], dtype=np.float32)
        img_pts = np.array(d.corners, dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: continue

        # Draw coordinate axes if solvePnP was successful
        imgpts_axis, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

        # Check if projected points are valid
        if np.any(np.isnan(imgpts_axis)):
            continue  # Skip invalid projections

        corner = tuple(img_pts[0].astype(int))
        
        # Draw lines only if the projected points are valid
        imgpts_axis = imgpts_axis.reshape(-1, 2)
        if not np.any(np.isnan(imgpts_axis)): 
            cv2.line(frame, corner, tuple(imgpts_axis[0].ravel().astype(int)), (0,0,255), 2)  # X轴 - 红色
            cv2.line(frame, corner, tuple(imgpts_axis[1].ravel().astype(int)), (0,255,0), 2)  # Y轴 - 绿色
            cv2.line(frame, corner, tuple(imgpts_axis[2].ravel().astype(int)), (255,0,0), 2)  # Z轴 - 蓝色

        # Compute Euler angles
        R, _ = cv2.Rodrigues(rvec)
        sy = np.hypot(R[0,0], R[1,0])
        yaw   = np.degrees(np.arctan2(R[1,0], R[0,0]))
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        roll  = np.degrees(np.arctan2(R[2,1], R[2,2]))

        # Mark tag center
        center_pt = tuple(np.mean(d.corners, axis=0).astype(int))
        cv2.circle(frame, center_pt, 5, (255,255,0), -1)
        cv2.putText(frame, f"ID:{d.tag_id}", (center_pt[0]+5, center_pt[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
        
        # 保存第一个检测到的标签的位姿
        if current_pose is None:
            current_pose = (rvec, tvec)
            
            # 显示位姿信息
            cv2.putText(frame, f"X: {tvec[0][0]:.2f}m", (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Y: {tvec[1][0]:.2f}m", (10, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Z: {tvec[2][0]:.2f}m", (10, 160), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Yaw: {yaw:.1f}°", (10, 190), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}°", (10, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Roll: {roll:.1f}°", (10, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 更新3D场景
    if current_pose is not None:
        rvec, tvec = current_pose
        
        # 为了更好的视觉效果，调整相机位置和旋转
        adjusted_tvec = tvec.copy()
        adjusted_rvec = rvec.copy()
        
        # 平移调整 - 增强深度效果
        adjusted_tvec[0][0] = -adjusted_tvec[0][0]  # 反转X轴方向，修复左右移动反向问题
        
        # 增强Z轴变化幅度，放大3-5倍
        depth_scale_factor = 10.0  # 深度灵敏度增强因子
        adjusted_tvec[2][0] = -adjusted_tvec[2][0] * depth_scale_factor  # 反转并放大Z轴效果
        
        # 减小基础偏移，让相机更靠近场景中心
        adjusted_tvec[2][0] += 3.0  # 减小固定偏移量，从1.0改为0.6
        
        # 旋转调整 - 反转Y轴旋转分量来修复旋转方向问题
        adjusted_rvec[1][0] = -adjusted_rvec[1][0]
        
        # 绘制3D场景
        scene = draw_3d_scene(frame, adjusted_rvec, adjusted_tvec, scene_camera_matrix)
        cv2.imshow("3D Scene", scene)

    # Compute FPS
    now = time.time()
    fps = 1.0 / (now - prev_time)
    prev_time = now

    # Draw info panel
    panel = []
    panel.append(f"FPS: {fps:.1f}")
    panel.append("Press [q] to exit")
    draw_text_panel(frame, panel)

    cv2.imshow("Live Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()