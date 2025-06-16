# apriltag_pcd_viewer.py

import sys
import os
import time
import copy
import numpy as np
import cv2
from plyfile import PlyData
import open3d as o3d
from pupil_apriltags import Detector

# ================================================
# 配置区域
# ================================================
# ---- 相机参数标定 (Phase 1) ----
CAMERA_INDEX     = 0           # cv2.VideoCapture 索引
WINDOW_W_CAL     = 1280
WINDOW_H_CAL     = 720

# AprilTag 标定板设置 (Phase 1 & 2)
TAG_FAMILY       = 'tag25h9'
TAG_SIZE         = 0.105   # meters
TAG_SPACING      = 0.117   # meters between tag centers (for board layout)
ROWS, COLS       = 1, 1        # 1×1 只有一个 Tag
BOARD_IDS        = [0]         # 只有一个 Tag，ID=0

TARGET_FRAMES    = 10          # Phase 1 采集足够多帧进行标定
MIN_TAGS_CAPTURE = 1           # 每次按 'c' 至少检测到的 Tag 数

# ---- 实时渲染 (Phase 2) ----
WINDOW_W_VIEW    = 1024
WINDOW_H_VIEW    = 768
POINT_SIZE       = 5.0
BACKGROUND_COLOR = np.array([0.1, 0.1, 0.1])  # 深灰

# PLY 点云文件路径 (请替换成你自己的点云)
PCD_FILE_PATH    = "input.ply"

# 点云世界坐标与 Tag–相机距离 的“缩放系数”
# 例如，如果点云本身单位是米，且 Tag–相机 单位也是米，则 SCALE=1.0 即可。
# 如果你想把点云当作更大／更小的“世界”，可以调节 SCALE。
SCALE = 2000.0

# ================================================
# 一些工具函数
# ================================================
def draw_progress(frame, count, target):
    """左下角绘制一个简单的进度条，用于 Phase 1 标定进度显示。"""
    h, w = frame.shape[:2]
    bar_w = int(w * (count / target))
    cv2.rectangle(frame, (0, h - 20), (bar_w, h), (0, 200, 0), -1)

def draw_text_panel(frame, lines):
    """
    在图像右上方绘制文本面板，每行显示一条内容。
    用于 Phase 2 显示 Tag ID、X/Y/Z、Yaw/Pitch/Roll 等信息。
    """
    h, w = frame.shape[:2]
    x0, y0 = w - 300 + 10, 10  # 300 像素宽的文本面板
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (x0, y0 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def load_ply_with_color(file_path: str) -> o3d.geometry.PointCloud:
    """利用 plyfile + Open3D，将带颜色和法线的 PLY 点云加载为 Open3D 点云对象。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PLY 文件不存在: {file_path}")

    ply = PlyData.read(file_path)
    vert = ply['vertex'].data

    # 提取点坐标 (xyz)、法线 (nx, ny, nz)、颜色 (red, green, blue)
    xyz     = np.vstack([vert['x'], vert['y'], vert['z']]).T
    normals = np.vstack([vert['nx'], vert['ny'], vert['nz']]).T
    colors  = np.vstack([vert['red'], vert['green'], vert['blue']]).T.astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(xyz)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors  = o3d.utility.Vector3dVector(colors)

    return pcd

# ================================================
# Phase 1：AprilTag 标定相机
# ================================================
def calibrate_camera():
    """
    用 AprilTag 棋盘拍摄多张图像，按 'c' 键采集，按 'q' 键结束。
    最后调用 cv2.calibrateCamera，返回 (ret, K, D, rvecs, tvecs)。
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    det = Detector(families=TAG_FAMILY)

    objpoints, imgpoints = [], []
    frame_count = 0
    centers = [(c * TAG_SPACING, r * TAG_SPACING)
               for r in range(ROWS) for c in range(COLS)]

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", WINDOW_W_CAL, WINDOW_H_CAL)
    print("Phase 1: Camera Calibration  —— 按 [c] 采集，按 [q] 结束")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧，退出。")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = det.detect(gray)

        # 画出检测到的 Tag
        for d in dets:
            pts = np.int32(d.corners).reshape(-1, 2)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{d.tag_id}", tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 半透明参考框（可选）
        h, w = frame.shape[:2]
        guide = frame.copy()
        cv2.rectangle(guide,
                      (int(w * 0.25), int(h * 0.25)),
                      (int(w * 0.75), int(h * 0.75)),
                      (255, 255, 255), 2)
        frame = cv2.addWeighted(guide, 0.3, frame, 0.7, 0)

        # UI 文本
        cv2.putText(frame,
                    f"Phase 1: Calibrating ({frame_count}/{TARGET_FRAMES})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
        cv2.putText(frame,
                    f"Press [c] to capture (>= {MIN_TAGS_CAPTURE} tags), [q] to finish",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        draw_progress(frame, frame_count, TARGET_FRAMES)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        # 用户按下 'c' 并且检测到足够 Tag 数时，保存一帧
        if key == ord('c') and len(dets) >= MIN_TAGS_CAPTURE:
            # 针对每个检测到的 Tag，生成 4 个角点的三维世界坐标和对应的图像坐标
            obj_pts, img_pts = [], []
            for d in dets:
                if d.tag_id not in BOARD_IDS:
                    continue
                idx = BOARD_IDS.index(d.tag_id)
                cx, cy = centers[idx]
                half = TAG_SIZE / 2
                # Tag 在世界坐标系下的 4 个角点 (Z=0 平面)
                obj_pts.append(np.array([
                    [cx - half, cy + half, 0],
                    [cx + half, cy + half, 0],
                    [cx + half, cy - half, 0],
                    [cx - half, cy - half, 0]
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
        print("采集帧数不足 (需要 >= 3)，标定失败。")
        sys.exit(1)

    # 执行相机标定
    h, w = gray.shape[:2]
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )
    print(f"Calibration 完成。重投影误差: {ret:.4f}")
    print("相机内参 K:\n", K)
    print("畸变系数 D:\n", D.ravel())
    return ret, K, D, rvecs, tvecs

# ================================================
# Phase 2 + 点云渲染：检测 Tag，实时更新点云视图
# ================================================
def run_live_view(K, D):
    """
    1) 打开摄像头，不断检测 AprilTag，
    2) 计算 Tag -> 相机 (rvec, tvec)，再求相机 -> Tag 的旋转和平移，
    3) 把“基于 Tag 坐标系”的点云转换到“相机坐标系”下并渲染。
    """
    # --- 初始化 AprilTag 检测 ---
    det = Detector(families=TAG_FAMILY)

    # --- 加载点云（只加载一次） ---
    print("正在加载点云文件，请稍候...")
    original_pcd = load_ply_with_color(PCD_FILE_PATH)

    # 对原始点云做一次预处理：按 SCALE 缩放，并将中心平移到 (0,0,0)（可选，看你的点云坐标是否已居中）
    # 下面假设点云坐标系下，已经把“Tag”当作(0,0,0)了。如果点云本身有偏移，需自行平移到原点。
    original_pcd.scale(SCALE, center=(0, 0, 0))

    # 创建一个 Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Live PointCloud View", width=WINDOW_W_VIEW, height=WINDOW_H_VIEW)
    vis.get_render_option().point_size = POINT_SIZE
    vis.get_render_option().background_color = BACKGROUND_COLOR

    # 把一个拷贝的点云对象加入渲染，此后我们会在循环里动态更新它的变换
    pcd_to_draw = copy.deepcopy(original_pcd)
    vis.add_geometry(pcd_to_draw)

    # 可选：渲染坐标系便于观察
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2 * SCALE)
    vis.add_geometry(axis)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.00005)


    # 打开摄像头
    cap = cv2.VideoCapture(CAMERA_INDEX)
    prev_time = time.time()
    centers = [(c * TAG_SPACING, r * TAG_SPACING) for r in range(ROWS) for c in range(COLS)]

    print("Phase 2: Live Pose Estimation + PointCloud 渲染 —— 按 [q] 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧，退出。")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = det.detect(gray)

        # 默认没有检测到 Tag 时，不更新点云，只显示之前状态
        cam_in_tag_R = None
        cam_in_tag_t = None
        panel_text = []

        if len(dets) > 0:
            # 这里只取第一个检测到的 Tag（BOARD_IDS 只有一个）
            d = dets[0]
            if d.tag_id in BOARD_IDS:
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
                if ok:
                    # 计算 Tag 坐标系到摄像头坐标系的旋转矩阵 R_tc 和平移向量 t_tc
                    R_tc, _ = cv2.Rodrigues(rvec)
                    t_tc = tvec.reshape(3)

                    # 取逆得到：相机在 Tag 坐标系下的位姿 R_ct = R_tc^T， t_ct = -R_tc^T * t_tc
                    R_ct = R_tc.T
                    t_ct = -R_ct.dot(t_tc)

                    cam_in_tag_R = R_ct
                    cam_in_tag_t = t_ct

                    # 计算 Euler 角 (Yaw, Pitch, Roll)
                    sy = np.hypot(R_ct[0, 0], R_ct[1, 0])
                    yaw   = np.degrees(np.arctan2(R_ct[1, 0], R_ct[0, 0]))
                    pitch = np.degrees(np.arctan2(-R_ct[2, 0], sy))
                    roll  = np.degrees(np.arctan2(R_ct[2, 1], R_ct[2, 2]))

                    # 绘制 Tag 轮廓和中心点
                    pts = np.int32(d.corners).reshape(-1, 2)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                    center_pt = tuple(np.mean(d.corners, axis=0).astype(int))
                    cv2.circle(frame, center_pt, 5, (255, 255, 0), -1)
                    cv2.putText(frame, f"ID:{d.tag_id}", (center_pt[0] + 5, center_pt[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    # 更新面板文本
                    panel_text = [
                        f"Tag ID: {d.tag_id}",
                        f"X={t_ct[0]:.2f}m",
                        f"Y={t_ct[1]:.2f}m",
                        f"Z={t_ct[2]:.2f}m",
                        f"Yaw={yaw:.1f}°",
                        f"Pitch={pitch:.1f}°",
                        f"Roll={roll:.1f}°"
                    ]

        # 绘制信息面板
        draw_text_panel(frame, panel_text)

        # 计算并显示 FPS
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Live Pose", frame)

        # 如果检测到有效的 相机->Tag 位姿，就更新点云变换
        # 更改这部分代码在 run_live_view 函数中
        if cam_in_tag_R is not None:
            # 保存当前视图参数
            view_control = vis.get_view_control()
            param = view_control.convert_to_pinhole_camera_parameters()
            
            # 构造 4×4 变换矩阵，把"Tag 坐标系下的点云" 变换到 "相机坐标系下"
            T = np.eye(4, dtype=np.float64)
            
            # 添加坐标系转换：OpenCV相机坐标系与Open3D坐标系的转换
            # OpenCV: X右、Y下、Z前，Open3D: X右、Y上、Z后
            # 创建坐标系修正矩阵（反转Y轴和Z轴，保持X轴不变）
            R_fix = np.array([
                [-1,  0,  0],  # 修改这里：X轴反向，让左右旋转方向反过来
                [0,  -1,  0],  # Y轴仍然反向
                [0,   0, -1]   # Z轴仍然反向
            ], dtype=np.float64)
            
            # 应用坐标系修正
            R_fixed = R_fix @ cam_in_tag_R
            T[:3, :3] = R_fixed
            T[:3, 3]  = cam_in_tag_t
            
            # 移除这行，因为我们已经在R_fix中反转了X轴
            T[2,:] = T[2,:] * -1  # 此行不再需要

            # 把原始点云先还原到初始状态，再应用新的变换
            pcd_to_draw = copy.deepcopy(original_pcd)
            pcd_to_draw.transform(T)


            # 更新 Visualizer 中的几何体
            vis.clear_geometries()
            vis.add_geometry(pcd_to_draw)
            vis.add_geometry(axis)
            
            # 恢复之前保存的视图参数
            view_control.convert_from_pinhole_camera_parameters(param)

        # 更新 Renderer
        vis.poll_events()
        vis.update_renderer()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()

# ================================================
# 整体入口
# ================================================
if __name__ == "__main__":
    # 先做相机标定（如果已经标定过，可以把下面两行替换为：K, D = np.load("K.npy"), np.load("D.npy") 之类的方式）
    ret, K, D, rvecs, tvecs = calibrate_camera()

    # 然后进入实时检测 + 点云渲染循环
    run_live_view(K, D)
