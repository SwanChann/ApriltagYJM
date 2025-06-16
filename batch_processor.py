import cv2
import numpy as np
import glob
import os
import argparse
from pupil_apriltags import Detector
import math
from collections import deque

# ====================
# 参数配置区域，默认值可以根据需要修改
# ====================
TAG_FAMILY       = 'tag25h9'
TAG_SIZE         = 0.105   # meters
TAG_SPACING      = 0.117   # meters between tag centers (for board layout)
ROWS, COLS       = 2, 2
BOARD_IDS        = [0, 1, 2, 3]
AXIS_LENGTH      = 0.10    # meters
NORM_LINE_LENGTH = 0.20    # meters
PANEL_WIDTH      = 300     # pixels
EMA_ALPHA        = 0.4     # 指数加权平均的平滑因子，可根据抖动情况调整

# 计算棋盘板中每个 tag 中心的世界坐标（XY 平面）
centers = [(c * TAG_SPACING, r * TAG_SPACING)
           for r in range(ROWS) for c in range(COLS)]


def draw_text_panel(frame, lines):
    h, w = frame.shape[:2]
    x0, y0 = w - PANEL_WIDTH + 10, 10
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (x0, y0 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def calibrate_camera_from_images(image_paths, min_tags_capture=1):
    """
    对给定的一组标定图像进行 Tag 检测并标定摄像头。
    返回: camera matrix K, distortion coefficients D
    """
    det = Detector(families=TAG_FAMILY)
    objpoints, imgpoints = [], []
    valid_frames = 0

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"无法读取图像: {img_path}")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = det.detect(gray)

        # 统计有效 tag
        if len(dets) < min_tags_capture:
            continue

        for d in dets:
            if d.tag_id not in BOARD_IDS:
                continue
            idx = BOARD_IDS.index(d.tag_id)
            cx, cy = centers[idx]
            half = TAG_SIZE / 2
            # 四个角点的世界坐标
            obj_pts = np.array([
                [cx - half, cy + half, 0],
                [cx + half, cy + half, 0],
                [cx + half, cy - half, 0],
                [cx - half, cy - half, 0]
            ], dtype=np.float32)
            img_pts = np.array(d.corners, dtype=np.float32)
            objpoints.append(obj_pts)
            imgpoints.append(img_pts)
        valid_frames += 1

    if valid_frames < 3:
        raise ValueError("标定图像太少，至少需要 3 帧有效检测。")

    # 使用最后一张图像的灰度尺寸来标定
    gray = cv2.cvtColor(cv2.imread(image_paths[-1]), cv2.COLOR_BGR2GRAY)
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    print(f"标定完成。重投影误差: {ret:.4f}")
    print("Camera matrix K:\n", K)
    print("Distortion coefficients D:\n", D.ravel())
    return K, D


def process_video_with_pose(video_path, K, D, output_path=None):
    """
    对视频逐帧进行 Tag 检测和位姿解算，渲染法线方向与姿态信息。
    如果提供 output_path，会将结果写入视频文件。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = None
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    if output_path:
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    det = Detector(families=TAG_FAMILY)
    normal_vec = np.array([0, 0, 1], dtype=np.float32) * NORM_LINE_LENGTH

    # ========= 改进点 1：在进入循环前加上 prev_time 的初始化 =========
    prev_time = cv2.getTickCount()

    # EMA 平滑需要的变量
    prev_rvec = None
    prev_tvec = None

    cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pose Estimation", width, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = det.detect(gray)

        # --------------------
        # 1) 多 Tag 汇总角点，做 solvePnPRansac + RefineLM
        # --------------------
        all_obj_pts = []
        all_img_pts = []
        valid_tag_indices = []
        for idx_d, d in enumerate(dets):
            if d.tag_id not in BOARD_IDS:
                continue
            idx_board = BOARD_IDS.index(d.tag_id)
            cx, cy = centers[idx_board]
            half = TAG_SIZE / 2
            obj_pts = np.array([
                [cx - half, cy + half, 0],
                [cx + half, cy + half, 0],
                [cx + half, cy - half, 0],
                [cx - half, cy - half, 0]
            ], dtype=np.float32)
            img_pts = np.array(d.corners, dtype=np.float32)
            all_obj_pts.append(obj_pts)
            all_img_pts.append(img_pts)
            valid_tag_indices.append(idx_board)

        if len(all_obj_pts) == 0:
            # 如果一帧里连一个都没检测到，就跳过——也可以用上一帧的平滑值渲染上一帧的结果
            if prev_rvec is not None:
                smooth_rvec = prev_rvec.copy()
                smooth_tvec = prev_tvec.copy()
            else:
                cv2.imshow("Pose Estimation", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
        else:
            # 把 objpoints/imgpoints 从 list 展平为 N×3 和 N×2
            all_obj_pts = np.vstack(all_obj_pts)
            all_img_pts = np.vstack(all_img_pts)

            # ransac + refine
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                all_obj_pts, all_img_pts, K, D,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=3.0,
                iterationsCount=100
            )
            if ok:
                rvec, tvec = cv2.solvePnPRefineLM(
                    all_obj_pts, all_img_pts, K, D, rvec, tvec
                )
                # ========= 改进点 2：对所有 tag 都用同一组 (smooth_rvec, smooth_tvec) ，然后分别渲染 =========
                if prev_rvec is None:
                    smooth_rvec = rvec.copy()
                    smooth_tvec = tvec.copy()
                else:
                    smooth_rvec = (1 - EMA_ALPHA) * prev_rvec + EMA_ALPHA * rvec
                    smooth_tvec = (1 - EMA_ALPHA) * prev_tvec + EMA_ALPHA * tvec
                prev_rvec = smooth_rvec.copy()
                prev_tvec = smooth_tvec.copy()
            else:
                # RANSAC 失败，就退回上一帧的值
                if prev_rvec is not None:
                    smooth_rvec = prev_rvec.copy()
                    smooth_tvec = prev_tvec.copy()
                else:
                    cv2.imshow("Pose Estimation", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

        # --------------------
        # 2) 用平滑后的 (smooth_rvec, smooth_tvec) 来计算旋转矩阵
        # --------------------
        R_mat, _ = cv2.Rodrigues(smooth_rvec)

        # --------------------
        # 3) 对每个检测到的 tag 渲染法线（改为对所有检测到的 tag 都画，而不是只画第一个）
        # --------------------
        for idx_d, d in enumerate(dets):
            if d.tag_id not in BOARD_IDS:
                continue
            idx_board = BOARD_IDS.index(d.tag_id)
            cx, cy = centers[idx_board]
            start_pt_tag = np.array([cx, cy, 0], dtype=np.float32)
            end_pt_tag   = start_pt_tag - (R_mat @ normal_vec)
            pts_norm, _  = cv2.projectPoints(
                np.vstack([start_pt_tag, end_pt_tag]),
                smooth_rvec, smooth_tvec, K, D
            )
            if np.any(np.isnan(pts_norm)):
                continue
            p0 = tuple(pts_norm[0].ravel().astype(int))
            p1 = tuple(pts_norm[1].ravel().astype(int))

            # 绘制法线箭头和同心圆以增强 3D 效果
            cv2.line(frame, p0, p1, (0, 255, 255), 3)
            radius_increment = 5
            num_circles = 3
            for i in range(1, num_circles + 1):
                radius = i * radius_increment
                cv2.circle(frame, p1, radius, (0, 255, 255), 2)

            # 标记该 tag 的中心并绘制 ID
            center_pt = tuple(np.mean(d.corners, axis=0).astype(int))
            cv2.circle(frame, center_pt, 5, (255, 255, 0), -1)
            cv2.putText(frame, f"ID:{d.tag_id}", (center_pt[0] + 5, center_pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # --------------------
        # 4) 绘制姿态信息面板（仅使用第一帧检测到的 tag 对应的旋转矩阵 R_mat，即对整个 board 来说是同一个 rvec/tvec）
        # --------------------
        now = cv2.getTickCount()
        dt = (now - prev_time) / cv2.getTickFrequency()
        prev_time = now
        fps_cur = 1.0 / dt if dt > 0 else 0.0

        panel = [f"FPS: {fps_cur:.1f}"]
        # 计算欧拉角
        sy = math.hypot(R_mat[0, 0], R_mat[1, 0])
        if sy < 1e-6:
            pitch = math.degrees(math.atan2(-R_mat[1, 2],  R_mat[1, 1]))
            yaw   = math.degrees(math.atan2(-R_mat[2, 0], sy))
            roll  = 0.0
        else:
            pitch = math.degrees(math.atan2(R_mat[2, 1], R_mat[2, 2]))
            yaw   = math.degrees(math.atan2(-R_mat[2, 0], sy))
            roll  = math.degrees(math.atan2(R_mat[1, 0], R_mat[0, 0]))

        panel += [
            f"X={smooth_tvec[0,0]:.2f}m",
            f"Y={smooth_tvec[1,0]:.2f}m",
            f"Z={smooth_tvec[2,0]:.2f}m",
            f"Yaw={yaw:.1f}degree",
            f"Pitch={pitch:.1f}degree",
            f"Roll={roll:.1f}degree"
        ]
        draw_text_panel(frame, panel)

        # 显示并可选保存
        cv2.imshow("Pose Estimation", frame)
        if out_writer is not None:
            out_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="批量标定并渲染Tag视频的脚本"
    )
    parser.add_argument('--images', type=str,  default = 'calib_images/*.jpg',
                        help="用于标定的图像文件夹或文件路径（支持通配符，如'images/*.jpg'）")
    parser.add_argument('--video', type=str,  default = './result/input.mp4',
                        help="待处理的视频文件路径")
    parser.add_argument('--output', type=str,  default = './result/output.mp4',
                        help="可选，输出渲染后的视频文件路径，如'output.mp4'")
    args = parser.parse_args()

    # 收集标定图像路径列表
    image_paths = glob.glob(args.images)
    if not image_paths:
        raise FileNotFoundError(f"未找到匹配的图像: {args.images}")

    # 标定摄像头
    K, D = calibrate_camera_from_images(image_paths)

    # 处理视频并渲染
    process_video_with_pose(args.video, K, D, args.output)


