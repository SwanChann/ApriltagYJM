# AprilTag 实时位姿估计与点云渲染系统

##  项目简介

本项目提供了一套基于 AprilTag 的完整解决方案，集相机标定、实时位姿估计与点云渲染于一体。通过高效检测 AprilTag 标定板，系统能够实时计算相机与标定板之间的精确位姿关系，并将预加载的点云数据实时转换到相机坐标系中进行渲染，为 AR/VR、机器人导航、三维重建等应用提供基础。

##  主要功能

1.  **相机高精度标定**：利用 AprilTag 标定板，通过多视角图像采集，精确计算相机的内参矩阵和畸变系数。
2.  **实时位姿跟踪**：基于 AprilTag 检测结果，实时、高鲁棒性地估计相机相对于世界坐标系（或标定板）的六自由度位姿。
3.  **沉浸式点云渲染**：支持加载 PLY 格式点云数据，并将其实时投影并渲染到相机视角中，实现虚实融合的视觉效果。
4.  **批量处理能力**：支持对图像序列和视频进行批量标定和位姿估计，并输出渲染结果。
5.  **虚拟场景交互**：提供在 3D 虚拟场景中实时渲染 AprilTag 及其位姿的功能。

##  项目结构

```
├── point_cloud_viewer.py     # 核心脚本：相机标定、实时位姿估计与点云渲染
├── batch_processor.py        # 批量相机标定与视频处理
├── virtual_scene_renderer.py # 3D 虚拟场景渲染与实时位姿估计
├── basic_calibration.py      # 基础相机标定与实时位姿估计示例
├── calib_images/             # 用于相机标定的图像文件夹
├── input.ply                 # 默认点云文件 (请替换为您的点云数据)
└── result/                   # 输出结果文件夹 (标定参数、处理后的视频等)
```

##  环境依赖

-   Python 3.8+
-   OpenCV (`opencv-python`)
-   Open3D
-   NumPy
-   Pupil AprilTags (`pupil-apriltags`)
-   Plyfile
-   Pillow

**安装依赖：**

```bash
pip install opencv-python open3d numpy pupil-apriltags plyfile pillow
```

##  使用指南

### 1. 相机标定

运行 `point_cloud_viewer.py` 中的 `calibrate_camera` 函数进行相机标定。程序将启动摄像头，请将 AprilTag 标定板置于视野中，并从不同角度、距离拍摄多张图像。

```bash
python point_cloud_viewer.py
# 运行后，程序将自动进入相机标定模式
# 在窗口中：
#   - 按 'c' 键采集当前帧图像用于标定
#   - 按 'q' 键结束图像采集并进行标定
```

标定完成后，相机的内参矩阵和畸变系数将显示在控制台，并可保存到文件。


### 2. 批量标定与视频处理

使用 `batch_processor.py` 脚本进行批量相机标定和视频处理。此脚本支持从指定图像路径进行标定，并对视频文件进行 AprilTag 检测和渲染。

```bash
# 示例：使用 calib_images/ 文件夹中的图片进行标定，并处理 input.mp4 视频
python batch_processor.py --images calib_images/*.jpg --video input.mp4 --output result/output_rendered.mp4
```

### 3. 3D 虚拟场景渲染

运行 `virtual_scene_renderer.py` 脚本，可以实时检测 AprilTag 并将其位姿映射到虚拟的 3D 场景中进行渲染，适用于可视化位姿或构建简单的 AR 场景。

```bash
python virtual_scene_renderer.py
```

### 4. 实时位姿估计与点云渲染

在完成相机标定后（或使用已有的相机参数），运行 `point_cloud_viewer.py` 中的 `run_live_view` 函数，即可实时检测 AprilTag 并将预设的点云数据渲染到相机画面中。

```bash
python point_cloud_viewer.py
# 运行后，程序将自动进入实时位姿估计与点云渲染模式
# 在窗口中：
#   - 按 'q' 键退出程序
```


### 5. AprilTag 图像工具

`apriltag_resources/tag_to_svg.py` 脚本可以将 AprilTag 的 PNG 图像转换为 SVG 格式，方便打印或在矢量图形软件中使用。

```bash
python apriltag_resources/tag_to_svg.py apriltag_resources/tag25h9/tag25_9_00000.png output.svg --size=20mm
```

##  配置参数

您可以在各个脚本文件的顶部找到配置区域，调整以下关键参数以适应您的硬件和需求：

-   `TAG_FAMILY`：AprilTag 家族名称（例如：`tag25h9`）。
-   `TAG_SIZE`：您使用的 AprilTag 物理边长（单位：米）。**请务必准确测量此值。**
-   `TAG_SPACING`：标定板中 AprilTag 之间的间距（单位：米）。
-   `WINDOW_W_CAL` 和 `WINDOW_H_CAL`：标定窗口的宽度和高度。
-   `WINDOW_W_VIEW` 和 `WINDOW_H_VIEW`：实时渲染窗口的宽度和高度。
-   `PCD_FILE_PATH`：默认加载的点云文件路径。
-   `SCALE`：点云的缩放系数，用于调整点云在相机坐标系中的大小。
-   `BOARD_IDS`：标定板上 AprilTag 的 ID 列表，**请确保与您使用的实际标定板 ID 一致**。

##  注意事项

-   **AprilTag ID 匹配**：请务必确保脚本中 `BOARD_IDS` 配置与您实际使用的 AprilTag 标定板上的 ID 列表完全一致，否则位姿估计将失败。
-   **点云文件格式**：点云文件需为标准的 PLY 格式，建议包含颜色和法线信息以获得更好的渲染效果。
-   **点云坐标系**：如果加载的点云在渲染时位置不正确，可能需要根据您的点云数据进行适当的平移或旋转变换，以使其与 AprilTag 标定板的坐标系对齐。

