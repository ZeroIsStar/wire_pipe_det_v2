# WireAndPipeDetectionNode 参数文档

本文档详细说明 `wireandpipe_detection_node` 的所有可配置参数及其作用。该节点基于 YOLO 目标检测，识别图像中的电线和/或水管，并在检测到障碍物靠近机器人路径时发布避障标志（`/is_pipes_and_wires_in_path`）。

---

## 📌 节点概述

- **节点名称**：`wire_andpipe_detection_node`
- **包名**：`wireandpipe_detection_cpp`
- **功能**：
  - 订阅摄像头压缩图像，运行 ONNX Runtime 的 YOLO 模型进行目标检测。
  - 通过单目视觉地平面假设估算障碍物距离。
  - 将检测到的障碍物坐标转换到 `map` 坐标系，并缓存最近检测到的障碍物。
  - 结合全局路径和局部路径，判断障碍物是否威胁机器人前进路径，并发布避障状态。
  - 提供可视化标记（标注图像、路径点、障碍物标记等）。

---

## 📤 发布的主题

| 话题名称 | 消息类型 | 说明 |
|----------|----------|------|
| `/is_pipes_and_wires_in_path` | `std_msgs::msg::Bool` | 避障标志，`true` 表示检测到电线/水管在路径上，需要停车/减速 |
| `/rgb_camera_front/annotated_image_wire` | `sensor_msgs::msg::Image` | 标注了检测框和距离信息的图像（若 `publish_annotated_image` 为 `true`） |
| `/obstacles/map_markers` | `visualization_msgs::msg::MarkerArray` | 地图坐标系下的障碍物标记球体 |
| `/downsampled_path/markers` | `visualization_msgs::msg::MarkerArray` | 下采样后的路径点（蓝色全局，黄色局部） |
| `/raw_path/markers` | `visualization_msgs::msg::MarkerArray` | 原始路径线条（蓝色全局，橙色局部） |
| `/wire_pipe_markers` | `visualization_msgs::msg::MarkerArray` | 电线（蓝色）和水管（绿色）的3D位置标记 |
| `/wireandpipe/debug_map_markers` | `visualization_msgs::msg::MarkerArray` | 调试用标记：路径点、障碍物立方体（若 `publish_debug_map_markers` 为 `true`） |
| `/detected_objects/poses` | `geometry_msgs::msg::PoseArray` | 检测到的目标在 `map` 坐标系下的位姿（仅位置） |

---

## 📥 订阅的主题

| 话题名称 | 消息类型 | 说明 |
|----------|----------|------|
| `camera_topic`（参数） | `sensor_msgs::msg::CompressedImage` | 摄像头压缩图像，默认 `/rgb_camera_front/compressed` |
| `global_plan_topic`（参数） | `nav_msgs::msg::Path` | 全局路径，默认 `teb_global_plan` |
| `local_poses_topic`（参数） | `geometry_msgs::msg::PoseArray` | 局部路径点，默认 `teb_poses` |
| `/add_obstacle_point` | `geometry_msgs::msg::PointStamped` | 手动添加障碍物点（点击地图发送），会吸附到最近路径点并加入缓存 |

---

## ⚙️ 参数详解（按声明顺序）

### 相机参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `camera_topic` | string | `/rgb_camera_front/compressed` | 订阅的压缩图像话题 |
| `rgb_camera_frame` | string | `front_camera_color_frame` | 相机坐标系名称（用于TF变换） |
| `fx` | double | `454.58425` | 相机内参 – 焦距 x（像素） |
| `fy` | double | `452.05297` | 相机内参 – 焦距 y（像素） |
| `cx` | double | `309.77324` | 相机内参 – 主点 x（像素） |
| `cy` | double | `280.9` | 相机内参 – 主点 y（像素） |
| `camera_width` | int | `640` | 图像宽度（像素），用于缩放计算 |
| `camera_height` | int | `480` | 图像高度（像素） |
| `h_fov_rad` | double | `1.3378` | 水平视场角（弧度），仅用于日志输出 |
| `v_fov_rad` | double | `1.1868` | 垂直视场角（弧度），仅用于日志输出 |
| `camera_x` | double | `0.0` | 相机在机器人基坐标系中的 x 偏移（米） |
| `camera_y` | double | `0.0` | 相机在机器人基坐标系中的 y 偏移（米） |
| `camera_z` | double | `0.905` | 相机在机器人基坐标系中的 z 偏移（米），即安装高度 |
| `camera_pitch` | double | `-0.080286` | 相机俯仰角（弧度），正值为向下倾斜 |
| `camera_yaw` | double | `0.0` | 相机偏航角（弧度） |
| `camera_roll` | double | `0.0` | 相机翻滚角（弧度） |
| `distortion_coefficients` | double array | `[0,0,0,0.0,0.0]` | 畸变系数 (k1, k2, p1, p2, k3)，用于 `cv::undistort` |

> **注意**：相机内参和畸变系数用于距离估算和图像校正。若所有畸变系数为0，则跳过畸变校正。

### YOLO 检测参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `yolo_model_path` | string | `/capella/lib/python3.10/site-packages/wire_pipe_det/src/7_10.onnx` | ONNX 模型文件路径（支持相对/绝对路径） |
| `class_names` | string array | `["wire", "water_pipe"]` | 类别名称列表（必须与模型输出顺序一致） |
| `wire_class_id` | int | `0` | 电线类别在 `class_names` 中的索引 |
| `water_pipe_class_id` | int | `1` | 水管类别在 `class_names` 中的索引 |
| `conf_threshold` | double | `0.6` | 目标检测置信度阈值（0~1），低于该值将被过滤 |
| `filter_above_horizon` | bool | `true` | 若为 `true`，检测框完全位于图像上半部分（地平线以上）时忽略（避免远处物体干扰） |
| `yolo_frame_skip` | int | `3` | 跳帧数，即每 N 帧执行一次 YOLO 推理（1 表示每帧都推理） |

### 目标跟踪参数（用于稳定检测）

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `confirm_threshold` | double | `1.0` | 跟踪累积对数分数（log-odds）阈值，超过该值才确认目标（防止虚检） |
| `tracking_iou_threshold` | double | `0.3` | 前后帧检测框之间的 IoU 阈值，用于匹配同一目标 |
| `max_track_age` | int | `5` | 目标最大“年龄”（帧数），超过后未匹配则丢弃 |
| `decay_factor` | double | `0.9` | 累积分数的衰减因子（0~1），每帧乘以该值 |

### 路径与避障参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `global_plan_topic` | string | `teb_global_plan` | 全局路径话题，通常来自全局规划器 |
| `local_poses_topic` | string | `teb_poses` | 局部路径点话题，通常来自局部规划器（如 TEB） |
| `global_search_distance` | double | `5.0` | 全局路径下采样的最大搜索距离（米），只考虑前方这么远的点 |
| `local_search_distance` | double | `5.0` | 局部路径下采样的最大搜索距离 |
| `pedestrian_distance_threshold` | double | `1.0` | **障碍物到路径的垂直距离阈值（米）**，小于该值视为在路径上，触发避障 |
| `trigger_robot_distance` | double | `2.0` | **障碍物到机器人的欧氏距离阈值（米）**，只有障碍物小于该距离且同时满足路径距离条件时才触发避障（避免远处误触） |
| `avoid_hold_seconds` | double | `1.8` | 触发避障后，即使障碍物消失，标志仍保持 `true` 的最短时间（秒），防止频繁切换 |
| `distance_log_throttle_sec` | double | `2.0` | 距离信息日志的节流时间（秒），避免刷屏 |

### 调试与可视化参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `publish_annotated_image` | bool | `true` | 是否发布标注了检测框和距离信息的图像 |
| `publish_debug_map_markers` | bool | `true` | 是否发布详细的调试标记（路径点、障碍物立方体等） |

---

## 🧩 核心避障逻辑

避障标志 `/is_pipes_and_wires_in_path` 的发布由以下条件共同决定：

1. **目标确认**：检测到的目标必须经过跟踪确认（累积分数 ≥ `confirm_threshold`）。
2. **距离路径**：目标在 `map` 坐标系下的位置到下采样全局或局部路径的最短距离 ≤ `pedestrian_distance_threshold`。
3. **距离机器人**：目标到机器人的欧氏距离 ≤ `trigger_robot_distance`（若 TF 可用）。
4. **超时管理**：一旦触发，标志保持 `true` 至少 `avoid_hold_seconds` 秒，即使目标暂时离开条件。

此外，障碍物缓存（`g_known_obstacles`）会合并 0.3m 内的检测，并在目标远离路径超过 `KNOWN_OBSTACLE_TIMEOUT`（5 秒）后自动清除。

---

## 🛠️ 手动添加障碍物

节点提供两种手动添加障碍物的方式：

- **服务** `/add_dummy_obstacle`（`std_srvs/srv/Empty`）：调用后在路径前方 1 米处生成一个虚拟障碍点。
- **话题** `/add_obstacle_point`（`geometry_msgs::msg::PointStamped`）：发布一个点（任意坐标系），节点会将其投影到最近的路径点并添加为障碍物。

手动添加的障碍物同样会参与避障判断，并在超时后自动清除。

---

## 📝 使用示例

```bash
# 启动节点（使用默认参数）
ros2 run wireandpipe_detection_cpp wireandpipe_detection_node

# 手动添加虚拟障碍物
ros2 service call /add_dummy_obstacle std_srvs/srv/Empty

# 从 RViz 点击地图发送点（需配置 RViz 的 "Publish Point" 工具）
# 话题为 /add_obstacle_point，坐标系需正确
