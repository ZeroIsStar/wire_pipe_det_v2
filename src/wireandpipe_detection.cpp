#include "wireandpipe_detection_cpp/wireandpipe_detection.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <opencv2/imgcodecs.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/utils.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>   
#include <algorithm>
#include <numeric>
#include <sstream>

using namespace std::chrono_literals;

namespace
{
// ---- 矩形辅助函数：计算水平矩形四个角点（angle=0） ----
static std::vector<cv::Point2f> getRectCorners(const ObbBBox &obb)
{
    std::vector<cv::Point2f> pts(4);
    const float cx = obb.cx;
    const float cy = obb.cy;
    const float dx = obb.width * 0.5F;
    const float dy = obb.height * 0.5F;
    pts[0] = cv::Point2f(cx - dx, cy - dy);
    pts[1] = cv::Point2f(cx + dx, cy - dy);
    pts[2] = cv::Point2f(cx + dx, cy + dy);
    pts[3] = cv::Point2f(cx - dx, cy + dy);
    return pts;
}

// ---- 矩形 IoU（水平框） ----
static float rectIoU(const ObbBBox &a, const ObbBBox &b)
{
    std::vector<cv::Point2f> inter;
    cv::intersectConvexConvex(getRectCorners(a), getRectCorners(b), inter, true);
    const float inter_area = inter.empty() ? 0.0F : static_cast<float>(cv::contourArea(inter));
    const float union_area = (a.width * a.height) + (b.width * b.height) - inter_area;
    return union_area > 0.0F ? inter_area / union_area : 0.0F;
}

// =============================================================================
// 新增：已知障碍物缓存（用于解决视野盲区）
// =============================================================================
struct KnownObstacle {
    geometry_msgs::msg::Point pos;      // map 坐标系
    rclcpp::Time last_seen;             // 最近一次被检测到的时间
};
static std::vector<KnownObstacle> g_known_obstacles;
static std::mutex g_known_obstacles_mutex;
const double KNOWN_OBSTACLE_MERGE_DIST = 0.3;   // 合并距离阈值（米）
const double KNOWN_OBSTACLE_TIMEOUT   = 5.0;    // 超时清除时间（秒）

// =============================================================================
// OrtYoloTracker: ONNX Runtime YOLO 通用目标检测
// =============================================================================
class OrtYoloTracker : public ObbYoloTracker
{
public:
    OrtYoloTracker(const std::string &model_path,
                   float conf_threshold,
                   const rclcpp::Logger &logger)
        : logger_(logger), steady_clock_(RCL_SYSTEM_TIME),
          env_(ORT_LOGGING_LEVEL_WARNING, "yolo"),
          conf_threshold_(conf_threshold)
    {
        try {
            Ort::SessionOptions session_options;
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);

            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

            auto input_name_ptr = session_->GetInputNameAllocated(0, allocator_);
            input_name_ = input_name_ptr.get();
            auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator_);
            output_name_ = output_name_ptr.get();

            yolo_enabled_ = true;
            RCLCPP_INFO(logger_, "[YOLO] Model loaded: %s", model_path.c_str());
            RCLCPP_INFO(logger_, "[YOLO] GPU backend enabled (ONNX Runtime CUDA EP)");

        } catch (const std::exception &e) {
            RCLCPP_ERROR(logger_, "[YOLO] Exception while loading model: %s", e.what());
            yolo_enabled_ = false;
        }
    }

    std::vector<ObbTrackItem> track(const cv::Mat &frame) override
    {
        std::vector<ObbTrackItem> tracks;
        if (!yolo_enabled_ || frame.empty()) return tracks;

        constexpr int input_size = 640;

        // ---- letterbox 预处理 ----
        const float scale_x = static_cast<float>(input_size) / static_cast<float>(frame.cols);
        const float scale_y = static_cast<float>(input_size) / static_cast<float>(frame.rows);
        const float scale = std::min(scale_x, scale_y);
        const int new_w = static_cast<int>(std::round(frame.cols * scale));
        const int new_h = static_cast<int>(std::round(frame.rows * scale));
        const int pad_left = (input_size - new_w) / 2;
        const int pad_top  = (input_size - new_h) / 2;

        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        cv::Mat letterboxed(input_size, input_size, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(letterboxed(cv::Rect(pad_left, pad_top, new_w, new_h)));

        cv::Mat blob;
        cv::dnn::blobFromImage(letterboxed, blob, 1.0 / 255.0,
                               cv::Size(input_size, input_size),
                               cv::Scalar(), true, false);

        std::vector<int64_t> input_shape = {1, 3, 640, 640};
        size_t input_tensor_size = 1 * 3 * 640 * 640;
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, (float*)blob.data, input_tensor_size, input_shape.data(), input_shape.size());

        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                            input_names, &input_tensor, 1,
                                            output_names, 1);

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        static bool first_inference_logged = false;
        if (!first_inference_logged) {
            RCLCPP_INFO(logger_, "[YOLO] First inference raw output:");
            RCLCPP_INFO(logger_, "[YOLO]   output_shape = [%ld, %ld, %ld]",
                (long)output_shape[0], (long)output_shape[1], (long)output_shape[2]);
            first_inference_logged = true;
        }

        int rows = static_cast<int>(output_shape[1]);
        int cols = static_cast<int>(output_shape[2]);

        cv::Mat det;
        if (rows < cols && rows != 1) {
            cv::Mat raw(rows, cols, CV_32F, output_data);
            cv::transpose(raw, det);
            RCLCPP_DEBUG(logger_, "[YOLO] Transposed from [%d,%d] to [%d,%d]",
                rows, cols, det.rows, det.cols);
        } else {
            det = cv::Mat(rows, cols, CV_32F, output_data).clone();
        }

        const int num_detections = det.rows;
        const int det_channels   = det.cols;
        if (det_channels < 5) {
            RCLCPP_WARN_THROTTLE(logger_, steady_clock_, 2000,
                "[YOLO] Feature channels < 5, invalid output");
            return tracks;
        }

        const int num_classes = det_channels - 4;
        if (num_classes < 1) {
            RCLCPP_WARN_THROTTLE(logger_, steady_clock_, 2000,
                "[YOLO] No class channels detected");
            return tracks;
        }

        int class_offset = 4;
        if (det_channels == 5 + num_classes) {
            class_offset = 5;
            RCLCPP_DEBUG(logger_, "[YOLO] Detected obj_conf channel, offset=5");
        } else {
            class_offset = 4;
        }

        std::vector<float> confidences;
        std::vector<int> class_ids;
        std::vector<ObbBBox> boxes;

        for (int i = 0; i < num_detections; ++i) {
            const float *row = det.ptr<float>(i);
            if (!row) continue;

            int best_class = -1;
            float best_score = 0.0F;
            for (int c = 0; c < num_classes; ++c) {
                const float score = row[class_offset + c];
                if (score > best_score) {
                    best_score = score;
                    best_class = c;
                }
            }

            if (best_score < conf_threshold_ || best_class < 0) continue;

            ObbBBox box;
            box.cx     = (row[0] - static_cast<float>(pad_left)) / scale;
            box.cy     = (row[1] - static_cast<float>(pad_top)) / scale;
            box.width  = row[2] / scale;
            box.height = row[3] / scale;
            box.angle  = 0.0F;

            confidences.push_back(best_score);
            class_ids.push_back(best_class);
            boxes.push_back(box);
        }

        std::vector<int> order(confidences.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
            [&](int a, int b) { return confidences[a] > confidences[b]; });

        std::vector<int> indices;
        std::vector<bool> suppressed(boxes.size(), false);
        const float nms_threshold = 0.65F;
        for (size_t i = 0; i < order.size(); ++i) {
            const int idx = order[i];
            if (suppressed[idx]) continue;
            indices.push_back(idx);
            for (size_t j = i + 1; j < order.size(); ++j) {
                const int jdx = order[j];
                if (suppressed[jdx]) continue;
                if (rectIoU(boxes[idx], boxes[jdx]) > nms_threshold) {
                    suppressed[jdx] = true;
                }
            }
        }

        tracks.reserve(indices.size());
        for (int idx : indices) {
            ObbTrackItem item;
            item.class_id   = class_ids[idx];
            item.confidence = confidences[idx];
            item.bbox       = boxes[idx];
            tracks.push_back(item);
        }

        return tracks;
    }

private:
    rclcpp::Logger logger_;
    rclcpp::Clock steady_clock_;
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string input_name_;
    std::string output_name_;
    bool yolo_enabled_{false};
    float conf_threshold_{0.25};
};

}  // namespace

// =============================================================================
// WireAndPipeDetectionNode 实现
// =============================================================================

WireAndPipeDetectionNode::WireAndPipeDetectionNode()
    : Node("wire_andpipe_detection_node")
{
    // ---- 声明参数 ----
    this->declare_parameter<std::string>("camera_topic", "/rgb_camera_front/compressed");
    this->declare_parameter<std::string>("rgb_camera_frame", "front_camera_color_frame");
    this->declare_parameter<double>("fx", 454.58425);       
    this->declare_parameter<double>("fy", 452.05297);       
    this->declare_parameter<double>("cx", 309.77324);       
    this->declare_parameter<double>("cy", 280.9);       
    this->declare_parameter<int>("camera_width", 640);     
    this->declare_parameter<int>("camera_height", 480);    
    this->declare_parameter<double>("h_fov_rad", 1.3378);
    this->declare_parameter<double>("v_fov_rad", 1.1868);
    this->declare_parameter<double>("camera_x", 0.0);
    this->declare_parameter<double>("camera_y", 0.0);
    this->declare_parameter<double>("camera_z", 0.905);
    this->declare_parameter<double>("camera_pitch", -0.080286);
    this->declare_parameter<double>("camera_yaw", 0.0);
    this->declare_parameter<double>("camera_roll", 0.0);

    this->declare_parameter<std::string>("yolo_model_path", "/capella/lib/python3.10/site-packages/wire_pipe_det/src/7_8_1.onnx");
    this->declare_parameter<std::vector<std::string>>("class_names", std::vector<std::string>{"wire", "water_pipe"});
    this->declare_parameter<int>("wire_class_id", 0);
    this->declare_parameter<int>("water_pipe_class_id", 1);
    this->declare_parameter<double>("conf_threshold", 0.6);
    this->declare_parameter<double>("distance_log_throttle_sec", 2.0);

    this->declare_parameter<std::string>("global_plan_topic", "teb_global_plan");
    this->declare_parameter<std::string>("local_poses_topic", "teb_poses");
    this->declare_parameter<double>("global_search_distance", 5.0);
    this->declare_parameter<double>("local_search_distance", 5.0);

    this->declare_parameter<double>("pedestrian_distance_threshold", 1.0);
    this->declare_parameter<double>("avoid_hold_seconds", 1.8);

    this->declare_parameter<int>("yolo_frame_skip", 3);
    this->declare_parameter<bool>("publish_annotated_image", true);
    this->declare_parameter<bool>("publish_debug_map_markers", true);
    this->declare_parameter<bool>("filter_above_horizon", true);

    // 跟踪参数
    this->declare_parameter<double>("confirm_threshold", 1.0);
    this->declare_parameter<double>("tracking_iou_threshold", 0.3);
    this->declare_parameter<int>("max_track_age", 5);
    this->declare_parameter<double>("decay_factor", 0.9);

    // 畸变系数
    this->declare_parameter<std::vector<double>>("distortion_coefficients",
        std::vector<double>{0, 0, 0, 0.0, 0.0});

    // ---- 读取参数 ----
    fx_ = this->get_parameter("fx").as_double();
    fy_ = this->get_parameter("fy").as_double();
    cx_ = this->get_parameter("cx").as_double();
    cy_ = this->get_parameter("cy").as_double();
    camera_width_ = this->get_parameter("camera_width").as_int();
    camera_height_ = this->get_parameter("camera_height").as_int();
    h_fov_rad_ = this->get_parameter("h_fov_rad").as_double();
    v_fov_rad_ = this->get_parameter("v_fov_rad").as_double();
    camera_x_ = this->get_parameter("camera_x").as_double();
    camera_y_ = this->get_parameter("camera_y").as_double();
    camera_z_ = this->get_parameter("camera_z").as_double();
    camera_pitch_ = this->get_parameter("camera_pitch").as_double();
    camera_yaw_ = this->get_parameter("camera_yaw").as_double();
    camera_roll_ = this->get_parameter("camera_roll").as_double();

    class_names_ = this->get_parameter("class_names").as_string_array();
    wire_class_id_ = this->get_parameter("wire_class_id").as_int();
    water_pipe_class_id_ = this->get_parameter("water_pipe_class_id").as_int();
    conf_threshold_ = this->get_parameter("conf_threshold").as_double();
    distance_log_throttle_sec_ = this->get_parameter("distance_log_throttle_sec").as_double();

    auto dist_vec = this->get_parameter("distortion_coefficients").as_double_array();
    if (dist_vec.size() >= 5) {
        dist_coeffs_ = cv::Mat(1, 5, CV_64F);
        for (size_t i = 0; i < 5; ++i) dist_coeffs_.at<double>(0, i) = dist_vec[i];
    } else {
        dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);
    }

    camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
    camera_matrix_.at<double>(0, 0) = fx_;
    camera_matrix_.at<double>(1, 1) = fy_;
    camera_matrix_.at<double>(0, 2) = cx_;
    camera_matrix_.at<double>(1, 2) = cy_;

    if (class_names_.empty()) {
        detection_label_ = "obstacle";
    } else {
        detection_label_ = class_names_[0];
        for (size_t i = 1; i < class_names_.size(); ++i) {
            detection_label_ += "/" + class_names_[i];
        }
    }

    global_search_distance_ = this->get_parameter("global_search_distance").as_double();
    local_search_distance_ = this->get_parameter("local_search_distance").as_double();
    obstacle_distance_threshold_ = this->get_parameter("pedestrian_distance_threshold").as_double();
    avoid_hold_seconds_ = this->get_parameter("avoid_hold_seconds").as_double();

    yolo_frame_skip_ = this->get_parameter("yolo_frame_skip").as_int();
    publish_annotated_image_ = this->get_parameter("publish_annotated_image").as_bool();
    publish_debug_map_markers_ = this->get_parameter("publish_debug_map_markers").as_bool();
    filter_above_horizon_ = this->get_parameter("filter_above_horizon").as_bool();

    confirm_threshold_ = this->get_parameter("confirm_threshold").as_double();
    tracking_iou_threshold_ = this->get_parameter("tracking_iou_threshold").as_double();
    max_track_age_ = this->get_parameter("max_track_age").as_int();
    decay_factor_ = this->get_parameter("decay_factor").as_double();

    std::string yolo_model_path = this->get_parameter("yolo_model_path").as_string();
    const std::string pkg_share = ament_index_cpp::get_package_share_directory("wireandpipe_detection_cpp");
    if (yolo_model_path.empty()) {
        yolo_model_path = pkg_share + "/6_25.onnx";
        RCLCPP_WARN(this->get_logger(), "No yolo_model_path provided, using default: %s", yolo_model_path.c_str());
    } else if (!std::filesystem::path(yolo_model_path).is_absolute()) {
        yolo_model_path = pkg_share + "/" + yolo_model_path;
    }
    yolo_ = std::make_shared<OrtYoloTracker>(yolo_model_path,
                                             static_cast<float>(conf_threshold_),
                                             this->get_logger());

    // callback groups
    global_plan_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    local_poses_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // QoS
    auto qos_transient_local = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
    auto qos_pose = rclcpp::QoS(10).reliable();

    // ---- 发布者 ----
    pub_avoiding_ = this->create_publisher<std_msgs::msg::Bool>(
        "/is_pipes_and_wires_in_path", qos_transient_local);
    pub_annotated_image_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/rgb_camera_front/annotated_image_wire", 10);
    pub_obstacle_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/obstacles/map_markers", 10);
    pub_downsampled_path_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/downsampled_path/markers", 10);
    pub_raw_path_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/raw_path/markers", 10);
    pub_wire_pipe_3d_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/wire_pipe_markers", 10);
    pub_debug_map_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/wireandpipe/debug_map_markers", 10);

    pub_object_poses_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
        "/detected_objects/poses", qos_pose);

    // ---- 订阅者 ----
    const std::string camera_topic = this->get_parameter("camera_topic").as_string();
    sub_camera_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
        camera_topic,
        rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
            this->imageCallback(msg);
        });

    auto global_plan_sub_options = rclcpp::SubscriptionOptions();
    global_plan_sub_options.callback_group = global_plan_callback_group_;
    sub_global_plan_ = this->create_subscription<nav_msgs::msg::Path>(
        this->get_parameter("global_plan_topic").as_string(), 10,
        [this](const nav_msgs::msg::Path::SharedPtr msg) {
            this->globalPlanCallback(msg);
        }, global_plan_sub_options);

    auto local_poses_sub_options = rclcpp::SubscriptionOptions();
    local_poses_sub_options.callback_group = local_poses_callback_group_;
    sub_local_poses_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
        this->get_parameter("local_poses_topic").as_string(), 10,
        [this](const geometry_msgs::msg::PoseArray::SharedPtr msg) {
            this->localPosesCallback(msg);
        }, local_poses_sub_options);

    // ---- TF ----
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // ---- Timer ----
    timer_ = this->create_wall_timer(100ms, [this]() { this->timerCallback(); });

    auto init_avoiding_msg = std_msgs::msg::Bool();
    init_avoiding_msg.data = false;
    pub_avoiding_->publish(init_avoiding_msg);
    last_published_avoiding_ = false;

    next_track_id_ = 0;

        // ----- 新增：创建手动添加障碍物服务 -----
    add_dummy_obstacle_srv_ = this->create_service<std_srvs::srv::Empty>(
        "add_dummy_obstacle",
        std::bind(&WireAndPipeDetectionNode::addDummyObstacleCallback, this,
                std::placeholders::_1, std::placeholders::_2));
    RCLCPP_INFO(this->get_logger(), "Service 'add_dummy_obstacle' ready.");
    RCLCPP_INFO(this->get_logger(), "WireAndPipeDetectionNode initialized (Pure Monocular Vision Distance Estimation)");
    RCLCPP_INFO(this->get_logger(), "  Camera: %s", camera_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "  Model: %s", yolo_model_path.c_str());
    RCLCPP_INFO(this->get_logger(), "  FOV: h=%.2f° v=%.2f°",
        h_fov_rad_ * 180.0 / M_PI, v_fov_rad_ * 180.0 / M_PI);
    RCLCPP_INFO(this->get_logger(), "  Camera height: %.2f m, Pitch: %.2f rad",
        camera_z_, camera_pitch_);
    RCLCPP_INFO(this->get_logger(), "  Confirm threshold: %.2f, IoU: %.2f, max age: %d",
        confirm_threshold_, tracking_iou_threshold_, max_track_age_);
    RCLCPP_INFO(this->get_logger(), "  Publishing object poses to /detected_objects/poses");
    RCLCPP_INFO(this->get_logger(), "  Publishing wire/pipe 3D markers to /wire_pipe_markers in map frame");
    if (!dist_coeffs_.empty() && cv::norm(dist_coeffs_) > 1e-6) {
        RCLCPP_INFO(this->get_logger(), "  Distortion correction enabled with coefficients: [%.4f, %.4f, %.4f, %.4f, %.4f]",
            dist_coeffs_.at<double>(0,0), dist_coeffs_.at<double>(0,1),
            dist_coeffs_.at<double>(0,2), dist_coeffs_.at<double>(0,3),
            dist_coeffs_.at<double>(0,4));
    } else {
        RCLCPP_INFO(this->get_logger(), "  Distortion correction disabled (zero coefficients)");
    }
}

// ---------------------------------------------------------------------------
// 图像解压
// ---------------------------------------------------------------------------
cv::Mat WireAndPipeDetectionNode::decodeCompressedImage(
    const sensor_msgs::msg::CompressedImage::SharedPtr &msg,
    rclcpp::Time &out_stamp)
{
    out_stamp = msg->header.stamp;
    cv::Mat img = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    if (img.empty()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "[decode] cv::imdecode failed");
    }
    return img;
}

// ---------------------------------------------------------------------------
// 纯视觉地平面测距（修正：添加 out_angle 参数以匹配头文件）
// ---------------------------------------------------------------------------
geometry_msgs::msg::Point WireAndPipeDetectionNode::estimate3DFromLaser(
    const ObbBBox &obb,
    int img_width,
    float &out_dist_laser,
    float &out_angle,               // 新增参数，匹配头文件
    std::string &out_laser_frame_id) const
{
    (void)img_width;
    (void)out_angle;  // 本实现不使用该参数

    geometry_msgs::msg::Point pt;
    pt.x = pt.y = pt.z = 0.0;
    out_dist_laser = std::numeric_limits<float>::quiet_NaN();
    out_laser_frame_id.clear();

    float height = static_cast<float>(camera_z_);
    float pitch = static_cast<float>(camera_pitch_);
    float fy = static_cast<float>(fy_);
    float cy = static_cast<float>(cy_);
    float cx = static_cast<float>(cx_);
    float fx = static_cast<float>(fx_);

    float bottom_y = obb.cy + obb.height * 0.5F;
    float delta_y = bottom_y - cy;
    if (std::abs(delta_y) < 1e-3) {
        return pt;
    }

    float tan_theta = delta_y / fy;
    float denominator = tan_theta * std::cos(pitch) - std::sin(pitch);
    if (std::abs(denominator) < 1e-6) {
        return pt;
    }

    float distance = height / denominator;
    if (distance < 0.2f || distance > 20.0f) {
        return pt;
    }

    float u_center = obb.cx;
    float Xc = (u_center - cx) / fx * distance;

    pt.x = distance;
    pt.y = Xc;
    pt.z = 0.0;

    out_dist_laser = distance;
    out_laser_frame_id = this->get_parameter("rgb_camera_frame").as_string();

    return pt;
}

// ---------------------------------------------------------------------------
// 目标跟踪更新
// ---------------------------------------------------------------------------
void WireAndPipeDetectionNode::updateTracker(const std::vector<ObbTrackItem> &new_detections)
{
    for (auto &target : tracked_targets_) {
        target.age++;
    }

    std::vector<bool> matched_new(new_detections.size(), false);
    std::vector<bool> matched_target(tracked_targets_.size(), false);

    std::vector<int> order_new(new_detections.size());
    std::iota(order_new.begin(), order_new.end(), 0);
    std::sort(order_new.begin(), order_new.end(),
        [&](int a, int b) { return new_detections[a].confidence > new_detections[b].confidence; });

    for (int i_new : order_new) {
        const auto &det = new_detections[i_new];
        int best_target = -1;
        float best_iou = tracking_iou_threshold_;
        for (size_t j = 0; j < tracked_targets_.size(); ++j) {
            if (matched_target[j]) continue;
            if (tracked_targets_[j].class_id != det.class_id) continue;
            float iou = rectIoU(tracked_targets_[j].bbox, det.bbox);
            if (iou > best_iou) {
                best_iou = iou;
                best_target = static_cast<int>(j);
            }
        }
        if (best_target >= 0) {
            matched_new[i_new] = true;
            matched_target[best_target] = true;
            auto &target = tracked_targets_[best_target];
            target.bbox = det.bbox;
            target.confidence = det.confidence;
            target.age = 0;
            target.last_update = std::chrono::steady_clock::now();

            float conf_clamped = std::min(std::max(det.confidence, 0.01f), 0.99f);
            float log_odds = std::log(conf_clamped / (1.0f - conf_clamped));
            target.accumulated_score = decay_factor_ * target.accumulated_score + log_odds;

            if (!target.confirmed && target.accumulated_score >= confirm_threshold_) {
                target.confirmed = true;
                RCLCPP_DEBUG(this->get_logger(), "Target %d confirmed with score %.2f", target.track_id, target.accumulated_score);
            }
        }
    }

    for (size_t i = 0; i < new_detections.size(); ++i) {
        if (matched_new[i]) continue;
        TrackedTarget new_target;
        new_target.class_id = new_detections[i].class_id;
        new_target.bbox = new_detections[i].bbox;
        new_target.confidence = new_detections[i].confidence;
        new_target.age = 0;
        new_target.confirmed = false;
        new_target.track_id = next_track_id_++;
        new_target.last_update = std::chrono::steady_clock::now();
        float conf_clamped = std::min(std::max(new_detections[i].confidence, 0.01f), 0.99f);
        new_target.accumulated_score = std::log(conf_clamped / (1.0f - conf_clamped));
        tracked_targets_.push_back(new_target);
        RCLCPP_DEBUG(this->get_logger(), "New target %d created", new_target.track_id);
    }

    tracked_targets_.erase(
        std::remove_if(tracked_targets_.begin(), tracked_targets_.end(),
            [this](const TrackedTarget &t) { return t.age > max_track_age_; }),
        tracked_targets_.end());
}

// ---------------------------------------------------------------------------
// 图像回调（含畸变校正 + 转换到 map 坐标系）
// ---------------------------------------------------------------------------
void WireAndPipeDetectionNode::imageCallback(
    const sensor_msgs::msg::CompressedImage::SharedPtr msg)
{
    static bool callback_logged_once = false;
    if (!callback_logged_once) {
        RCLCPP_INFO(this->get_logger(), "imageCallback triggered");
        callback_logged_once = true;
    }
    if (!msg || msg->data.empty()) return;

    static int frame_skip_counter = 0;
    if (yolo_frame_skip_ > 1 && ++frame_skip_counter % yolo_frame_skip_ != 0) return;

    rclcpp::Time img_stamp;
    cv::Mat frame = decodeCompressedImage(msg, img_stamp);
    if (frame.empty()) return;

    static bool decoded_logged_once = false;
    if (!decoded_logged_once) {
        RCLCPP_INFO(this->get_logger(), "[imageCallback] Decoded image %dx%d",
            frame.cols, frame.rows);
        decoded_logged_once = true;
    }

    if (!dist_coeffs_.empty() && cv::norm(dist_coeffs_) > 1e-6) {
        cv::Mat undistorted;
        cv::undistort(frame, undistorted, camera_matrix_, dist_coeffs_);
        frame = undistorted;
    }

    std::vector<ObbTrackItem> tracks;
    if (!runYoloTrack(frame, tracks)) return;

    updateTracker(tracks);

    std::vector<TrackedTarget> confirmed_targets;
    for (const auto &target : tracked_targets_) {
        if (target.confirmed) {
            confirmed_targets.push_back(target);
        }
    }

    int obstacle_count = confirmed_targets.size();
    if (obstacle_count == 0) {
        if (!waiting_detect_log_printed_) {
            RCLCPP_INFO(this->get_logger(), "Waiting to detect %s...", detection_label_.c_str());
            waiting_detect_log_printed_ = true;
        }
    } else {
        waiting_detect_log_printed_ = false;
        if (!first_obstacle_detected_logged_) {
            RCLCPP_INFO(this->get_logger(), "%s detected for the first time", detection_label_.c_str());
            first_obstacle_detected_logged_ = true;
        }
    }

    cv::Mat annotated = frame.clone();
    DetectionResult result;
    result.detected = false;
    result.stamp = msg->header.stamp;
    result.laser_frame_id = this->get_parameter("rgb_camera_frame").as_string();
    result.laser_stamp = msg->header.stamp;

    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header.stamp = msg->header.stamp;
    pose_array.header.frame_id = "map";

    std::vector<geometry_msgs::msg::Point> wire_pts_map;
    std::vector<geometry_msgs::msg::Point> pipe_pts_map;

    for (const auto &target : confirmed_targets) {
        const auto &det_bbox = target.bbox;
        float conf = target.confidence;

        if (filter_above_horizon_) {
            const float horizon_y = static_cast<float>(frame.rows) * 0.5F;
            bool all_above = true;
            for (const auto &corner : getRectCorners(det_bbox)) {
                if (corner.y >= horizon_y) {
                    all_above = false;
                    break;
                }
            }
            if (all_above) continue;
        }

        float dist_laser = std::numeric_limits<float>::quiet_NaN();
        std::string laser_frame_id;
        float dummy_angle = 0.0f;   // 占位参数，匹配函数签名
        geometry_msgs::msg::Point pt_laser = estimate3DFromLaser(
            det_bbox, frame.cols, dist_laser, dummy_angle, laser_frame_id);

        bool valid_laser = std::isfinite(dist_laser) &&
                           (std::abs(pt_laser.x) > 1e-3 || std::abs(pt_laser.y) > 1e-3);
        if (valid_laser) {
            result.obstacles_laser.push_back(pt_laser);
            if (!result.laser_frame_id.empty() && !laser_frame_id.empty()) {
                result.laser_frame_id = laser_frame_id;
            }
            result.detected = true;

            geometry_msgs::msg::PointStamped p_in;
            p_in.header.frame_id = laser_frame_id;
            p_in.header.stamp = rclcpp::Time(0, 0, RCL_ROS_TIME);
            p_in.point = pt_laser;

            geometry_msgs::msg::PointStamped p_map;
            try {
                tf_buffer_->transform(p_in, p_map, "map", tf2::durationFromSec(0.05));
                geometry_msgs::msg::Pose pose;
                pose.position = p_map.point;
                pose.orientation.w = 1.0;
                pose_array.poses.push_back(pose);
            } catch (const tf2::TransformException &e) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "[TF] transform to map failed: %s", e.what());
            }

            try {
                geometry_msgs::msg::PointStamped p_map_point;
                tf_buffer_->transform(p_in, p_map_point, "map", tf2::durationFromSec(0.05));
                if (target.class_id == wire_class_id_) {
                    wire_pts_map.push_back(p_map_point.point);
                } else {
                    pipe_pts_map.push_back(p_map_point.point);
                }
            } catch (const tf2::TransformException &e) {
                // 转换失败则忽略
            }
        }

        float dist_to_global = std::numeric_limits<float>::quiet_NaN();
        float dist_to_local  = std::numeric_limits<float>::quiet_NaN();
        if (valid_laser && !laser_frame_id.empty()) {
            geometry_msgs::msg::PointStamped p_in;
            p_in.header.frame_id = laser_frame_id;
            p_in.header.stamp = rclcpp::Time(0, 0, RCL_ROS_TIME);
            p_in.point = pt_laser;
            try {
                geometry_msgs::msg::PointStamped p_map;
                tf_buffer_->transform(p_in, p_map, "map", tf2::durationFromSec(0.05));

                std::vector<geometry_msgs::msg::Point> ds_global_copy;
                std::vector<geometry_msgs::msg::Point> ds_local_copy;
                {
                    std::lock_guard<std::mutex> lock(path_cache_mutex_);
                    ds_global_copy = cached_ds_global_;
                    ds_local_copy  = cached_ds_local_;
                }

                dist_to_global = static_cast<float>(minDistanceToPath(p_map.point, ds_global_copy));
                dist_to_local  = static_cast<float>(minDistanceToPath(p_map.point, ds_local_copy));
            } catch (const tf2::TransformException &e) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "[TF] laser->map transform failed: %s", e.what());
            }
        }

        const int throttle_ms = static_cast<int>(distance_log_throttle_sec_ * 1000.0);
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), throttle_ms,
            "[distance] estimated=%.2fm global_path=%.2fm local_path=%.2fm",
            std::isfinite(dist_laser) ? dist_laser : -1.0f,
            std::isfinite(dist_to_global) ? dist_to_global : -1.0f,
            std::isfinite(dist_to_local) ? dist_to_local : -1.0f);

        cv::Scalar color = (target.class_id == wire_class_id_)
            ? cv::Scalar(255, 0, 0)
            : cv::Scalar(0, 255, 0);

        std::vector<cv::Point> corners_int;
        for (const auto &p : getRectCorners(det_bbox)) {
            corners_int.emplace_back(
                static_cast<int>(std::round(p.x)),
                static_cast<int>(std::round(p.y)));
        }
        const cv::Point *pts = corners_int.data();
        const int npts = static_cast<int>(corners_int.size());
        cv::polylines(annotated, &pts, &npts, 1, true, color, 2);

        const std::string cls_name = (target.class_id >= 0 && target.class_id < static_cast<int>(class_names_.size()))
            ? class_names_[target.class_id]
            : ("class_" + std::to_string(target.class_id));

        std::ostringstream line1;
        line1 << cls_name << " conf:" << std::fixed << std::setprecision(2) << conf
              << " score:" << std::fixed << std::setprecision(2) << target.accumulated_score;

        std::ostringstream line2;
        if (std::isfinite(dist_laser)) {
            line2 << "dist:" << std::fixed << std::setprecision(2) << dist_laser << "m";
        } else {
            line2 << "dist:N/A";
        }

        const double font_scale = 0.6;
        const int thickness = 2;
        const int font = cv::FONT_HERSHEY_SIMPLEX;
        const int line_gap = 20;
        const int text_x = std::max(0, static_cast<int>(det_bbox.cx - det_bbox.width * 0.5F));
        const int text_y = std::max(20, static_cast<int>(det_bbox.cy - det_bbox.height * 0.5F - 5));

        cv::putText(annotated, line1.str(), cv::Point(text_x, text_y),
                    font, font_scale, color, thickness);
        cv::putText(annotated, line2.str(), cv::Point(text_x, text_y + line_gap),
                    font, font_scale, cv::Scalar(0, 255, 255), thickness);
    }

    if (!pose_array.poses.empty()) {
        pub_object_poses_->publish(pose_array);
    }

    // ===== 更新已知障碍物缓存 =====
    {
        std::vector<geometry_msgs::msg::Point> detected_pts;
        detected_pts.insert(detected_pts.end(), wire_pts_map.begin(), wire_pts_map.end());
        detected_pts.insert(detected_pts.end(), pipe_pts_map.begin(), pipe_pts_map.end());

        if (!detected_pts.empty()) {
            std::lock_guard<std::mutex> lock(g_known_obstacles_mutex);
            const double merge_dist_sq = KNOWN_OBSTACLE_MERGE_DIST * KNOWN_OBSTACLE_MERGE_DIST;
            for (const auto &pt : detected_pts) {
                bool found = false;
                for (auto &obs : g_known_obstacles) {
                    double dx = pt.x - obs.pos.x;
                    double dy = pt.y - obs.pos.y;
                    if (dx*dx + dy*dy <= merge_dist_sq) {
                        obs.pos = pt;                 // 更新位置
                        obs.last_seen = msg->header.stamp;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    KnownObstacle new_obs;
                    new_obs.pos = pt;
                    new_obs.last_seen = msg->header.stamp;
                    g_known_obstacles.push_back(new_obs);
                }
            }
        }
    }

    if (pub_wire_pipe_3d_) {
        visualization_msgs::msg::MarkerArray wp_markers;
        int wp_id = 0;

        visualization_msgs::msg::Marker del_all;
        del_all.header.frame_id = "map";
        del_all.header.stamp = msg->header.stamp;
        del_all.ns = "wire_pipe_3d";
        del_all.id = wp_id++;
        del_all.action = visualization_msgs::msg::Marker::DELETEALL;
        wp_markers.markers.push_back(del_all);

        if (!wire_pts_map.empty()) {
            visualization_msgs::msg::Marker wire_m;
            wire_m.header.frame_id = "map";
            wire_m.header.stamp = msg->header.stamp;
            wire_m.ns = "wire_pipe_3d";
            wire_m.id = wp_id++;
            wire_m.type = visualization_msgs::msg::Marker::SPHERE_LIST;
            wire_m.action = visualization_msgs::msg::Marker::ADD;
            wire_m.scale.x = 0.15;
            wire_m.scale.y = 0.15;
            wire_m.scale.z = 0.15;
            wire_m.color.r = 0.0F; wire_m.color.g = 0.4F; wire_m.color.b = 1.0F; wire_m.color.a = 0.9F;
            wire_m.points = wire_pts_map;
            wp_markers.markers.push_back(wire_m);
        }

        if (!pipe_pts_map.empty()) {
            visualization_msgs::msg::Marker pipe_m;
            pipe_m.header.frame_id = "map";
            pipe_m.header.stamp = msg->header.stamp;
            pipe_m.ns = "wire_pipe_3d";
            pipe_m.id = wp_id++;
            pipe_m.type = visualization_msgs::msg::Marker::SPHERE_LIST;
            pipe_m.action = visualization_msgs::msg::Marker::ADD;
            pipe_m.scale.x = 0.15;
            pipe_m.scale.y = 0.15;
            pipe_m.scale.z = 0.15;
            pipe_m.color.r = 0.0F; pipe_m.color.g = 0.9F; pipe_m.color.b = 0.0F; pipe_m.color.a = 0.9F;
            pipe_m.points = pipe_pts_map;
            wp_markers.markers.push_back(pipe_m);
        }

        pub_wire_pipe_3d_->publish(wp_markers);
    }

    if (pub_annotated_image_ && publish_annotated_image_) {
        auto annotated_msg = sensor_msgs::msg::Image();
        annotated_msg.header = msg->header;
        annotated_msg.height = static_cast<uint32_t>(annotated.rows);
        annotated_msg.width = static_cast<uint32_t>(annotated.cols);
        annotated_msg.encoding = sensor_msgs::image_encodings::BGR8;
        annotated_msg.is_bigendian = 0;
        annotated_msg.step = static_cast<sensor_msgs::msg::Image::_step_type>(
            annotated.cols * annotated.elemSize());
        const size_t image_bytes = static_cast<size_t>(annotated_msg.step)
                                   * static_cast<size_t>(annotated.rows);
        annotated_msg.data.assign(annotated.data, annotated.data + image_bytes);
        pub_annotated_image_->publish(annotated_msg);
    }

    {
        std::lock_guard<std::mutex> lock(detection_mutex_);
        detection_result_ = std::move(result);
    }
}

// ---------------------------------------------------------------------------
bool WireAndPipeDetectionNode::runYoloTrack(const cv::Mat &frame,
                                            std::vector<ObbTrackItem> &tracks)
{
    if (!yolo_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "YOLO tracker is not initialized.");
        return false;
    }
    try {
        tracks = yolo_->track(frame);
        return true;
    } catch (const std::exception &e) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "YOLO inference failed: %s", e.what());
        return false;
    } catch (...) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "YOLO inference failed with unknown exception.");
        return false;
    }
}

// ---------------------------------------------------------------------------
// 全局路径回调（转换到 map 坐标系）
// ---------------------------------------------------------------------------
void WireAndPipeDetectionNode::globalPlanCallback(const nav_msgs::msg::Path::SharedPtr msg)
{
    if (!msg) return;

    // 转换到 map 坐标系
    std::vector<geometry_msgs::msg::Point> global_points_map;
    global_points_map.reserve(msg->poses.size());

    for (const auto &pose_stamped : msg->poses) {
        geometry_msgs::msg::PointStamped p_in;
        p_in.header = pose_stamped.header;
        p_in.point = pose_stamped.pose.position;
        try {
            geometry_msgs::msg::PointStamped p_map;
            tf_buffer_->transform(p_in, p_map, "map", tf2::durationFromSec(0.5));
            global_points_map.push_back(p_map.point);
        } catch (const tf2::TransformException &e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[globalPlan] transform to map failed: %s", e.what());
            // 回退：使用原始坐标（可能不准确）
            global_points_map.push_back(pose_stamped.pose.position);
        }
    }

    {
        std::lock_guard<std::mutex> lock(path_cache_mutex_);
        cached_global_raw_ = global_points_map;
    }

    last_global_plan_ = msg;
    global_plan_dirty_ = true;

    static bool once = true;
    if (once) {
        RCLCPP_INFO(this->get_logger(), "globalPlanCallback: received path with %zu poses", msg->poses.size());
        once = false;
    }

    // 发布原始路径标记（map 坐标系）
    if (pub_raw_path_markers_ && !global_points_map.empty()) {
        visualization_msgs::msg::MarkerArray raw_markers;
        int mid = 0;

        visualization_msgs::msg::Marker del;
        del.header.frame_id = "map";
        del.header.stamp = msg->header.stamp;
        del.ns = "raw_global_path";
        del.id = mid++;
        del.action = visualization_msgs::msg::Marker::DELETEALL;
        raw_markers.markers.push_back(del);

        visualization_msgs::msg::Marker line;
        line.header.frame_id = "map";
        line.header.stamp = msg->header.stamp;
        line.ns = "raw_global_path";
        line.id = mid++;
        line.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line.action = visualization_msgs::msg::Marker::ADD;
        line.scale.x = 0.05;
        line.color.r = 0.2F; line.color.g = 0.6F; line.color.b = 1.0F; line.color.a = 0.7F;
        line.points = global_points_map;
        raw_markers.markers.push_back(line);

        pub_raw_path_markers_->publish(raw_markers);
    }
}

// ---------------------------------------------------------------------------
// 局部路径回调（转换到 map 坐标系）
// ---------------------------------------------------------------------------
void WireAndPipeDetectionNode::localPosesCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
{
    if (!msg) return;

    std::vector<geometry_msgs::msg::Point> local_points_map;
    local_points_map.reserve(msg->poses.size());

    for (const auto &pose : msg->poses) {
        geometry_msgs::msg::PointStamped p_in;
        p_in.header.frame_id = msg->header.frame_id;
        p_in.header.stamp = msg->header.stamp;
        p_in.point = pose.position;
        try {
            geometry_msgs::msg::PointStamped p_map;
            tf_buffer_->transform(p_in, p_map, "map", tf2::durationFromSec(0.5));
            local_points_map.push_back(p_map.point);
        } catch (const tf2::TransformException &e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[localPoses] transform to map failed: %s", e.what());
            local_points_map.push_back(pose.position);
        }
    }

    {
        std::lock_guard<std::mutex> lock(path_cache_mutex_);
        cached_local_raw_ = local_points_map;
    }

    last_local_poses_ = msg;
    local_poses_dirty_ = true;

    static bool once = true;
    if (once) {
        RCLCPP_INFO(this->get_logger(), "localPosesCallback: received %zu poses", msg->poses.size());
        once = false;
    }

    if (pub_raw_path_markers_ && !local_points_map.empty()) {
        visualization_msgs::msg::MarkerArray raw_markers;
        int mid = 0;

        visualization_msgs::msg::Marker del;
        del.header.frame_id = "map";
        del.header.stamp = msg->header.stamp;
        del.ns = "raw_local_path";
        del.id = mid++;
        del.action = visualization_msgs::msg::Marker::DELETEALL;
        raw_markers.markers.push_back(del);

        visualization_msgs::msg::Marker line;
        line.header.frame_id = "map";
        line.header.stamp = msg->header.stamp;
        line.ns = "raw_local_path";
        line.id = mid++;
        line.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line.action = visualization_msgs::msg::Marker::ADD;
        line.scale.x = 0.03;
        line.color.r = 1.0F; line.color.g = 0.65F; line.color.b = 0.0F; line.color.a = 0.7F;
        line.points = local_points_map;
        raw_markers.markers.push_back(line);

        pub_raw_path_markers_->publish(raw_markers);
    }
}

// ---------------------------------------------------------------------------
// Timer 回调（核心避让逻辑，基于缓存障碍物）
// ---------------------------------------------------------------------------
void WireAndPipeDetectionNode::timerCallback()
{
    bool trigger_now = false;
    double hit_dist = 0.0;
    const double threshold = obstacle_distance_threshold_;
    const auto now_stamp = this->now();

    // 更新下采样路径（若脏）
    if (last_global_plan_ || last_local_poses_) {
        if (global_plan_dirty_ && !cached_global_raw_.empty()) {
            std::lock_guard<std::mutex> lock(path_cache_mutex_);
            cached_ds_global_ = downsamplePath(cached_global_raw_, threshold, global_search_distance_);
            global_plan_dirty_ = false;
        }
        if (local_poses_dirty_ && !cached_local_raw_.empty()) {
            std::lock_guard<std::mutex> lock(path_cache_mutex_);
            cached_ds_local_ = downsamplePath(cached_local_raw_, threshold, local_search_distance_);
            local_poses_dirty_ = false;
        }

        const auto &ds_global = cached_ds_global_;
        const auto &ds_local = cached_ds_local_;

        // 发布下采样路径标记（保持不变）
        if (pub_downsampled_path_markers_) {
            visualization_msgs::msg::MarkerArray ds_marker_array;
            int marker_id = 0;

            visualization_msgs::msg::Marker delete_all;
            delete_all.header.frame_id = "map";
            delete_all.header.stamp = now_stamp;
            delete_all.ns = "downsampled_path";
            delete_all.id = marker_id++;
            delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
            ds_marker_array.markers.push_back(delete_all);

            for (const auto &pt : ds_global) {
                visualization_msgs::msg::Marker m;
                m.header.frame_id = "map";
                m.header.stamp = now_stamp;
                m.ns = "downsampled_path";
                m.id = marker_id++;
                m.type = visualization_msgs::msg::Marker::SPHERE;
                m.action = visualization_msgs::msg::Marker::ADD;
                m.pose.position = pt;
                m.pose.position.z = 0.05;
                m.pose.orientation.w = 1.0;
                m.scale.x = 0.12; m.scale.y = 0.12; m.scale.z = 0.12;
                m.color.r = 0.0F; m.color.g = 0.0F; m.color.b = 1.0F; m.color.a = 0.8F;
                ds_marker_array.markers.push_back(m);
            }
            for (const auto &pt : ds_local) {
                visualization_msgs::msg::Marker m;
                m.header.frame_id = "map";
                m.header.stamp = now_stamp;
                m.ns = "downsampled_path";
                m.id = marker_id++;
                m.type = visualization_msgs::msg::Marker::SPHERE;
                m.action = visualization_msgs::msg::Marker::ADD;
                m.pose.position = pt;
                m.pose.position.z = 0.05;
                m.pose.orientation.w = 1.0;
                m.scale.x = 0.12; m.scale.y = 0.12; m.scale.z = 0.12;
                m.color.r = 1.0F; m.color.g = 1.0F; m.color.b = 0.0F; m.color.a = 0.8F;
                ds_marker_array.markers.push_back(m);
            }
            pub_downsampled_path_markers_->publish(ds_marker_array);
        }

        // ----- 基于缓存障碍物的避让判断 -----
        {
            std::lock_guard<std::mutex> lock(g_known_obstacles_mutex);
            // 清理远离路径且超时的障碍物
            auto it = g_known_obstacles.begin();
            while (it != g_known_obstacles.end()) {
                double dist_global = minDistanceToPath(it->pos, ds_global);
                double dist_local  = minDistanceToPath(it->pos, ds_local);
                double min_dist = std::min(dist_global, dist_local);
                if (!std::isfinite(min_dist)) min_dist = std::numeric_limits<double>::max();
                double dt = (now_stamp - it->last_seen).seconds();
                if (min_dist > threshold && dt > KNOWN_OBSTACLE_TIMEOUT) {
                    it = g_known_obstacles.erase(it);
                } else {
                    ++it;
                }
            }

            // 检查缓存中是否有障碍物在路径附近
            for (const auto &obs : g_known_obstacles) {
                double dist_global = minDistanceToPath(obs.pos, ds_global);
                double dist_local  = minDistanceToPath(obs.pos, ds_local);
                double min_dist = std::min(dist_global, dist_local);
                if (std::isfinite(min_dist) && min_dist <= threshold) {
                    trigger_now = true;
                    hit_dist = min_dist;
                    break;
                }
            }
        }

        // 根据 trigger_now 更新计时器和状态
        if (trigger_now) {
            last_obstacle_detect_time_ = now_stamp;
            has_recent_detection_ = true;
            if (!last_trigger_state_) {
                RCLCPP_WARN(this->get_logger(),
                    "[timer] Obstacle detected on path, distance to path: %.2fm", hit_dist);
            }
        } else {
            if (has_recent_detection_) {
                double dt = (now_stamp - last_obstacle_detect_time_).seconds();
                if (dt >= avoid_hold_seconds_) {
                    has_recent_detection_ = false;
                    RCLCPP_INFO(this->get_logger(),
                        "[timer] Path clear, obstacle no longer in range");
                }
            }
        }

        // 发布避让状态
        bool is_avoiding = has_recent_detection_;
        if (is_avoiding && !last_published_avoiding_) {
            auto out = std_msgs::msg::Bool();
            out.data = true;
            pub_avoiding_->publish(out);
            ++warning_event_id_;
            RCLCPP_WARN(this->get_logger(), "%s obstacle warning, id: %llu",
                detection_label_.c_str(),
                static_cast<unsigned long long>(warning_event_id_));
            last_published_avoiding_ = true;
        } else if (!is_avoiding && last_published_avoiding_) {
            auto out = std_msgs::msg::Bool();
            out.data = false;
            pub_avoiding_->publish(out);
            RCLCPP_INFO(this->get_logger(), "%s obstacle warning cleared, id: %llu",
                detection_label_.c_str(),
                static_cast<unsigned long long>(warning_event_id_));
            last_published_avoiding_ = false;
        }
    }

    // ----- 以下代码保留原障碍物标记发布（使用 detection_result_，仅用于可视化） -----
    DetectionResult det;
    {
        std::lock_guard<std::mutex> lock(detection_mutex_);
        det = detection_result_;
    }

    if (det.detected && !det.obstacles_laser.empty() && !det.laser_frame_id.empty()) {
        std::vector<geometry_msgs::msg::PointStamped> obstacles_map;
        obstacles_map.reserve(det.obstacles_laser.size());

        for (const auto &pt_laser : det.obstacles_laser) {
            geometry_msgs::msg::PointStamped p_in;
            p_in.header.frame_id = det.laser_frame_id;
            p_in.header.stamp = rclcpp::Time(0, 0, RCL_ROS_TIME);
            p_in.point = pt_laser;
            try {
                geometry_msgs::msg::PointStamped p_map;
                tf_buffer_->transform(p_in, p_map, "map", tf2::durationFromSec(0.2));
                obstacles_map.push_back(p_map);
            } catch (const tf2::TransformException &e) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                    "[TF] laser->map transform failed: %s", e.what());
            }
        }

        if (!obstacles_map.empty()) {
            visualization_msgs::msg::MarkerArray marker_array;
            marker_array.markers.reserve(obstacles_map.size() + 1);

            visualization_msgs::msg::Marker delete_all;
            delete_all.header.frame_id = "map";
            delete_all.header.stamp = now_stamp;
            delete_all.ns = "obstacles";
            delete_all.id = 0;
            delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
            marker_array.markers.push_back(delete_all);

            for (size_t i = 0; i < obstacles_map.size(); ++i) {
                const auto &p = obstacles_map[i];
                visualization_msgs::msg::Marker marker;
                marker.header.frame_id = "map";
                marker.header.stamp = now_stamp;
                marker.ns = "obstacles";
                marker.id = static_cast<int>(i) + 1;
                marker.type = visualization_msgs::msg::Marker::SPHERE;
                marker.action = visualization_msgs::msg::Marker::ADD;
                marker.pose.position = p.point;
                marker.pose.orientation.w = 1.0;
                marker.scale.x = 0.25; marker.scale.y = 0.25; marker.scale.z = 0.25;
                marker.color.r = 1.0F; marker.color.g = 0.3F; marker.color.b = 0.0F;
                marker.color.a = 0.9F;
                marker_array.markers.push_back(marker);
            }
            if (pub_obstacle_markers_) pub_obstacle_markers_->publish(marker_array);

            if (pub_debug_map_markers_) {
                visualization_msgs::msg::MarkerArray debug_markers;

                visualization_msgs::msg::Marker del_debug;
                del_debug.header.frame_id = "map";
                del_debug.header.stamp = now_stamp;
                del_debug.ns = "wireandpipe_debug";
                del_debug.id = 0;
                del_debug.action = visualization_msgs::msg::Marker::DELETEALL;
                debug_markers.markers.push_back(del_debug);

                int debug_id = 1;

                std::vector<geometry_msgs::msg::Point> ds_global, ds_local;
                {
                    std::lock_guard<std::mutex> lock(path_cache_mutex_);
                    ds_global = cached_ds_global_;
                    ds_local = cached_ds_local_;
                }
                for (const auto &pt : ds_global) {
                    visualization_msgs::msg::Marker m;
                    m.header.frame_id = "map";
                    m.header.stamp = now_stamp;
                    m.ns = "wireandpipe_debug";
                    m.id = debug_id++;
                    m.type = visualization_msgs::msg::Marker::SPHERE;
                    m.action = visualization_msgs::msg::Marker::ADD;
                    m.pose.position = pt;
                    m.pose.position.z = 0.05;
                    m.pose.orientation.w = 1.0;
                    m.scale.x = 0.12; m.scale.y = 0.12; m.scale.z = 0.12;
                    m.color.r = 0.0F; m.color.g = 0.0F; m.color.b = 1.0F; m.color.a = 0.8F;
                    debug_markers.markers.push_back(m);
                }
                for (const auto &pt : ds_local) {
                    visualization_msgs::msg::Marker m;
                    m.header.frame_id = "map";
                    m.header.stamp = now_stamp;
                    m.ns = "wireandpipe_debug";
                    m.id = debug_id++;
                    m.type = visualization_msgs::msg::Marker::SPHERE;
                    m.action = visualization_msgs::msg::Marker::ADD;
                    m.pose.position = pt;
                    m.pose.position.z = 0.05;
                    m.pose.orientation.w = 1.0;
                    m.scale.x = 0.12; m.scale.y = 0.12; m.scale.z = 0.12;
                    m.color.r = 1.0F; m.color.g = 1.0F; m.color.b = 0.0F; m.color.a = 0.8F;
                    debug_markers.markers.push_back(m);
                }

                for (const auto &p : obstacles_map) {
                    visualization_msgs::msg::Marker m;
                    m.header.frame_id = "map";
                    m.header.stamp = now_stamp;
                    m.ns = "wireandpipe_debug";
                    m.id = debug_id++;
                    m.type = visualization_msgs::msg::Marker::CUBE;
                    m.action = visualization_msgs::msg::Marker::ADD;
                    m.pose.position = p.point;
                    m.pose.position.z = 0.05;
                    m.pose.orientation.w = 1.0;
                    m.scale.x = 0.30; m.scale.y = 0.30; m.scale.z = 0.05;
                    m.color.r = 1.0F; m.color.g = 0.0F; m.color.b = 0.0F; m.color.a = 0.8F;
                    debug_markers.markers.push_back(m);
                }

                pub_debug_map_markers_->publish(debug_markers);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 路径下采样
// ---------------------------------------------------------------------------
std::vector<geometry_msgs::msg::Point> WireAndPipeDetectionNode::downsamplePath(
    const std::vector<geometry_msgs::msg::Point> &nav_points,
    double min_distance,
    double lookahead_distance)
{
    std::vector<geometry_msgs::msg::Point> out_points;
    if (nav_points.empty()) return out_points;

    if (nav_points.size() >= 2) {
        const double end_dx = nav_points.back().x - nav_points.front().x;
        const double end_dy = nav_points.back().y - nav_points.front().y;
        const double end_dist = std::sqrt(end_dx * end_dx + end_dy * end_dy);
        if (end_dist < lookahead_distance) return nav_points;
    }

    out_points.push_back(nav_points.front());
    if (nav_points.size() == 1) return out_points;

    const double sample_dist_sq = min_distance * min_distance;
    size_t last_keep_idx = 0;
    double accumulated_len = 0.0;

    for (size_t i = 1; i < nav_points.size(); ++i) {
        const double seg_dx = nav_points[i].x - nav_points[i - 1].x;
        const double seg_dy = nav_points[i].y - nav_points[i - 1].y;
        const double seg_len = std::sqrt(seg_dx * seg_dx + seg_dy * seg_dy);
        if (accumulated_len + seg_len > lookahead_distance) break;
        accumulated_len += seg_len;

        const double dx = nav_points[i].x - nav_points[last_keep_idx].x;
        const double dy = nav_points[i].y - nav_points[last_keep_idx].y;
        const double dist_sq = dx * dx + dy * dy;
        if (dist_sq >= sample_dist_sq) {
            out_points.push_back(nav_points[i]);
            last_keep_idx = i;
        }
    }
    return out_points;
}

// ---------------------------------------------------------------------------
// 点到路径最短距离
// ---------------------------------------------------------------------------
double WireAndPipeDetectionNode::minDistanceToPath(
    const geometry_msgs::msg::Point &point,
    const std::vector<geometry_msgs::msg::Point> &path) const
{
    if (path.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double min_dist_sq = std::numeric_limits<double>::max();
    for (const auto &p : path) {
        const double dx = point.x - p.x;
        const double dy = point.y - p.y;
        const double dist_sq = dx * dx + dy * dy;
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
        }
    }
    return std::sqrt(min_dist_sq);
}

WireAndPipeDetectionNode::~WireAndPipeDetectionNode() = default;

// ---------------------------------------------------------------------------
// 获取路径前方 distance_meters 处的点（沿路径方向，map 坐标系）
// ---------------------------------------------------------------------------
geometry_msgs::msg::Point WireAndPipeDetectionNode::getPointOnPathAhead(double distance_meters)
{
    geometry_msgs::msg::Point result;
    result.x = result.y = result.z = 0.0;

    // 获取当前全局路径（map 坐标系）
    std::vector<geometry_msgs::msg::Point> global_path;
    {
        std::lock_guard<std::mutex> lock(path_cache_mutex_);
        global_path = cached_global_raw_;   // 原始路径点（已转换到 map）
    }
    if (global_path.empty()) {
        RCLCPP_WARN(this->get_logger(), "No global path available, cannot compute ahead point.");
        return result;
    }

    // 获取机器人当前位置（map 坐标系）
    geometry_msgs::msg::TransformStamped robot_transform;
    try {
        robot_transform = tf_buffer_->lookupTransform("map", "base_link", rclcpp::Time(0));
    } catch (const tf2::TransformException &e) {
        RCLCPP_ERROR(this->get_logger(), "Cannot get robot transform: %s", e.what());
        return result;
    }
    geometry_msgs::msg::Point robot_pos;
    robot_pos.x = robot_transform.transform.translation.x;
    robot_pos.y = robot_transform.transform.translation.y;
    robot_pos.z = robot_transform.transform.translation.z;

    // 找到路径上离机器人最近的点
    double min_dist_sq = std::numeric_limits<double>::max();
    size_t nearest_idx = 0;
    for (size_t i = 0; i < global_path.size(); ++i) {
        double dx = global_path[i].x - robot_pos.x;
        double dy = global_path[i].y - robot_pos.y;
        double dist_sq = dx*dx + dy*dy;
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            nearest_idx = i;
        }
    }

    // 从最近点开始，沿路径向前累加距离
    double accumulated = 0.0;
    for (size_t i = nearest_idx; i + 1 < global_path.size(); ++i) {
        double dx = global_path[i+1].x - global_path[i].x;
        double dy = global_path[i+1].y - global_path[i].y;
        double seg_len = std::sqrt(dx*dx + dy*dy);
        if (accumulated + seg_len >= distance_meters) {
            // 插值出精确点
            double ratio = (distance_meters - accumulated) / seg_len;
            result.x = global_path[i].x + ratio * (global_path[i+1].x - global_path[i].x);
            result.y = global_path[i].y + ratio * (global_path[i+1].y - global_path[i].y);
            result.z = 0.0;
            return result;
        }
        accumulated += seg_len;
    }

    // 如果路径总长度不足，取终点
    if (!global_path.empty()) {
        result = global_path.back();
    }
    return result;
}

// ---------------------------------------------------------------------------
// 服务回调：手动添加虚拟障碍物（路径前方 1m）
// ---------------------------------------------------------------------------
void WireAndPipeDetectionNode::addDummyObstacleCallback(
    const std_srvs::srv::Empty::Request::SharedPtr /*req*/,
    std_srvs::srv::Empty::Response::SharedPtr /*res*/)
{
    const double AHEAD_DIST = 1.0;   // 前方 1 米
    geometry_msgs::msg::Point dummy_pt = getPointOnPathAhead(AHEAD_DIST);
    if (dummy_pt.x == 0.0 && dummy_pt.y == 0.0) {
        RCLCPP_WARN(this->get_logger(), "Could not compute dummy point, abort.");
        return;
    }

    // 添加到全局缓存（与图像检测相同的机制）
    {
        std::lock_guard<std::mutex> lock(g_known_obstacles_mutex);
        // 检查是否已存在相近点（避免重复）
        bool exists = false;
        for (const auto &obs : g_known_obstacles) {
            double dx = obs.pos.x - dummy_pt.x;
            double dy = obs.pos.y - dummy_pt.y;
            if (dx*dx + dy*dy < 0.1*0.1) { // 10cm 内视为已存在
                exists = true;
                break;
            }
        }
        if (!exists) {
            KnownObstacle new_obs;
            new_obs.pos = dummy_pt;
            new_obs.last_seen = this->now();  // 当前时间
            g_known_obstacles.push_back(new_obs);
            RCLCPP_INFO(this->get_logger(), "Dummy obstacle added at (%.2f, %.2f)", 
                        dummy_pt.x, dummy_pt.y);
        } else {
            RCLCPP_INFO(this->get_logger(), "Dummy point already exists in cache.");
        }
    }
    // timerCallback 会在下一次周期自动检测此点并触发避让
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<WireAndPipeDetectionNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
