#ifndef WIREANDPIPE_DETECTION_CPP__WIREANDPIPE_DETECTION_HPP_
#define WIREANDPIPE_DETECTION_CPP__WIREANDPIPE_DETECTION_HPP_

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <opencv2/core.hpp>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <std_srvs/srv/empty.hpp> 

// ...（其他已有包含）

struct ObbBBox
{
    float cx, cy, width, height, angle;
};

struct ObbTrackItem
{
    int class_id;
    float confidence;
    ObbBBox bbox;
};

struct TrackedTarget
{
    int class_id;
    ObbBBox bbox;
    float confidence;
    int age{0};
    bool confirmed{false};
    int track_id{-1};
    float accumulated_score{0.0f};
    std::chrono::steady_clock::time_point last_update;
};

struct DetectionResult
{
    bool detected{false};
    rclcpp::Time stamp;
    std::string laser_frame_id;
    rclcpp::Time laser_stamp;
    std::vector<geometry_msgs::msg::Point> obstacles_laser;
    std::vector<float> obstacle_heights;
    // ---- 新增 ----
    std::vector<geometry_msgs::msg::Point> wire_pts_map;
    std::vector<geometry_msgs::msg::Point> pipe_pts_map;
};

class ObbYoloTracker
{
public:
    virtual ~ObbYoloTracker() = default;
    virtual std::vector<ObbTrackItem> track(const cv::Mat &frame) = 0;
};

class WireAndPipeDetectionNode : public rclcpp::Node
{
public:
    WireAndPipeDetectionNode();
    ~WireAndPipeDetectionNode() override;   // 新增析构

private:
    // ---- 回调函数 ----
    void imageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
    void globalPlanCallback(const nav_msgs::msg::Path::SharedPtr msg);
    void localPosesCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg);
    void timerCallback();

    // ---- 辅助函数 ----
    cv::Mat decodeCompressedImage(const sensor_msgs::msg::CompressedImage::SharedPtr &msg,
                                  rclcpp::Time &out_stamp);
    geometry_msgs::msg::Point estimate3DFromLaser(const ObbBBox &obb,
                                                  int img_width,
                                                  float &out_dist_laser,
                                                  float &out_height,
                                                  std::string &out_laser_frame_id) const;
    void updateTracker(const std::vector<ObbTrackItem> &new_detections);
    bool runYoloTrack(const cv::Mat &frame, std::vector<ObbTrackItem> &tracks);
    std::vector<geometry_msgs::msg::Point> downsamplePath(
        const std::vector<geometry_msgs::msg::Point> &nav_points,
        double min_distance,
        double lookahead_distance);
    bool checkObstacleOnGlobalPath(
        const std::vector<geometry_msgs::msg::PointStamped> &obstacles,
        const std::vector<geometry_msgs::msg::Point> &downsampled_points,
        double threshold,
        double *hit_distance);
    bool checkObstacleOnLocalPath(
        const std::vector<geometry_msgs::msg::PointStamped> &obstacles,
        const std::vector<geometry_msgs::msg::Point> &downsampled_points,
        double threshold,
        double *hit_distance);
    double minDistanceToPath(const geometry_msgs::msg::Point &point,
                             const std::vector<geometry_msgs::msg::Point> &path) const;

    // ---- 新增线程处理函数 ----
    void processYoloThread();

    // ---- ROS2 成员 ----
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_camera_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_global_plan_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr sub_local_poses_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_avoiding_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_annotated_image_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_obstacle_markers_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_downsampled_path_markers_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_raw_path_markers_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_wire_pipe_3d_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_debug_map_markers_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_object_poses_;

    rclcpp::TimerBase::SharedPtr timer_;

    rclcpp::CallbackGroup::SharedPtr global_plan_callback_group_;
    rclcpp::CallbackGroup::SharedPtr local_poses_callback_group_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // ---- 参数变量 ----
    double fx_, fy_, cx_, cy_;
    int camera_width_, camera_height_;
    double h_fov_rad_, v_fov_rad_;
    double camera_x_, camera_y_, camera_z_, camera_pitch_, camera_yaw_, camera_roll_;
    std::vector<std::string> class_names_;
    int wire_class_id_, water_pipe_class_id_;
    double conf_threshold_;
    double distance_log_throttle_sec_;
    double global_search_distance_;
    double local_search_distance_;
    double obstacle_distance_threshold_;
    double avoid_hold_seconds_;
    int yolo_frame_skip_;
    bool publish_annotated_image_;
    bool publish_debug_map_markers_;
    bool filter_above_horizon_;
    double confirm_threshold_;
    double tracking_iou_threshold_;
    int max_track_age_;
    double decay_factor_;
    double collision_time_threshold_;

    // ---- 新增参数 ----
    double no_path_distance_threshold_;
    double marker_pub_interval_;

    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;

    std::shared_ptr<ObbYoloTracker> yolo_;

    std::vector<TrackedTarget> tracked_targets_;
    int next_track_id_{0};

    DetectionResult detection_result_;
    std::mutex detection_mutex_;

    // 路径缓存
    std::vector<geometry_msgs::msg::Point> cached_global_raw_;
    std::vector<geometry_msgs::msg::Point> cached_local_raw_;
    std::vector<geometry_msgs::msg::Point> cached_ds_global_;
    std::vector<geometry_msgs::msg::Point> cached_ds_local_;
    std::mutex path_cache_mutex_;
    bool global_plan_dirty_{false};
    bool local_poses_dirty_{false};
    nav_msgs::msg::Path::SharedPtr last_global_plan_;
    geometry_msgs::msg::PoseArray::SharedPtr last_local_poses_;

    bool last_trigger_state_{false};
    bool has_recent_detection_{false};
    rclcpp::Time last_obstacle_detect_time_;
    std::string detection_label_;
    bool waiting_detect_log_printed_{false};
    bool first_obstacle_detected_logged_{false};
    bool last_global_match_{false};
    bool last_local_match_{false};
    unsigned long long warning_event_id_{0};
    bool last_published_avoiding_{false};

    // 速度
    geometry_msgs::msg::Twist current_vel_;
    std::mutex vel_mutex_;

    // ---- 异步处理成员 ----
    std::thread yolo_thread_;
    std::atomic<bool> yolo_thread_running_{true};
    std::queue<sensor_msgs::msg::CompressedImage::SharedPtr> image_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    rclcpp::Time last_marker_pub_time_;

        // ===== 新增：手动添加虚拟障碍物服务 =====
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr add_dummy_obstacle_srv_;
    void addDummyObstacleCallback(
        const std_srvs::srv::Empty::Request::SharedPtr req,
        std_srvs::srv::Empty::Response::SharedPtr res);

    // 辅助：获取路径前方 distance_meters 处的点（map 坐标系）
    geometry_msgs::msg::Point getPointOnPathAhead(double distance_meters);
};

#endif  // WIREANDPIPE_DETECTION_CPP__WIREANDPIPE_DETECTION_HPP_
