#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <Eigen/Dense>
#include <string>

#include "io_utils.hpp"

using std::placeholders::_1;

class ColorLidarNode : public rclcpp::Node {
public:
  ColorLidarNode() : Node("color_lidar_node") {
    const std::string pkg_share =
        ament_index_cpp::get_package_share_directory(
            "livox_camera_lidar_calibration_ros2");

    cloud_topic_ = declare_parameter<std::string>(
        "cloud_topic", "/livox/points");
    image_topic_ = declare_parameter<std::string>(
        "image_topic", "/camera/image_raw");

    intr_file_ = declare_parameter<std::string>(
        "intrinsic_file", pkg_share + "/config/intrinsic.txt");
    extr_file_ = declare_parameter<std::string>(
        "extrinsic_file", pkg_share + "/config/extrinsic.txt");

    out_colored_ = declare_parameter<std::string>(
        "output_colored_pcd", pkg_share + "/data/pcdFiles/colored.pcd");

    if (!load_intrinsics_txt(intr_file_, K_)) {
      RCLCPP_FATAL(get_logger(), "Load intrinsics failed: %s",
                   intr_file_.c_str());
      rclcpp::shutdown();
      return;
    }
    if (!load_extrinsic_txt(extr_file_, T_lc_)) {
      RCLCPP_FATAL(get_logger(), "Load extrinsic failed: %s",
                   extr_file_.c_str());
      rclcpp::shutdown();
      return;
    }

    sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_topic_, rclcpp::SensorDataQoS(),
        std::bind(&ColorLidarNode::onCloud, this, _1));
    sub_img_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic_, rclcpp::SensorDataQoS(),
        std::bind(&ColorLidarNode::onImage, this, _1));

    RCLCPP_INFO(get_logger(),
                "intrinsic_file=%s, extrinsic_file=%s, output=%s",
                intr_file_.c_str(), extr_file_.c_str(), out_colored_.c_str());
  }

private:
  void onImage(const sensor_msgs::msg::Image::SharedPtr msg) { last_img_ = msg; }

  void onCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!last_img_) return;

    cv::Mat img = cv_bridge::toCvShare(last_img_, "bgr8")->image;

    pcl::PointCloud<pcl::PointXYZRGB> out;
    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::fromROSMsg(*msg, cloud);
    out.points.reserve(cloud.size());

    const Eigen::Matrix3d R = T_lc_.block<3,3>(0,0);
    const Eigen::Vector3d t = T_lc_.block<3,1>(0,3);

    for (const auto& p : cloud.points) {
      Eigen::Vector3d Pc = R * Eigen::Vector3d(p.x, p.y, p.z) + t;

      pcl::PointXYZRGB q;
      q.x = p.x; q.y = p.y; q.z = p.z;

      if (Pc.z() > 0.05) {
        cv::Point2d uv = project_cam(K_, Pc);
        if (uv.x >= 0 && uv.y >= 0 && uv.x < img.cols && uv.y < img.rows) {
          const cv::Vec3b c = img.at<cv::Vec3b>(static_cast<int>(uv.y),
                                                static_cast<int>(uv.x));
          q.r = c[2]; q.g = c[1]; q.b = c[0];
        } else {
          q.r = q.g = q.b = 128;
        }
      } else {
        q.r = q.g = q.b = 0;
      }
      out.points.push_back(q);
    }
    out.width = static_cast<uint32_t>(out.points.size());
    out.height = 1;

    if (pcl::io::savePCDFileBinary(out_colored_, out) == 0) {
      RCLCPP_INFO(get_logger(), "Saved colored PCD: %s (%u pts)",
                  out_colored_.c_str(), out.width);
    } else {
      RCLCPP_WARN(get_logger(), "Failed to save colored PCD: %s",
                  out_colored_.c_str());
    }
  }

  // params & I/O
  std::string cloud_topic_, image_topic_;
  std::string intr_file_, extr_file_, out_colored_;

  // subs
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_;

  // cache
  sensor_msgs::msg::Image::SharedPtr last_img_;
  CameraIntrinsics K_;
  Eigen::Matrix4d T_lc_{Eigen::Matrix4d::Identity()};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ColorLidarNode>());
  rclcpp::shutdown();
  return 0;
}
