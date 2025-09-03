#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include "io_utils.hpp"

using std::placeholders::_1;

class ProjectCloudNode : public rclcpp::Node
{
public:
    ProjectCloudNode() : Node("project_cloud_node")
    {
        const std::string pkg_share =
            ament_index_cpp::get_package_share_directory(
                "livox_camera_lidar_calibration_ros2");
                
        cloud_topic_ = declare_parameter<std::string>("cloud_topic", "/livox/points");
        image_topic_ = declare_parameter<std::string>("image_topic", "/camera/image_raw");
        intr_file_ = declare_parameter<std::string>("intrinsic_file", pkg_share + "/config/intrinsic.txt");
        extr_file_ = declare_parameter<std::string>("extrinsic_file", pkg_share + "/config/extrinsic.txt");
        if (!load_intrinsics_txt(intr_file_, K_))
            RCLCPP_FATAL(get_logger(), "Load intrinsics failed: %s", intr_file_.c_str());
        if (!load_extrinsic_txt(extr_file_, T_lc_))
            RCLCPP_FATAL(get_logger(), "Load extrinsic failed: %s", extr_file_.c_str());

        sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(cloud_topic_, rclcpp::SensorDataQoS(),
                                                                        std::bind(&ProjectCloudNode::onCloud, this, _1));
        sub_img_ = create_subscription<sensor_msgs::msg::Image>(image_topic_, rclcpp::SensorDataQoS(),
                                                                std::bind(&ProjectCloudNode::onImage, this, _1));
        pub_img_ = image_transport::create_publisher(this, "projected_image");

        RCLCPP_INFO(get_logger(), "Loaded K: fx=%.1f fy=%.1f cx=%.1f cy=%.1f size=%dx%d",
                    K_.fx, K_.fy, K_.cx, K_.cy, K_.width, K_.height);
    }

private:
    void onImage(const sensor_msgs::msg::Image::SharedPtr msg) { last_img_ = msg; }

    void onCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (!last_img_)
            return;
        cv::Mat img = cv_bridge::toCvShare(last_img_, "bgr8")->image.clone();

        pcl::PointCloud<pcl::PointXYZI> cloud;
        pcl::fromROSMsg(*msg, cloud);
        Eigen::Matrix3d R = T_lc_.block<3, 3>(0, 0);
        Eigen::Vector3d t = T_lc_.block<3, 1>(0, 3);

        for (const auto &p : cloud.points)
        {
            Eigen::Vector3d Pl(p.x, p.y, p.z);
            Eigen::Vector3d Pc = R * Pl + t;
            if (Pc.z() <= 0.05)
                continue;
            cv::Point2d uv = project_cam(K_, Pc);
            if (uv.x >= 0 && uv.y >= 0 && uv.x < img.cols && uv.y < img.rows)
            {
                cv::circle(img, cv::Point((int)uv.x, (int)uv.y), 1, cv::Scalar(0, 255, 0), -1);
            }
        }
        pub_img_.publish(cv_bridge::CvImage(last_img_->header, "bgr8", img).toImageMsg());
    }

    std::string cloud_topic_, image_topic_, intr_file_, extr_file_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_;
    image_transport::Publisher pub_img_;
    sensor_msgs::msg::Image::SharedPtr last_img_;
    CameraIntrinsics K_;
    Eigen::Matrix4d T_lc_{Eigen::Matrix4d::Identity()};
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ProjectCloudNode>());
    rclcpp::shutdown();
    return 0;
}
