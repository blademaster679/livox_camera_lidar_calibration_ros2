#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>

#include <filesystem>
#include <iomanip>
#include <sstream>

using std::placeholders::_1;

class PcdTransferNode : public rclcpp::Node
{
public:
    PcdTransferNode() : Node("pcd_transfer_node")
    {
        const std::string pkg_share =
            ament_index_cpp::get_package_share_directory(
                "livox_camera_lidar_calibration_ros2");

        cloud_topic_ = declare_parameter<std::string>(
            "cloud_topic", "/livox/points");
        // 默认把 PCD 存到包的 data/pcdFiles；如无写权限，请在启动时覆盖 save_dir 参数
        save_dir_ = declare_parameter<std::string>(
            "save_dir", pkg_share + "/data/pcdFiles");
        every_n_ = declare_parameter<int>("every_n", 1);

        std::error_code ec;
        std::filesystem::create_directories(save_dir_, ec);
        if (ec)
        {
            RCLCPP_WARN(get_logger(), "create_directories failed for %s: %s",
                        save_dir_.c_str(), ec.message().c_str());
        }

        sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic_, rclcpp::SensorDataQoS(),
            std::bind(&PcdTransferNode::cb, this, _1));

        RCLCPP_INFO(get_logger(), "Saving PCD to %s, topic=%s, every_n=%d",
                    save_dir_.c_str(), cloud_topic_.c_str(), every_n_);
    }

private:
    void cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (++count_ % every_n_ != 0)
            return;

        // 建文件名
        std::ostringstream oss;
        oss << save_dir_ << "/" << std::setw(6) << std::setfill('0') << idx_++ << ".pcd";
        const std::string path = oss.str();

        // 转成强类型点云（这里按 Livox 常见的 XYZ + 强度）
        pcl::PointCloud<pcl::PointXYZI> cloud_xyzi;
        pcl::fromROSMsg(*msg, cloud_xyzi);

        if (pcl::io::savePCDFileBinary(path, cloud_xyzi) == 0)
        {
            RCLCPP_INFO(get_logger(), "Saved %s", path.c_str());
        }
        else
        {
            RCLCPP_WARN(get_logger(), "Failed to save %s", path.c_str());
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    std::string cloud_topic_;
    std::string save_dir_;
    int every_n_{1};
    int count_{0};
    int idx_{0};
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PcdTransferNode>());
    rclcpp::shutdown();
    return 0;
}
