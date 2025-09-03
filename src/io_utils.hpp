#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

struct CameraIntrinsics {
  double fx{0}, fy{0}, cx{0}, cy{0};
  int width{0}, height{0};
  // optional distortion
  double k1{0}, k2{0}, p1{0}, p2{0}, k3{0};
  bool has_dist{false};
};

bool load_intrinsics_txt(const std::string& path, CameraIntrinsics& K);
bool load_extrinsic_txt(const std::string& path, Eigen::Matrix4d& T_lc);
bool save_extrinsic_txt(const std::string& path, const Eigen::Matrix4d& T_lc);

// corner files
// corner_photo.txt: each line -> img_name u1 v1 u2 v2 u3 v3 u4 v4
// corner_lidar.txt: each line -> pcd_name x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4
struct PhotoCorners { std::string name; std::vector<cv::Point2d> uv; };
struct LidarCorners { std::string name; std::vector<Eigen::Vector3d> xyz; };

bool load_photo_corners(const std::string& path, std::vector<PhotoCorners>& out);
bool load_lidar_corners(const std::string& path, std::vector<LidarCorners>& out);

// chessboard helpers
bool find_chessboard_4_corners(const cv::Mat& gray, cv::Size board_size,
                               double square_size, std::vector<cv::Point2f>& four);

// project point 3D in camera frame to pixel
inline cv::Point2d project_cam(const CameraIntrinsics& K, const Eigen::Vector3d& Pc) {
  double u = K.fx * (Pc.x() / Pc.z()) + K.cx;
  double v = K.fy * (Pc.y() / Pc.z()) + K.cy;
  return {u, v};
}

// undistort single point (approx; for visualization)
cv::Point2d undistort_pt(const CameraIntrinsics& K, const cv::Point2d& pt);
