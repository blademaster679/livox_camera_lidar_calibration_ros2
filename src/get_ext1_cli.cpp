#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <string>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "io_utils.hpp"

// 像素重投影残差
struct ReprojError {
  ReprojError(const cv::Point2d& uv, const Eigen::Vector3d& Pl, const CameraIntrinsics& K)
    : uv_(uv), Pl_(Pl), K_(K) {}
  template <typename T>
  bool operator()(const T* const q_xyzw, const T* const t_xyz, T* residuals) const {
    Eigen::Quaternion<T> q(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]); // w,x,y,z
    Eigen::Matrix<T,3,1> t(t_xyz[0], t_xyz[1], t_xyz[2]);
    Eigen::Matrix<T,3,1> Pl(T(Pl_.x()), T(Pl_.y()), T(Pl_.z()));
    Eigen::Matrix<T,3,1> Pc = q * Pl + t;
    T u = T(K_.fx) * (Pc.x()/Pc.z()) + T(K_.cx);
    T v = T(K_.fy) * (Pc.y()/Pc.z()) + T(K_.cy);
    residuals[0] = u - T(uv_.x);
    residuals[1] = v - T(uv_.y);
    return true;
  }
  static ceres::CostFunction* Create(const cv::Point2d& uv, const Eigen::Vector3d& Pl, const CameraIntrinsics& K) {
    return new ceres::AutoDiffCostFunction<ReprojError, 2, 4, 3>(new ReprojError(uv, Pl, K));
  }
  cv::Point2d uv_; Eigen::Vector3d Pl_; CameraIntrinsics K_;
};

int main(int argc, char** argv) {
  std::string pkg_share;
  try {
    pkg_share = ament_index_cpp::get_package_share_directory(
        "livox_camera_lidar_calibration_ros2");
  } catch (const std::exception& e) {
    std::cerr << "WARNING: get_package_share_directory failed: " << e.what()
              << "\nFallback to current dir.\n";
    pkg_share = ".";
  }

  std::string intr = (argc>1)? argv[1]: (pkg_share + "/config/intrinsic.txt");
  std::string ph   = (argc>2)? argv[2]: (pkg_share + "/data/corner_photo.txt");
  std::string ld   = (argc>3)? argv[3]: (pkg_share + "/data/corner_lidar.txt");
  std::string outT = (argc>4)? argv[4]: (pkg_share + "/config/extrinsic.txt");

  CameraIntrinsics K;
  if (!load_intrinsics_txt(intr, K)) { std::cerr << "Bad intrinsics\n"; return 1; }
  std::vector<PhotoCorners> photos;
  if (!load_photo_corners(ph, photos)) { std::cerr << "Bad photo corners\n"; return 1; }
  std::vector<LidarCorners> lidars;
  if (!load_lidar_corners(ld, lidars)) { std::cerr << "Bad lidar corners\n"; return 1; }

  std::map<std::string, std::pair<PhotoCorners, LidarCorners>> pairs;
  for (auto& p: photos) for (auto& l: lidars) if (p.name == l.name) pairs[p.name] = {p, l};
  if (pairs.empty()) { std::cerr << "No matched names between photo and lidar\n"; return 1; }

  double q_xyzw[4] = {0,0,0,1};
  double t_xyz[3]  = {0,0,0};

  ceres::Problem problem;
  for (auto& kv : pairs) {
    auto& pc = kv.second.first;
    auto& lc = kv.second.second;
    for (int i=0; i<4; ++i) {
      ceres::CostFunction* cost = ReprojError::Create(pc.uv[i], lc.xyz[i], K);
      problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), q_xyzw, t_xyz);
    }
  }
  problem.SetParameterization(q_xyzw, new ceres::EigenQuaternionParameterization());

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 200;
  options.num_threads = 4;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";

  Eigen::Quaterniond q(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
  Eigen::Matrix3d R = q.normalized().toRotationMatrix();
  Eigen::Vector3d t(t_xyz[0], t_xyz[1], t_xyz[2]);

  Eigen::Matrix4d T; T.setIdentity();
  T.block<3,3>(0,0) = R; T.block<3,1>(0,3) = t;

  if (!save_extrinsic_txt(outT, T)) { std::cerr << "Save extrinsic failed\n"; return 1; }
  std::cout << "Saved extrinsic to " << outT << "\n";
  return 0;
}
