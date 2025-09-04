#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <string>
#include <fstream>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "io_utils.hpp"

struct ReprojErrorIO {
  ReprojErrorIO(const cv::Point2d& uv, const Eigen::Vector3d& Pl)
    : uv_(uv), Pl_(Pl) {}
  template <typename T>
  bool operator()(const T* const q_xyzw, const T* const t_xyz,
                  const T* const intr6, T* residuals) const {
    // intr6: fx, fy, cx, cy, (k1,k2占位不用)
    Eigen::Quaternion<T> q(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
    Eigen::Matrix<T,3,1> t(t_xyz[0], t_xyz[1], t_xyz[2]);
    Eigen::Matrix<T,3,1> Pl(T(Pl_.x()), T(Pl_.y()), T(Pl_.z()));
    Eigen::Matrix<T,3,1> Pc = q*Pl + t;

    const T fx = intr6[0], fy = intr6[1], cx = intr6[2], cy = intr6[3];
    T u = fx * (Pc.x()/Pc.z()) + cx;
    T v = fy * (Pc.y()/Pc.z()) + cy;

    residuals[0] = u - T(uv_.x);
    residuals[1] = v - T(uv_.y);
    return true;
  }
  static ceres::CostFunction* Create(const cv::Point2d& uv, const Eigen::Vector3d& Pl) {
    return new ceres::AutoDiffCostFunction<ReprojErrorIO, 2, 4, 3, 6>(
      new ReprojErrorIO(uv, Pl));
  }
  cv::Point2d uv_; Eigen::Vector3d Pl_;
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
  std::string outK = (argc>5)? argv[5]: (pkg_share + "/config/intrinsic.txt");

  CameraIntrinsics K;
  if (!load_intrinsics_txt(intr, K)) { std::cerr << "Bad intrinsics\n"; return 1; }
  std::vector<PhotoCorners> photos;
  if (!load_photo_corners(ph, photos)) { std::cerr << "Bad photo corners\n"; return 1; }
  std::vector<LidarCorners> lidars;
  if (!load_lidar_corners(ld, lidars)) { std::cerr << "Bad lidar corners\n"; return 1; }

  std::map<std::string, std::pair<PhotoCorners, LidarCorners>> pairs;
  for (auto& p: photos) for (auto& l: lidars) if (p.name == l.name) pairs[p.name] = {p, l};
  if (pairs.empty()) { std::cerr << "No matched names\n"; return 1; }

  double q_xyzw[4] = {0,0,0,1};
  double t_xyz[3]  = {0,0,0};
  double intr6[6]  = {K.fx, K.fy, K.cx, K.cy, 0, 0}; // 只优化 fx,fy,cx,cy

  ceres::Problem problem;
  for (auto& kv : pairs) {
    auto& pc = kv.second.first;
    auto& lc = kv.second.second;
    for (int i=0; i<4; ++i) {
      auto* cost = ReprojErrorIO::Create(pc.uv[i], lc.xyz[i]);
      problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), q_xyzw, t_xyz, intr6);
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

  // 覆写/保存内参（只更新 fx,fy,cx,cy；其余保持原值）
  std::ofstream ofs(outK);
  ofs << "fx: " << intr6[0] << "\n";
  ofs << "fy: " << intr6[1] << "\n";
  ofs << "cx: " << intr6[2] << "\n";
  ofs << "cy: " << intr6[3] << "\n";
  ofs << "width: "  << K.width  << "\n";
  ofs << "height: " << K.height << "\n";
  ofs << "k1: " << K.k1 << "\n";
  ofs << "k2: " << K.k2 << "\n";
  ofs << "p1: " << K.p1 << "\n";
  ofs << "p2: " << K.p2 << "\n";
  ofs << "k3: " << K.k3 << "\n";

  std::cout << "Saved extrinsic to " << outT << " and intrinsics to " << outK << "\n";
  return 0;
}
