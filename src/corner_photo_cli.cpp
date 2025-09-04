#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "io_utils.hpp"

// usage:
// ros2 run livox_camera_lidar_calibration_ros2 corner_photo_cli \
//   [photos_dir] [output_txt] [cols rows square_m]
int main(int argc, char** argv) {
  // 用 ament 索引获取包路径，作为默认路径前缀
  std::string pkg_share;
  try {
    pkg_share = ament_index_cpp::get_package_share_directory(
        "livox_camera_lidar_calibration_ros2");
  } catch (const std::exception& e) {
    std::cerr << "WARNING: get_package_share_directory failed: " << e.what()
              << "\nFallback to current dir.\n";
    pkg_share = ".";
  }

  std::string photos_dir = (argc > 1) ? argv[1] : (pkg_share + "/data/camera/photos");
  std::string out_txt    = (argc > 2) ? argv[2] : (pkg_share + "/data/corner_photo.txt");
  int cols   = (argc > 3) ? std::stoi(argv[3]) : 9;
  int rows   = (argc > 4) ? std::stoi(argv[4]) : 6;
  double sq  = (argc > 5) ? std::stod(argv[5]) : 0.025;

  std::vector<std::string> files;
  for (auto& p : std::filesystem::directory_iterator(photos_dir)) {
    if (!p.is_regular_file()) continue;
    auto ext = p.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext==".jpg" || ext==".jpeg" || ext==".png" || ext==".bmp") {
      files.push_back(p.path().string());
    }
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    std::cerr << "No images found in " << photos_dir << "\n";
    return 1;
  }

  std::ofstream ofs(out_txt);
  if (!ofs) {
    std::cerr << "Cannot write " << out_txt << "\n";
    return 1;
  }

  for (auto& f : files) {
    cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);
    if (img.empty()) { std::cerr << "Skip " << f << "\n"; continue; }
    cv::Mat gray; cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> four;
    bool ok = find_chessboard_4_corners(gray, {cols, rows}, sq, four);
    if (!ok) { std::cerr << "Chessboard not found in " << f << "\n"; continue; }

    // 可视化确认
    for (auto& p : four) cv::circle(img, p, 6, {0,0,255}, 2);
    cv::imshow("corners", img);
    cv::waitKey(10);

    ofs << std::filesystem::path(f).filename().string();
    for (auto& p : four) ofs << " " << p.x << " " << p.y;
    ofs << "\n";
    std::cout << "Saved corners for " << f << "\n";
  }
  std::cout << "Wrote " << out_txt << "\n";
  return 0;
}
