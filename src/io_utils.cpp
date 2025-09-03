#include "io_utils.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

static inline std::string trim(const std::string& s) {
  auto a = s.find_first_not_of(" \t\r\n");
  auto b = s.find_last_not_of(" \t\r\n");
  if (a==std::string::npos) return "";
  return s.substr(a, b-a+1);
}

static bool get_kv(const std::string& line, std::string& k, double& v, std::string& ks, std::string& vs) {
  auto pos = line.find(':');
  if (pos == std::string::npos) return false;
  ks = trim(line.substr(0,pos));
  vs = trim(line.substr(pos+1));
  try { v = std::stod(vs); } catch (...) { return false; }
  k = ks;
  return true;
}

bool load_intrinsics_txt(const std::string& path, CameraIntrinsics& K) {
  std::ifstream ifs(path);
  if (!ifs) return false;
  std::string line, ks, vs, key; double val;
  while (std::getline(ifs, line)) {
    line = trim(line);
    if (line.empty() || line[0]=='#') continue;
    if (!get_kv(line, key, val, ks, vs)) continue;
    if (key=="fx") K.fx=val;
    else if (key=="fy") K.fy=val;
    else if (key=="cx") K.cx=val;
    else if (key=="cy") K.cy=val;
    else if (key=="width") K.width=(int)val;
    else if (key=="height") K.height=(int)val;
    else if (key=="k1") {K.k1=val; K.has_dist=true;}
    else if (key=="k2") {K.k2=val; K.has_dist=true;}
    else if (key=="p1") {K.p1=val; K.has_dist=true;}
    else if (key=="p2") {K.p2=val; K.has_dist=true;}
    else if (key=="k3") {K.k3=val; K.has_dist=true;}
  }
  return (K.fx>0 && K.fy>0 && K.width>0 && K.height>0);
}

bool load_extrinsic_txt(const std::string& path, Eigen::Matrix4d& T_lc) {
  std::ifstream ifs(path);
  if (!ifs) return false;
  std::string line; T_lc.setIdentity();
  int r=0;
  while (std::getline(ifs, line)) {
    line = trim(line);
    if (line.empty() || line[0]=='#') continue;
    std::stringstream ss(line);
    double a,b,c,d;
    if (!(ss>>a>>b>>c>>d)) continue;
    if (r<4) {
      T_lc(r,0)=a; T_lc(r,1)=b; T_lc(r,2)=c; T_lc(r,3)=d;
      r++;
      if (r==4) break;
    }
  }
  return (r==4);
}

bool save_extrinsic_txt(const std::string& path, const Eigen::Matrix4d& T_lc) {
  std::ofstream ofs(path);
  if (!ofs) return false;
  ofs << "# 4x4 T_lidar_to_camera\n";
  ofs.setf(std::ios::fixed); ofs.precision(9);
  for (int i=0;i<4;++i) {
    ofs << T_lc(i,0) << " " << T_lc(i,1) << " " << T_lc(i,2) << " " << T_lc(i,3) << "\n";
  }
  return true;
}

bool load_photo_corners(const std::string& path, std::vector<PhotoCorners>& out) {
  std::ifstream ifs(path);
  if (!ifs) return false;
  std::string line;
  while (std::getline(ifs, line)) {
    line = trim(line);
    if (line.empty()||line[0]=='#') continue;
    std::stringstream ss(line);
    PhotoCorners pc; ss>>pc.name;
    pc.uv.resize(4);
    for (int i=0;i<4;++i) { double u,v; ss>>u>>v; pc.uv[i]=cv::Point2d(u,v); }
    if (pc.name.size()) out.push_back(pc);
  }
  return !out.empty();
}

bool load_lidar_corners(const std::string& path, std::vector<LidarCorners>& out) {
  std::ifstream ifs(path);
  if (!ifs) return false;
  std::string line;
  while (std::getline(ifs, line)) {
    line = trim(line);
    if (line.empty()||line[0]=='#') continue;
    std::stringstream ss(line);
    LidarCorners lc; ss>>lc.name;
    lc.xyz.resize(4);
    for (int i=0;i<4;++i) { double x,y,z; ss>>x>>y>>z; lc.xyz[i]=Eigen::Vector3d(x,y,z); }
    if (lc.name.size()) out.push_back(lc);
  }
  return !out.empty();
}

bool find_chessboard_4_corners(const cv::Mat& gray, cv::Size board_size,
                               double square_size, std::vector<cv::Point2f>& four) {
  std::vector<cv::Point2f> corners;
  bool ok = cv::findChessboardCorners(gray, board_size, corners,
    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
  if (!ok) return false;
  cv::Mat grayf; cv::cvtColor(gray, grayf, gray.channels()==1?cv::COLOR_GRAY2BGR:cv::COLOR_BGR2GRAY);
  cv::cornerSubPix(gray, corners, cv::Size(5,5), cv::Size(-1,-1),
                   cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER, 30, 0.01));
  // 提取四个外角：根据行列数推 4 角索引
  int cols = board_size.width, rows = board_size.height;
  auto idx = [&](int r,int c){ return r*cols + c; };
  four.clear();
  four.push_back(corners[idx(0,0)]);
  four.push_back(corners[idx(0,cols-1)]);
  four.push_back(corners[idx(rows-1,cols-1)]);
  four.push_back(corners[idx(rows-1,0)]);
  return true;
}

cv::Point2d undistort_pt(const CameraIntrinsics& K, const cv::Point2d& pt) {
  if (!K.has_dist) return pt;
  double x = (pt.x - K.cx)/K.fx;
  double y = (pt.y - K.cy)/K.fy;
  double r2 = x*x + y*y;
  double x_d = x*(1 + K.k1*r2 + K.k2*r2*r2 + K.k3*r2*r2*r2) + 2*K.p1*x*y + K.p2*(r2 + 2*x*x);
  double y_d = y*(1 + K.k1*r2 + K.k2*r2*r2 + K.k3*r2*r2*r2) + K.p1*(r2 + 2*y*y) + 2*K.p2*x*y;
  return { x_d*K.fx + K.cx, y_d*K.fy + K.cy };
}
