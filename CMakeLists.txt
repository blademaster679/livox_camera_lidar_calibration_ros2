cmake_minimum_required(VERSION 3.10)
project(livox_camera_lidar_calibration_ros2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(PCL REQUIRED COMPONENTS io common)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(ament_index_cpp REQUIRED)

ament_target_dependencies(project_cloud_node
  rclcpp sensor_msgs image_transport cv_bridge tf2 tf2_ros tf2_eigen ament_index_cpp)

ament_target_dependencies(color_lidar_node
  rclcpp sensor_msgs image_transport cv_bridge tf2 tf2_ros tf2_eigen ament_index_cpp)

ament_target_dependencies(pcd_transfer_node
  rclcpp sensor_msgs ament_index_cpp)

ament_target_dependencies(corner_photo_cli ament_index_cpp)
ament_target_dependencies(get_ext1_cli     ament_index_cpp)
ament_target_dependencies(get_ext2_cli     ament_index_cpp)

add_definitions(-DPROJECT_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\")

include_directories(
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# Utils
add_library(calib_io_utils src/io_utils.cpp)
target_link_libraries(calib_io_utils ${OpenCV_LIBS} ${PCL_LIBRARIES})

# --- ROS2 Nodes ---
add_executable(pcd_transfer_node src/pcd_transfer_node.cpp)
ament_target_dependencies(pcd_transfer_node rclcpp sensor_msgs)
target_link_libraries(pcd_transfer_node ${PCL_LIBRARIES})

add_executable(project_cloud_node src/project_cloud_node.cpp)
ament_target_dependencies(project_cloud_node
  rclcpp sensor_msgs image_transport cv_bridge tf2 tf2_ros tf2_eigen)
target_link_libraries(project_cloud_node ${OpenCV_LIBS} ${PCL_LIBRARIES} calib_io_utils)

add_executable(color_lidar_node src/color_lidar_node.cpp)
ament_target_dependencies(color_lidar_node
  rclcpp sensor_msgs image_transport cv_bridge tf2 tf2_ros tf2_eigen)
target_link_libraries(color_lidar_node ${OpenCV_LIBS} ${PCL_LIBRARIES} calib_io_utils)

# --- Offline CLIs ---
add_executable(corner_photo_cli src/corner_photo_cli.cpp)
target_link_libraries(corner_photo_cli ${OpenCV_LIBS} calib_io_utils)

add_executable(get_ext1_cli src/get_ext1_cli.cpp)
target_link_libraries(get_ext1_cli ${CERES_LIBRARIES} ${OpenCV_LIBS} calib_io_utils)

add_executable(get_ext2_cli src/get_ext2_cli.cpp)
target_link_libraries(get_ext2_cli ${CERES_LIBRARIES} ${OpenCV_LIBS} calib_io_utils)

install(TARGETS
  pcd_transfer_node project_cloud_node color_lidar_node
  corner_photo_cli get_ext1_cli get_ext2_cli calib_io_utils
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config data
  DESTINATION share/${PROJECT_NAME})

ament_package()
