from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='livox_camera_lidar_calibration_ros2',
            executable='color_lidar_node',
            name='color_lidar',
            parameters=[{
                'cloud_topic': '/livox/points',
                'image_topic': '/camera/image_raw',
                'intrinsic_file': 'config/intrinsic.txt',
                'extrinsic_file': 'config/extrinsic.txt',
                'output_colored_pcd': 'data/pcdFiles/colored.pcd'
            }],
        )
    ])
