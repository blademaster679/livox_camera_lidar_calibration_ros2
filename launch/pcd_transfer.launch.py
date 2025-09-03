from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='livox_camera_lidar_calibration_ros2',
            executable='pcd_transfer_node',
            name='pcd_transfer',
            parameters=[{
                'cloud_topic': '/livox/points',   # 或你的 /livox/lidar
                'save_dir': 'data/pcdFiles',
                'every_n': 1
            }]
        )
    ])
