"""
Phase 2 런치: core + ROS-TCP-Endpoint (Unity 연결).

sketch_ui 는 띄우지 않음 (Unity 가 GUI 역할).

사용:
  ros2 launch sketch_control phase2_unity.launch.py
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('sketch_control')

    core_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'core.launch.py')
        )
    )

    ros_tcp_endpoint_node = Node(
        package='ros_tcp_endpoint',
        executable='default_server_endpoint',
        name='ros_tcp_endpoint',
        output='screen',
        parameters=[
            {'ROS_IP': '0.0.0.0'},
            {'ROS_TCP_PORT': 10000},
        ],
    )

    return LaunchDescription([
        core_launch,
        ros_tcp_endpoint_node,
    ])
