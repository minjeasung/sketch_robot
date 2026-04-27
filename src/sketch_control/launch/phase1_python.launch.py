"""
Phase 1 런치: core + Python tkinter 스케치 UI

사용:
  ros2 launch sketch_control phase1_python.launch.py
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

    sketch_ui_node = Node(
        package='sketch_control',
        executable='sketch_ui',
        name='sketch_ui',
        output='screen',
    )

    return LaunchDescription([
        core_launch,
        sketch_ui_node,
    ])
