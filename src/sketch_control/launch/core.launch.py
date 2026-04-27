"""
공통 코어 런치 (Phase 1 / Phase 2 가 공통으로 include).

띄우는 노드:
  - robot_state_publisher (URDF)
  - static_transform_publisher (world -> World)
  - MoveIt2 (move_group + RViz, ur_moveit.launch.py include)
  - moveit_executor
  - weld_visualizer
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # ---- robot_description (xacro -> URDF) --------------------------------
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]), " ",
        PathJoinSubstitution([
            FindPackageShare("ur_description"), "urdf", "ur.urdf.xacro"
        ]),
        " ", "ur_type:=ur10",
        " ", "name:=ur",
    ])
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    # ---- MoveIt2 for UR10 -------------------------------------------------
    ur_moveit_config = get_package_share_directory('ur_moveit_config')
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ur_moveit_config, 'launch', 'ur_moveit.launch.py')
        ),
        launch_arguments={
            'ur_type': 'ur10',
            'use_sim_time': 'true',
        }.items(),
    )

    # ---- Static TF: world -> World (Isaac Sim 은 "World", URDF 는 "world") --
    static_tf_world_bridge = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_World_bridge",
        arguments=["--frame-id", "world", "--child-frame-id", "World",
                   "--x", "0", "--y", "0", "--z", "0",
                   "--roll", "0", "--pitch", "0", "--yaw", "0"],
        output="log",
    )

    # ---- MoveIt Executor 노드 ---------------------------------------------
    moveit_executor_node = Node(
        package='sketch_control',
        executable='moveit_executor',
        name='moveit_executor',
        output='screen',
    )

    # ---- 용접 비드 시각화 (RViz MarkerArray) ------------------------------
    weld_visualizer_node = Node(
        package='sketch_control',
        executable='weld_visualizer',
        name='weld_visualizer',
        output='screen',
    )

    return LaunchDescription([
        robot_state_publisher_node,
        static_tf_world_bridge,
        moveit_launch,
        moveit_executor_node,
        weld_visualizer_node,
    ])
