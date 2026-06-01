"""
공통 코어 런치 (Phase 1 / Phase 2 가 공통으로 include).

띄우는 노드:
  - robot_state_publisher (URDF)
  - static_transform_publisher (world -> World)
  - MoveIt2 (move_group + RViz, ur_moveit.launch.py include)
  - ZED perception (wall plane, yellow work area, obstacle scanner)
  - Isaac Sim depth -> point cloud bridge
  - moveit_executor
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    use_sim_depth_pointcloud = LaunchConfiguration('use_sim_depth_pointcloud')

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

    # ---- ZED perception pipeline ------------------------------------------
    depth_to_pointcloud_node = Node(
        package='sketch_control',
        executable='depth_to_pointcloud',
        name='depth_to_pointcloud',
        output='screen',
        condition=IfCondition(use_sim_depth_pointcloud),
    )

    wall_detector_node = Node(
        package='sketch_control',
        executable='wall_detector',
        name='wall_detector',
        output='screen',
    )

    target_selector_node = Node(
        package='sketch_control',
        executable='target_selector',
        name='target_selector',
        output='screen',
    )

    wall_projector_node = Node(
        package='sketch_control',
        executable='wall_projector',
        name='wall_projector',
        output='screen',
    )

    sketch_to_waypoints_node = Node(
        package='sketch_control',
        executable='sketch_to_waypoints',
        name='sketch_to_waypoints',
        output='screen',
    )

    environment_scanner_node = Node(
        package='sketch_control',
        executable='environment_scanner',
        name='environment_scanner',
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_depth_pointcloud',
            default_value='true',
            description='Isaac Sim depth image 를 PointCloud2 로 변환할지 여부'),
        robot_state_publisher_node,
        static_tf_world_bridge,
        moveit_launch,
        depth_to_pointcloud_node,
        wall_detector_node,
        target_selector_node,
        wall_projector_node,
        sketch_to_waypoints_node,
        environment_scanner_node,
        moveit_executor_node,
    ])
