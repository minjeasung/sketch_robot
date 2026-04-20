"""
전체 시스템 런치 파일
터미널 1: Isaac Sim (별도 실행)
터미널 2: 이 런치 파일 (robot_state_publisher + MoveIt2 + 스케치 UI)
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

    # ---- 스케치 UI 노드 ---------------------------------------------------
    sketch_ui_node = Node(
        package='sketch_control',
        executable='sketch_ui',
        name='sketch_ui',
        output='screen',
    )

    # ---- MoveIt Executor 노드 ---------------------------------------------
    moveit_executor_node = Node(
        package='sketch_control',
        executable='moveit_executor',
        name='moveit_executor',
        output='screen',
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

    # ---- 용접 비드 시각화 (RViz MarkerArray) ------------------------------
    weld_visualizer_node = Node(
        package='sketch_control',
        executable='weld_visualizer',
        name='weld_visualizer',
        output='screen',
    )

    # ---- ROS-TCP-Endpoint (Unity 연결용 TCP 서버) -------------------------
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
        robot_state_publisher_node,
        static_tf_world_bridge,
        moveit_launch,
        sketch_ui_node,
        moveit_executor_node,
        weld_visualizer_node,
        ros_tcp_endpoint_node,
    ])
