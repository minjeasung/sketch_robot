"""
MoveIt + robot_state_publisher 통합 launch
터미널 3에서 이걸 실행:
  ros2 launch sketch_control moveit_with_rsp.launch.py
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ur_type = "ur10"

    robot_description_content = Command([
        FindExecutable(name="xacro"), " ",
        PathJoinSubstitution(
            [FindPackageShare("ur_description"), "urdf", "ur.urdf.xacro"]),
        " ",
        "ur_type:=", ur_type, " ",
        "name:=ur", " ",
        "safety_limits:=true", " ",
        "safety_pos_margin:=0.15", " ",
        "safety_k_position:=20",
    ])
    robot_description = {
        "robot_description": ParameterValue(
            robot_description_content, value_type=str),
    }

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    ur_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("ur_moveit_config"),
            "launch",
            "ur_moveit.launch.py",
        ])),
        launch_arguments={
            "ur_type": ur_type,
            "use_sim_time": "true",
            "launch_rviz": "false",
        }.items(),
    )

    return LaunchDescription([rsp, ur_moveit])
