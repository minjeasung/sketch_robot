import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


robot_ip = LaunchConfiguration("robot_ip")
use_fake_hardware = LaunchConfiguration("use_fake_hardware")
use_isaac_sim = LaunchConfiguration("use_isaac_sim")
fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
model_id = LaunchConfiguration("model_id")
cb_simulation = LaunchConfiguration("cb_simulation")
use_sim_time_cfg = LaunchConfiguration("use_sim_time")


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("robot_ip", default_value="10.0.2.7"),
        DeclareLaunchArgument("use_fake_hardware", default_value="false"),
        DeclareLaunchArgument("use_isaac_sim", default_value="true"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("fake_sensor_commands", default_value="false"),
        DeclareLaunchArgument("cb_simulation", default_value="false"),
        DeclareLaunchArgument("model_id", default_value="rb10_1300e_u"),
    ]
    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )


def _named_srdf():
    srdf_path = os.path.join(
        get_package_share_directory("sketch_control"),
        "config",
        "rbpodo_named.srdf",
    )
    with open(srdf_path, "r", encoding="utf-8") as f:
        return f.read()


def _robot_description_with_eoat():
    robot_description_content = Command([
        FindExecutable(name="xacro"),
        " ",
        PathJoinSubstitution([
            FindPackageShare("sketch_control"),
            "urdf",
            "rbpodo_with_eoat.urdf.xacro",
        ]),
        " robot_ip:=", robot_ip,
        " use_fake_hardware:=", use_fake_hardware,
        " use_isaac_sim:=", use_isaac_sim,
        " fake_sensor_commands:=", fake_sensor_commands,
        " cb_simulation:=", cb_simulation,
        " model_id:=", model_id,
    ])
    return {
        "robot_description": ParameterValue(
            robot_description_content,
            value_type=str,
        )
    }


def launch_setup(context, *args, **kwargs):
    use_sim_time = {"use_sim_time": use_sim_time_cfg}
    mappings = {
        "robot_ip": robot_ip,
        "use_fake_hardware": use_fake_hardware,
        "use_isaac_sim": use_isaac_sim,
        "fake_sensor_commands": fake_sensor_commands,
        "model_id": model_id,
        "cb_simulation": cb_simulation,
    }

    moveit_config = (
        MoveItConfigsBuilder("rbpodo", package_name="rbpodo_moveit_config")
        .robot_description(file_path="config/rbpodo.urdf.xacro", mappings=mappings)
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .planning_pipelines(
            pipelines=["ompl", "chomp", "pilz_industrial_motion_planner"]
        )
        .to_moveit_configs()
    )
    moveit_config.robot_description = _robot_description_with_eoat()
    moveit_config.robot_description_semantic = {
        "robot_description_semantic": _named_srdf()
    }

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), use_sim_time],
    )

    # world is the real RB10 pendant/base frame. rbpodo URDF link0 is rotated
    # 90deg clockwise relative to that frame, so publish world->link0 as +90deg.
    # Robot link TF still comes from robot_state_publisher.
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="fallback_world_to_link0",
        output="log",
        arguments=[
            "--frame-id", "world",
            "--child-frame-id", "link0",
            "--x", "0", "--y", "0", "--z", "0",
            "--qx", "0", "--qy", "0",
            "--qz", "0.7071067811865475",
            "--qw", "0.7071067811865476",
        ],
    )

    static_tf_world_bridge = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_World_bridge",
        output="log",
        arguments=[
            "--frame-id",
            "world",
            "--child-frame-id",
            "World",
            "--x",
            "0",
            "--y",
            "0",
            "--z",
            "0",
            "--qx",
            "0",
            "--qy",
            "0",
            "--qz",
            "0",
            "--qw",
            "1",
        ],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description, use_sim_time],
    )

    controllers_path = os.path.join(
        get_package_share_directory("rbpodo_bringup"),
        "config",
        "controllers.yaml",
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[moveit_config.robot_description, controllers_path, use_sim_time],
        output="both",
        remappings=[
            ("joint_states", "controller_manager/joint_states"),
        ],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager-timeout",
            "300",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_trajectory_controller",
            "-c",
            "/controller_manager",
            "--controller-manager-timeout",
            "300",
            "--switch-timeout",
            "60",
        ],
    )

    activate_arm_after_joint_state_broadcaster = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner],
        )
    )

    rviz_config = os.path.join(
        get_package_share_directory("rbpodo_moveit_config"),
        "config",
        "moveit.rviz",
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rb10_full_moveit_rviz",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            use_sim_time,
        ],
    )

    return [
        static_tf,
        static_tf_world_bridge,
        robot_state_publisher,
        move_group,
        ros2_control_node,
        joint_state_broadcaster_spawner,
        activate_arm_after_joint_state_broadcaster,
        rviz,
    ]
