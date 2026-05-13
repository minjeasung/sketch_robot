"""
sim_bringup.launch.py — Isaac Sim + ros2_control bringup for RB10.

실로봇 bringup (rbpodo_bringup/launch/rbpodo.launch.py) 과 같은 컨트롤러 스택을 시뮬에 적용:
  - controller_manager (ros2_control_node, 100Hz)
  - joint_state_broadcaster (output 을 controller_manager/joint_states 로 remap → /joint_states 충돌 회피)
  - joint_trajectory_controller (position command)
  - robot_state_publisher (URDF: rb10_1300e_u, use_isaac_sim:=true)

Isaac Sim 측은 별도 실행:
  source ~/isaac_env/bin/activate
  isaacsim --exec ~/sketch_robot_ws/src/sketch_control/sketch_control/isaac_sim_rb10.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    TextSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


SIM_CONTROLLERS_YAML = os.path.expanduser(
    "~/sketch_robot_ws/config/sim_controllers.yaml"
)


def generate_launch_description():
    model_id = LaunchConfiguration("model_id")
    model_path = LaunchConfiguration("model_path")

    # xacro → URDF, use_isaac_sim:=true 강제.
    # ParameterValue(value_type=str) 로 감싸야 launch 가 yaml 파싱 시도 안 함.
    robot_description = ParameterValue(
        Command(
            [
                FindExecutable(name="xacro"),
                " ",
                model_path,
                " use_isaac_sim:=true",
                " use_fake_hardware:=false",
            ]
        ),
        value_type=str,
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "model_id",
            default_value="rb10_1300e_u",
            description="Rainbow Robotics RB model id",
        ),
        DeclareLaunchArgument(
            "model_path",
            default_value=[
                TextSubstitution(text=os.path.join(
                    get_package_share_directory("rbpodo_description"),
                    "robots", "",
                )),
                model_id,
                TextSubstitution(text=".urdf.xacro"),
            ],
            description="Path to robot xacro",
        ),

        # robot_state_publisher (URDF + TF for fixed links)
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="both",
            parameters=[{"robot_description": robot_description}],
        ),

        # ros2_control_node — joint_states publish 를 controller_manager/joint_states 로 remap.
        # JointStateTopicSystem 은 Isaac Sim 의 /joint_states 를 subscribe (param 으로 처리됨).
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[
                SIM_CONTROLLERS_YAML,
                {"robot_description": robot_description},
            ],
            remappings=[
                ("joint_states", "controller_manager/joint_states"),
                ("~/robot_description", "/robot_description"),
            ],
            output="both",
            on_exit=Shutdown(),
        ),

        # joint_state_broadcaster — controller_manager/joint_states 로 발행 (위 remap 결과).
        # jtc 는 의도적으로 spawn 안 함 (E 진단): command_interface 를 claim 하는 controller 가
        # active 되기 전에 controller_manager update loop 가 시작하면 JointStateTopicSystem::write
        # 가 invalid handle 의 get_command() 호출 → segfault. jsb 는 state 만 — command 안 건드림.
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster"],
            output="screen",
        ),
        # jtc 활성화는 수동 (검증용):
        #   ros2 control load_controller --set-state active joint_trajectory_controller
    ])
