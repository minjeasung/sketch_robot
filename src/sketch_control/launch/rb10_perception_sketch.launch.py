from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_depth_pointcloud = LaunchConfiguration("use_sim_depth_pointcloud")
    use_d405_refinement = LaunchConfiguration("use_d405_refinement")
    use_d405_mount_tf = LaunchConfiguration("use_d405_mount_tf")
    use_ft_normal_controller = LaunchConfiguration("use_ft_normal_controller")
    d405_cloud_topic = LaunchConfiguration("d405_cloud_topic")
    ft_wrench_topic = LaunchConfiguration("ft_wrench_topic")
    ft_force_sign = LaunchConfiguration("ft_force_sign")
    ft_target_force_n = LaunchConfiguration("ft_target_force_n")
    ft_abort_force_n = LaunchConfiguration("ft_abort_force_n")

    zed_x = LaunchConfiguration("zed_x")
    zed_y = LaunchConfiguration("zed_y")
    zed_z = LaunchConfiguration("zed_z")
    zed_qx = LaunchConfiguration("zed_qx")
    zed_qy = LaunchConfiguration("zed_qy")
    zed_qz = LaunchConfiguration("zed_qz")
    zed_qw = LaunchConfiguration("zed_qw")

    static_zed_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="zed_camera_static_tf",
        output="screen",
        arguments=[
            "--frame-id", "World",
            "--child-frame-id", "zed_left_camera_frame_optical",
            "--x", zed_x,
            "--y", zed_y,
            "--z", zed_z,
            "--qx", zed_qx,
            "--qy", zed_qy,
            "--qz", zed_qz,
            "--qw", zed_qw,
        ],
    )

    # D405 is rigidly mounted on the TCP/EOAT. The RealSense driver normally
    # publishes d405_link -> optical frames; this bridges the moving robot TCP
    # to the camera base link so the D405 cloud can be transformed to ZED/world.
    static_d405_mount_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="d405_mount_static_tf",
        output="screen",
        condition=IfCondition(use_d405_mount_tf),
        arguments=[
            "--frame-id", "tcp",
            "--child-frame-id", "d405_link",
            "--x", "0.00905",
            "--y", "-0.07640",
            "--z", "0.04375",
            "--qx", "0.0",
            "--qy", "0.0",
            "--qz", "-0.7071067811865475",
            "--qw", "0.7071067811865476",
        ],
    )

    depth_to_pointcloud = Node(
        package="sketch_control",
        executable="depth_to_pointcloud",
        name="depth_to_pointcloud",
        output="screen",
        condition=IfCondition(use_sim_depth_pointcloud),
    )

    wall_detector = Node(
        package="sketch_control",
        executable="wall_detector",
        name="wall_detector",
        output="screen",
    )

    target_selector = Node(
        package="sketch_control",
        executable="target_selector",
        name="target_selector",
        output="screen",
    )

    wall_projector = Node(
        package="sketch_control",
        executable="wall_projector",
        name="wall_projector",
        output="screen",
    )

    sketch_to_waypoints = Node(
        package="sketch_control",
        executable="sketch_to_waypoints",
        name="sketch_to_waypoints",
        output="screen",
    )

    environment_scanner = Node(
        package="sketch_control",
        executable="environment_scanner",
        name="environment_scanner",
        output="screen",
    )

    d405_surface_refiner = Node(
        package="sketch_control",
        executable="d405_surface_refiner",
        name="d405_surface_refiner",
        output="screen",
        condition=IfCondition(use_d405_refinement),
        parameters=[{
            "cloud_topic": d405_cloud_topic,
        }],
    )

    ft_normal_controller = Node(
        package="sketch_control",
        executable="ft_normal_controller",
        name="ft_normal_controller",
        output="screen",
        condition=IfCondition(use_ft_normal_controller),
        parameters=[{
            "wrench_topic": ft_wrench_topic,
            "base_frame": "link0",
            "sensor_frame": "tcp",
            "force_sign": ft_force_sign,
            "target_force_n": ft_target_force_n,
            "abort_force_n": ft_abort_force_n,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_depth_pointcloud",
            default_value="true",
            description="Isaac Sim depth image -> ZED-compatible PointCloud2",
        ),
        DeclareLaunchArgument(
            "use_d405_refinement",
            default_value="true",
            description="Use wrist D405 point cloud to refine work-area distance",
        ),
        DeclareLaunchArgument(
            "use_d405_mount_tf",
            default_value="true",
            description="Publish tcp->d405_link static mount transform",
        ),
        DeclareLaunchArgument(
            "use_ft_normal_controller",
            default_value="true",
            description="Compute wall-normal contact force from AFT200 wrench",
        ),
        DeclareLaunchArgument(
            "d405_cloud_topic",
            default_value="/d405/d405/depth/color/points",
            description="D405 PointCloud2 topic used for local surface refinement",
        ),
        DeclareLaunchArgument(
            "ft_wrench_topic",
            default_value="/aft200/ft",
            description="AFT200 WrenchStamped topic",
        ),
        DeclareLaunchArgument(
            "ft_force_sign",
            default_value="1.0",
            description="Use -1.0 if pushing the wall reports negative normal force",
        ),
        DeclareLaunchArgument(
            "ft_target_force_n",
            default_value="10.0",
            description="Desired roller normal force during painting",
        ),
        DeclareLaunchArgument(
            "ft_abort_force_n",
            default_value="30.0",
            description="Emergency stop threshold for contact stages",
        ),
        DeclareLaunchArgument("zed_x", default_value="-0.4715750877078567"),
        DeclareLaunchArgument("zed_y", default_value="0.2562866180459926"),
        DeclareLaunchArgument("zed_z", default_value="1.0085930379522832"),
        DeclareLaunchArgument("zed_qx", default_value="-0.5362126233398715"),
        DeclareLaunchArgument("zed_qy", default_value="0.6265018657327951"),
        DeclareLaunchArgument("zed_qz", default_value="-0.4297485208739163"),
        DeclareLaunchArgument("zed_qw", default_value="0.36781468650800403"),
        static_zed_tf,
        static_d405_mount_tf,
        depth_to_pointcloud,
        wall_detector,
        target_selector,
        wall_projector,
        sketch_to_waypoints,
        environment_scanner,
        d405_surface_refiner,
        ft_normal_controller,
    ])
