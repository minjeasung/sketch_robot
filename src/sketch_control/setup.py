from setuptools import find_packages, setup
from glob import glob

package_name = 'sketch_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config', glob('config/*.rviz')),
        ('share/' + package_name + '/config', glob('config/*.srdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'sketch_ui = sketch_control.sketch_ui:main',
            'moveit_executor = sketch_control.moveit_executor:main',
            'joint_calibrator = sketch_control.joint_calibrator:main',
            'weld_visualizer = sketch_control.weld_visualizer:main',
            'publish_test_waypoint = sketch_control.publish_test_waypoint:main',
            'wall_detector = sketch_control.wall_detector_node:main',
            'target_selector = sketch_control.target_selector_node:main',
            'wall_projector = sketch_control.wall_projector_node:main',
            'sketch_to_waypoints = sketch_control.sketch_to_waypoints_node:main',
            'environment_scanner = sketch_control.environment_scanner_node:main',
            'd405_surface_refiner = sketch_control.d405_surface_refiner_node:main',
            'ft_normal_controller = sketch_control.ft_normal_controller_node:main',
            'depth_to_pointcloud = sketch_control.depth_to_pointcloud_node:main',
        ],
    },
)
