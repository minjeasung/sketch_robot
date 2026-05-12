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
        ],
    },
)
