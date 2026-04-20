import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/minjea/sketch_robot_ws/install/ros_tcp_endpoint'
