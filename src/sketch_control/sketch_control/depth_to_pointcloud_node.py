"""
depth_to_pointcloud_node — Isaac Sim depth image 를 ZED 호환 PointCloud2 로 변환.

실제 ZED wrapper 는 보통 /zed/zed_node/point_cloud/cloud_registered 를 직접
발행한다. Isaac Sim native ROS2 Camera Helper 는 RGB/depth/camera_info 만
발행하므로, 시뮬레이션에서 perception scanner 를 검증하기 위한 보조 노드다.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2


DEFAULT_DEPTH_TOPIC = "/zed/zed_node/depth/depth_registered"
DEFAULT_CAMERA_INFO_TOPIC = "/zed/zed_node/depth/camera_info"
DEFAULT_POINT_CLOUD_TOPIC = "/zed/zed_node/point_cloud/cloud_registered"


class DepthToPointCloudNode(Node):
    def __init__(self):
        super().__init__("depth_to_pointcloud_node")
        self.declare_parameter("depth_topic", DEFAULT_DEPTH_TOPIC)
        self.declare_parameter("camera_info_topic", DEFAULT_CAMERA_INFO_TOPIC)
        self.declare_parameter("point_cloud_topic", DEFAULT_POINT_CLOUD_TOPIC)
        self.declare_parameter("stride", 3)
        self.declare_parameter("min_depth_m", 0.15)
        self.declare_parameter("max_depth_m", 5.0)

        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(
            self.get_parameter("camera_info_topic").value)
        self.point_cloud_topic = str(
            self.get_parameter("point_cloud_topic").value)
        self.stride = max(1, int(self.get_parameter("stride").value))
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)

        self.K = None
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._on_info,
            qos_profile_sensor_data)
        self.create_subscription(
            Image, self.depth_topic, self._on_depth, qos_profile_sensor_data)
        self.pub = self.create_publisher(PointCloud2, self.point_cloud_topic, 10)
        self.get_logger().info(
            f"depth_to_pointcloud 시작: {self.depth_topic} + "
            f"{self.camera_info_topic} -> {self.point_cloud_topic} "
            f"(stride={self.stride})")

    def _on_info(self, msg: CameraInfo):
        self.K = np.asarray(msg.k, dtype=float).reshape(3, 3)

    def _on_depth(self, msg: Image):
        if self.K is None:
            return
        try:
            depth = self._decode_depth(msg)
        except Exception as e:
            self.get_logger().warn(f"depth decode 실패: {e}")
            return

        h, w = depth.shape
        ys = np.arange(0, h, self.stride, dtype=np.float32)
        xs = np.arange(0, w, self.stride, dtype=np.float32)
        uu, vv = np.meshgrid(xs, ys)
        z = depth[::self.stride, ::self.stride]

        valid = (
            np.isfinite(z)
            & (z > self.min_depth_m)
            & (z < self.max_depth_m)
        )
        if not np.any(valid):
            return

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy
        pts = np.column_stack((x[valid], y[valid], z[valid])).astype(np.float32)

        cloud = pc2.create_cloud_xyz32(msg.header, pts.tolist())
        self.pub.publish(cloud)

    @staticmethod
    def _decode_depth(msg: Image):
        h, w = msg.height, msg.width
        if msg.encoding in ("32FC1", "32fc1"):
            row_floats = msg.step // 4
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape(h, row_floats)
            return arr[:, :w].copy()
        if msg.encoding in ("16UC1", "mono16"):
            row_uint16 = msg.step // 2
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, row_uint16)
            return arr[:, :w].astype(np.float32) * 0.001
        raise ValueError(f"unsupported depth encoding: {msg.encoding}")


def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointCloudNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
