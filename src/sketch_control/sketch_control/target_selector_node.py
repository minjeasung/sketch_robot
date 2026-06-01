"""
target_selector_node — ZED 전역 이미지 위 사용자 스케치로 작업대상 표면 선택.

입력:
  /target_selection_pixels              geometry_msgs/PoseArray, frame_id="zed_raw"
  /zed/zed_node/depth/depth_registered  sensor_msgs/Image
  /zed/zed_node/depth/camera_info       sensor_msgs/CameraInfo

출력:
  /perception/target_surface            geometry_msgs/PoseStamped

사용자가 벽/판/물체 위에 대략 동그라미/박스를 그리면, 그 stroke 를 감싸는
픽셀 영역의 depth 를 3D point 로 변환하고 RANSAC plane 을 추정한다. 이 plane 이
이후 작업영역 projection 과 경로 생성의 기준 surface 가 된다.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, qos_profile_sensor_data

from geometry_msgs.msg import PoseArray, PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from sketch_control.pointcloud_utils import ransac_plane


TARGET_SELECTION_TOPIC = "/target_selection_pixels"
DEPTH_TOPIC = "/zed/zed_node/depth/depth_registered"
CAMERA_INFO_TOPIC = "/zed/zed_node/depth/camera_info"
TARGET_SURFACE_TOPIC = "/perception/target_surface"

ROI_PADDING_PX = 16
ROI_SAMPLE_STRIDE = 4
MIN_TARGET_POINTS = 80
RANSAC_DIST = 0.015
RANSAC_ITERS = 2000


LATCHED_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def _normal_to_quaternion(normal: np.ndarray):
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.dot(z, n))
    if dot > 0.9999:
        return (0.0, 0.0, 0.0, 1.0)
    if dot < -0.9999:
        return (1.0, 0.0, 0.0, 0.0)
    axis = np.cross(z, n)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    angle = np.arccos(dot)
    s = np.sin(angle / 2.0)
    return (float(axis[0] * s), float(axis[1] * s),
            float(axis[2] * s), float(np.cos(angle / 2.0)))


class TargetSelectorNode(Node):
    def __init__(self):
        super().__init__("target_selector_node")
        self.K = None
        self.latest_depth = None
        self.latest_depth_header = None

        self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self._on_info, qos_profile_sensor_data)
        self.create_subscription(
            Image, DEPTH_TOPIC, self._on_depth, qos_profile_sensor_data)
        self.create_subscription(
            PoseArray, TARGET_SELECTION_TOPIC, self._on_selection, 10)

        self.pub = self.create_publisher(
            PoseStamped, TARGET_SURFACE_TOPIC, LATCHED_QOS)

        self.get_logger().info(
            f"target_selector 시작: {TARGET_SELECTION_TOPIC} + depth -> "
            f"{TARGET_SURFACE_TOPIC}")

    def _on_info(self, msg: CameraInfo):
        self.K = np.asarray(msg.k, dtype=float).reshape(3, 3)

    def _on_depth(self, msg: Image):
        try:
            self.latest_depth = self._decode_depth(msg)
            self.latest_depth_header = msg.header
        except Exception as e:
            self.get_logger().warn(f"depth decode 실패: {e}")

    def _on_selection(self, msg: PoseArray):
        if (msg.header.frame_id or "") != "zed_raw":
            self.get_logger().warn(
                f"target selection frame_id='{msg.header.frame_id}' skip")
            return
        if self.K is None or self.latest_depth is None:
            self.get_logger().warn("CameraInfo/depth 미수신 — target selection 보류")
            return
        if not msg.poses:
            self.get_logger().warn("target selection 비어있음")
            return

        pts_px = np.array([
            [p.position.x, p.position.y] for p in msg.poses
        ], dtype=float)
        h, w = self.latest_depth.shape
        u0 = int(max(0, np.floor(pts_px[:, 0].min() - ROI_PADDING_PX)))
        u1 = int(min(w - 1, np.ceil(pts_px[:, 0].max() + ROI_PADDING_PX)))
        v0 = int(max(0, np.floor(pts_px[:, 1].min() - ROI_PADDING_PX)))
        v1 = int(min(h - 1, np.ceil(pts_px[:, 1].max() + ROI_PADDING_PX)))
        if u1 <= u0 or v1 <= v0:
            self.get_logger().warn("target selection ROI invalid")
            return

        points = self._points_from_roi(u0, u1, v0, v1)
        if points.shape[0] < MIN_TARGET_POINTS:
            self.get_logger().warn(
                f"target ROI point 부족: {points.shape[0]} < {MIN_TARGET_POINTS}")
            return

        try:
            plane_model, inliers = self._ransac_plane(points)
        except Exception as e:
            self.get_logger().warn(f"target RANSAC 실패: {e}")
            return
        if len(inliers) < MIN_TARGET_POINTS:
            self.get_logger().warn(
                f"target plane inlier 부족: {len(inliers)} < {MIN_TARGET_POINTS}")
            return

        a, b, c, _d = plane_model
        normal = np.array([a, b, c], dtype=float)
        normal /= np.linalg.norm(normal) + 1e-12
        inlier_pts = points[np.asarray(inliers, dtype=int)]
        centroid = inlier_pts.mean(axis=0)

        # Camera frame 원점 쪽을 free-space 로 본다.
        if float(np.dot(normal, centroid)) > 0.0:
            normal = -normal

        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = (
            self.latest_depth_header.frame_id
            if self.latest_depth_header is not None
            else "zed_left_camera_frame_optical"
        )
        out.pose.position.x = float(centroid[0])
        out.pose.position.y = float(centroid[1])
        out.pose.position.z = float(centroid[2])
        qx, qy, qz, qw = _normal_to_quaternion(normal)
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw
        self.pub.publish(out)

        self.get_logger().info(
            f"target_surface publish: centroid=({centroid[0]:+.3f},"
            f"{centroid[1]:+.3f},{centroid[2]:+.3f}) "
            f"normal=({normal[0]:+.2f},{normal[1]:+.2f},{normal[2]:+.2f}) "
            f"inliers={len(inliers)}/{points.shape[0]}")

    def _points_from_roi(self, u0, u1, v0, v1):
        depth = self.latest_depth
        ys = np.arange(v0, v1 + 1, ROI_SAMPLE_STRIDE, dtype=np.float32)
        xs = np.arange(u0, u1 + 1, ROI_SAMPLE_STRIDE, dtype=np.float32)
        uu, vv = np.meshgrid(xs, ys)
        z = depth[vv.astype(int), uu.astype(int)]
        valid = np.isfinite(z) & (z > 0.15) & (z < 5.0)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy
        return np.column_stack((x[valid], y[valid], z[valid])).astype(np.float32)

    @staticmethod
    def _ransac_plane(points):
        return ransac_plane(points, RANSAC_DIST, RANSAC_ITERS)

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
    node = TargetSelectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
