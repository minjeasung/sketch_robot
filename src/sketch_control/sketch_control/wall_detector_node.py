"""
wall_detector_node — ZED point cloud 에서 RANSAC 으로 가장 큰 평면 (벽) 검출.

입력  : /zed/zed_node/point_cloud/cloud_registered  (sensor_msgs/PointCloud2, zed_left_camera_frame)
출력  : /perception/wall_plane     (geometry_msgs/PoseStamped, normal=Z 축)
        /perception/wall_inliers   (sensor_msgs/PointCloud2, 검증 시각용)

알고리즘:
  PointCloud2 → numpy → voxel downsample → numpy RANSAC → (a,b,c,d) + inliers
  pose.position = inlier centroid
  pose.orientation = +Z 를 (a,b,c) normal 로 회전시키는 quaternion
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from sketch_control.pointcloud_utils import ransac_plane, voxel_downsample


# RANSAC 파라미터 — 사용자 instruction 의 값.
DISTANCE_THRESHOLD = 0.02
RANSAC_N = 3
NUM_ITERATIONS = 1000

# 입력 cloud 가 너무 크면 RANSAC 이 느려짐 → voxel downsample.
VOXEL_SIZE = 0.01  # 10mm

# 카메라부터 작업공간 max 거리. 이걸 넘는 점은 ground plane / 배경 → RANSAC 에서 제외.
MAX_DISTANCE = 2.0  # m

INPUT_TOPIC = "/zed/zed_node/point_cloud/cloud_registered"
PLANE_TOPIC = "/perception/wall_plane"
INLIER_TOPIC = "/perception/wall_inliers"


def _normal_to_quaternion(normal):
    """+Z = (0,0,1) 을 normal 단위벡터로 회전시키는 quaternion (x,y,z,w)."""
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.dot(z, n))
    if dot > 0.9999:
        return (0.0, 0.0, 0.0, 1.0)
    if dot < -0.9999:
        # 180° flip — 임의의 직교축 (X) 둘레로
        return (1.0, 0.0, 0.0, 0.0)
    axis = np.cross(z, n)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    angle = np.arccos(dot)
    s = np.sin(angle / 2.0)
    return (float(axis[0] * s), float(axis[1] * s), float(axis[2] * s),
            float(np.cos(angle / 2.0)))


class WallDetectorNode(Node):
    def __init__(self):
        super().__init__("wall_detector_node")

        self.create_subscription(
            PointCloud2, INPUT_TOPIC, self._on_cloud, qos_profile_sensor_data)
        self.pose_pub = self.create_publisher(PoseStamped, PLANE_TOPIC, 10)
        self.inliers_pub = self.create_publisher(PointCloud2, INLIER_TOPIC, 10)

        self.get_logger().info(
            f"wall_detector_node 시작 (in={INPUT_TOPIC}, "
            f"out={PLANE_TOPIC} + {INLIER_TOPIC})")

    def _on_cloud(self, msg: PointCloud2):
        # PointCloud2 → numpy (Nx3, finite only).
        # jazzy 의 pc2.read_points 는 structured array 반환 → field 추출 후 column_stack.
        # 더 옛 humble 등은 iterable of tuples → fallback.
        try:
            raw = pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True)
            try:
                points = np.column_stack(
                    [raw["x"], raw["y"], raw["z"]]).astype(np.float32)
            except (TypeError, IndexError, ValueError):
                points = np.array(
                    [(p[0], p[1], p[2]) for p in raw], dtype=np.float32)
            # skip_nans=True 가 jazzy 에서 안 통할 때 대비 — 직접 finite filter
            points = points[np.isfinite(points).all(axis=1)]
        except Exception as e:
            self.get_logger().warn(f"PointCloud2 parse 실패: {e}")
            return

        if points.shape[0] < 100:
            self.get_logger().warn(f"점 수 부족 ({points.shape[0]}) — skip")
            return

        points_ds = voxel_downsample(points, VOXEL_SIZE)
        self.get_logger().info(
            f"[debug] points.shape={points.shape}, downsample={points_ds.shape[0]}")
        if points_ds.shape[0] < 100:
            self.get_logger().warn("downsample 후 점 수 부족 — skip")
            return

        # 카메라부터 너무 먼 점 (작업공간 외) 제외 — RANSAC 이 ground plane 잡는 거 방지
        distances = np.linalg.norm(points_ds, axis=1)  # zed_left_camera_frame 기준 거리
        points_arr = points_ds[distances < MAX_DISTANCE]
        if points_arr.shape[0] < 100:
            self.get_logger().warn(
                f"distance crop (< {MAX_DISTANCE}m) 후 점 수 부족 — skip")
            return

        # RANSAC plane segmentation
        try:
            plane_model, inliers = ransac_plane(
                points_arr, DISTANCE_THRESHOLD, NUM_ITERATIONS)
        except Exception as e:
            self.get_logger().warn(f"segment_plane 실패: {e}")
            return

        a, b, c, d = plane_model
        normal = np.array([a, b, c], dtype=float)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            self.get_logger().warn("invalid plane normal — skip")
            return
        normal = normal / normal_norm

        inlier_pts = points_arr[np.asarray(inliers, dtype=int)]
        if inlier_pts.shape[0] < 50:
            self.get_logger().warn(f"inlier 부족 ({inlier_pts.shape[0]}) — skip")
            return
        centroid = inlier_pts.mean(axis=0)

        # RANSAC plane normal sign is arbitrary. Force it to point toward the
        # camera/free-space side so downstream offsets move away from the wall.
        if float(np.dot(normal, centroid)) > 0.0:
            normal = -normal
            a, b, c, d = -a, -b, -c, -d

        # PoseStamped publish (frame = ZED left, msg.header.stamp 그대로)
        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = msg.header.frame_id or "zed_left_camera_frame"
        pose.pose.position.x = float(centroid[0])
        pose.pose.position.y = float(centroid[1])
        pose.pose.position.z = float(centroid[2])
        qx, qy, qz, qw = _normal_to_quaternion(normal)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        self.pose_pub.publish(pose)

        # Inlier PointCloud2 publish (시각 검증용)
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = pose.header.frame_id
#         inlier_msg = pc2.create_cloud_xyz32(
#             header, inlier_pts.astype(np.float32).tolist())
#         self.inliers_pub.publish(inlier_msg)

        self.get_logger().info(
            f"plane: n=({a:+.3f},{b:+.3f},{c:+.3f}) d={d:+.3f} "
            f"inliers={len(inliers)}/{points_arr.shape[0]} "
            f"centroid=({centroid[0]:+.3f},{centroid[1]:+.3f},{centroid[2]:+.3f})")


def main(args=None):
    rclpy.init(args=args)
    node = WallDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
