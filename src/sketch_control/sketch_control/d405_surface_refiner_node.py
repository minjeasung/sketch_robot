"""
d405_surface_refiner_node - refine the active work-area plane with wrist D405.

ZED owns global perception: target object, work-area corners, and obstacle map.
The wrist-mounted D405 is only used once the robot is near the surface, where
millimeter-scale depth matters for roller contact.

Inputs:
  /perception/work_area_plane       PoseStamped, ZED/global work area plane
  /perception/work_area_corners     PoseArray, TL/TR/BR/BL in the same frame
  D405 PointCloud2                  configurable, default realsense D405 topic

Outputs:
  /perception/work_area_plane_refined  PoseStamped, same frame as ZED plane
  /perception/d405_surface_refinement_status  JSON status string
"""
import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from sketch_control.pointcloud_utils import ransac_plane, voxel_downsample
from sketch_control.rotation_utils import quat_apply, quat_from_matrix


WORK_AREA_PLANE_TOPIC = "/perception/work_area_plane"
WORK_AREA_CORNERS_TOPIC = "/perception/work_area_corners"
REFINED_PLANE_TOPIC = "/perception/work_area_plane_refined"
STATUS_TOPIC = "/perception/d405_surface_refinement_status"

LATCHED_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def _pose_normal(pose):
    q = pose.orientation
    normal = quat_apply([q.x, q.y, q.z, q.w], [0.0, 0.0, 1.0])
    normal = np.asarray(normal, dtype=float)
    return normal / (np.linalg.norm(normal) + 1e-12)


def _normal_to_quaternion(normal):
    z_axis = np.asarray(normal, dtype=float)
    z_axis /= np.linalg.norm(z_axis) + 1e-12
    seed = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(seed, z_axis))) > 0.95:
        seed = np.array([1.0, 0.0, 0.0], dtype=float)
    x_axis = np.cross(seed, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-12
    return quat_from_matrix(np.column_stack([x_axis, y_axis, z_axis]))


class D405SurfaceRefinerNode(Node):
    def __init__(self):
        super().__init__("d405_surface_refiner_node")

        self.declare_parameter("cloud_topic", "/d405/d405/depth/color/points")
        self.declare_parameter("max_points", 120000)
        self.declare_parameter("min_points", 150)
        self.declare_parameter("min_inliers", 100)
        self.declare_parameter("voxel_size_m", 0.003)
        self.declare_parameter("ransac_distance_m", 0.004)
        self.declare_parameter("ransac_iterations", 180)
        self.declare_parameter("roi_margin_m", 0.08)
        self.declare_parameter("roi_plane_window_m", 0.12)
        self.declare_parameter("max_plane_shift_m", 0.08)
        self.declare_parameter("max_normal_angle_deg", 12.0)
        self.declare_parameter("min_update_period_s", 0.15)

        self.cloud_topic = str(self.get_parameter("cloud_topic").value)
        self.max_points = int(self.get_parameter("max_points").value)
        self.min_points = int(self.get_parameter("min_points").value)
        self.min_inliers = int(self.get_parameter("min_inliers").value)
        self.voxel_size_m = float(self.get_parameter("voxel_size_m").value)
        self.ransac_distance_m = float(self.get_parameter("ransac_distance_m").value)
        self.ransac_iterations = int(self.get_parameter("ransac_iterations").value)
        self.roi_margin_m = float(self.get_parameter("roi_margin_m").value)
        self.roi_plane_window_m = float(self.get_parameter("roi_plane_window_m").value)
        self.max_plane_shift_m = float(self.get_parameter("max_plane_shift_m").value)
        self.max_normal_angle_rad = math.radians(
            float(self.get_parameter("max_normal_angle_deg").value)
        )
        self.min_update_period_s = float(
            self.get_parameter("min_update_period_s").value
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_plane = None
        self.latest_corners = None
        self._last_update = 0.0

        self.create_subscription(
            PoseStamped, WORK_AREA_PLANE_TOPIC, self._on_work_area_plane, LATCHED_QOS
        )
        self.create_subscription(
            PoseArray, WORK_AREA_CORNERS_TOPIC, self._on_work_area_corners, LATCHED_QOS
        )
        self.create_subscription(
            PointCloud2, self.cloud_topic, self._on_cloud, qos_profile_sensor_data
        )

        self.refined_pub = self.create_publisher(
            PoseStamped, REFINED_PLANE_TOPIC, LATCHED_QOS
        )
        self.status_pub = self.create_publisher(String, STATUS_TOPIC, 10)

        self.get_logger().info(
            "D405 surface refiner 시작\n"
            f"  cloud   : {self.cloud_topic}\n"
            f"  zed     : {WORK_AREA_PLANE_TOPIC}, {WORK_AREA_CORNERS_TOPIC}\n"
            f"  refined : {REFINED_PLANE_TOPIC}"
        )

    def _on_work_area_plane(self, msg):
        self.latest_plane = msg

    def _on_work_area_corners(self, msg):
        if len(msg.poses) >= 4:
            self.latest_corners = msg

    def _on_cloud(self, msg):
        now = time.monotonic()
        if now - self._last_update < self.min_update_period_s:
            return
        self._last_update = now

        if self.latest_plane is None:
            self._publish_status(False, "waiting_for_work_area_plane")
            return

        target_frame = self.latest_plane.header.frame_id or "zed_left_camera_frame"
        source_frame = msg.header.frame_id or ""
        if not source_frame:
            self._publish_status(False, "cloud_frame_empty")
            return

        same_frame = target_frame == source_frame
        transform = None if same_frame else self._lookup_transform(target_frame, source_frame)
        if transform is None and not same_frame:
            self._publish_status(
                False, "tf_missing", target_frame=target_frame,
                source_frame=source_frame,
            )
            return

        points = self._cloud_to_numpy(msg)
        if points is None or points.shape[0] < self.min_points:
            self._publish_status(False, "too_few_cloud_points",
                                 points=0 if points is None else points.shape[0])
            return

        points = self._transform_points(points, transform)

        p0 = np.array([
            self.latest_plane.pose.position.x,
            self.latest_plane.pose.position.y,
            self.latest_plane.pose.position.z,
        ], dtype=float)
        n0 = _pose_normal(self.latest_plane.pose)
        roi = self._select_surface_roi(points, p0, n0, target_frame)
        if roi.shape[0] < self.min_points:
            self._publish_status(False, "too_few_roi_points",
                                 points=int(roi.shape[0]))
            return

        if self.voxel_size_m > 0.0:
            roi = voxel_downsample(roi, self.voxel_size_m)
        if roi.shape[0] < self.min_points:
            self._publish_status(False, "too_few_voxel_points",
                                 points=int(roi.shape[0]))
            return

        try:
            model, inliers = ransac_plane(
                roi, self.ransac_distance_m, self.ransac_iterations)
        except Exception as exc:
            self._publish_status(False, "ransac_failed", detail=str(exc))
            return

        if len(inliers) < self.min_inliers:
            self._publish_status(False, "too_few_inliers",
                                 points=int(roi.shape[0]),
                                 inliers=int(len(inliers)))
            return

        normal = np.asarray(model[:3], dtype=float)
        normal /= np.linalg.norm(normal) + 1e-12
        if float(np.dot(normal, n0)) < 0.0:
            normal = -normal
        inlier_center = roi[inliers].mean(axis=0)
        d = -float(np.dot(normal, inlier_center))

        dot = float(np.clip(np.dot(normal, n0), -1.0, 1.0))
        angle = math.acos(dot)
        if angle > self.max_normal_angle_rad:
            self._publish_status(
                False, "normal_angle_rejected",
                angle_deg=math.degrees(angle),
            )
            return

        signed_shift = float(np.dot(p0, normal) + d)
        if abs(signed_shift) > self.max_plane_shift_m:
            self._publish_status(
                False, "plane_shift_rejected", shift_m=signed_shift)
            return

        refined_center = p0 - signed_shift * normal

        refined = PoseStamped()
        refined.header.stamp = self.get_clock().now().to_msg()
        refined.header.frame_id = target_frame
        refined.pose.position.x = float(refined_center[0])
        refined.pose.position.y = float(refined_center[1])
        refined.pose.position.z = float(refined_center[2])
        qx, qy, qz, qw = _normal_to_quaternion(normal)
        refined.pose.orientation.x = float(qx)
        refined.pose.orientation.y = float(qy)
        refined.pose.orientation.z = float(qz)
        refined.pose.orientation.w = float(qw)
        self.refined_pub.publish(refined)
        self._publish_status(
            True,
            "refined",
            target_frame=target_frame,
            source_frame=source_frame,
            roi_points=int(roi.shape[0]),
            inliers=int(len(inliers)),
            shift_m=signed_shift,
            normal_angle_deg=math.degrees(angle),
        )

    def _select_surface_roi(self, points, p0, n0, target_frame):
        signed = (points - p0) @ n0
        mask = np.abs(signed) <= self.roi_plane_window_m

        basis = self._work_area_basis(target_frame, p0, n0)
        if basis is not None:
            center, u_axis, v_axis, half_u, half_v = basis
            rel = points - center
            u = rel @ u_axis
            v = rel @ v_axis
            mask &= np.abs(u) <= half_u
            mask &= np.abs(v) <= half_v

        return points[mask]

    def _work_area_basis(self, target_frame, p0, n0):
        msg = self.latest_corners
        if msg is None or len(msg.poses) < 4:
            return self._fallback_basis(p0, n0)
        if (msg.header.frame_id or target_frame) != target_frame:
            return self._fallback_basis(p0, n0)

        tl, tr, br, bl = [
            np.array([p.position.x, p.position.y, p.position.z], dtype=float)
            for p in msg.poses[:4]
        ]
        center = (tl + tr + br + bl) / 4.0
        u_vec = ((tr - tl) + (br - bl)) / 2.0
        v_vec = ((bl - tl) + (br - tr)) / 2.0
        width = float(np.linalg.norm(u_vec))
        height = float(np.linalg.norm(v_vec))
        if width < 1e-6 or height < 1e-6:
            return self._fallback_basis(p0, n0)
        return (
            center,
            u_vec / width,
            v_vec / height,
            width / 2.0 + self.roi_margin_m,
            height / 2.0 + self.roi_margin_m,
        )

    def _fallback_basis(self, p0, n0):
        seed = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(seed, n0))) > 0.95:
            seed = np.array([1.0, 0.0, 0.0], dtype=float)
        u_axis = np.cross(n0, seed)
        u_axis /= np.linalg.norm(u_axis) + 1e-12
        v_axis = np.cross(u_axis, n0)
        v_axis /= np.linalg.norm(v_axis) + 1e-12
        return (p0, u_axis, v_axis, 0.30 + self.roi_margin_m,
                0.25 + self.roi_margin_m)

    def _lookup_transform(self, target_frame, source_frame):
        if target_frame == source_frame:
            return None
        try:
            return self.tf_buffer.lookup_transform(
                target_frame, source_frame, Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException:
            return None

    @staticmethod
    def _transform_points(points, transform):
        if transform is None:
            return points
        t = transform.transform.translation
        q = transform.transform.rotation
        return (
            quat_apply([q.x, q.y, q.z, q.w], points)
            + np.array([t.x, t.y, t.z], dtype=float)
        )

    def _cloud_to_numpy(self, msg):
        pts = []
        try:
            for p in point_cloud2.read_points(
                    msg, field_names=("x", "y", "z"), skip_nans=True):
                try:
                    x, y, z = float(p[0]), float(p[1]), float(p[2])
                except Exception:
                    x, y, z = float(p["x"]), float(p["y"]), float(p["z"])
                if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                    pts.append((x, y, z))
        except Exception as exc:
            self._publish_status(False, "pointcloud_parse_failed", detail=str(exc))
            return None
        if not pts:
            return np.empty((0, 3), dtype=np.float32)
        arr = np.asarray(pts, dtype=np.float32)
        if arr.shape[0] > self.max_points:
            step = max(int(math.ceil(arr.shape[0] / self.max_points)), 1)
            arr = arr[::step]
        return arr

    def _publish_status(self, ok, state, **fields):
        payload = {"ok": bool(ok), "state": state}
        payload.update(fields)
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)
        if ok:
            self.get_logger().info(
                "[D405 REFINE] "
                f"shift={payload.get('shift_m', 0.0)*1000:+.1f}mm, "
                f"inliers={payload.get('inliers', 0)}, "
                f"angle={payload.get('normal_angle_deg', 0.0):.2f}deg",
                throttle_duration_sec=1.0,
            )
        else:
            self.get_logger().warn(
                f"[D405 REFINE] {state}: {fields}",
                throttle_duration_sec=2.0,
            )


def main(args=None):
    rclpy.init(args=args)
    node = D405SurfaceRefinerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
