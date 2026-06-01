"""
ft_normal_controller_node - AFT200 force layer for roller contact.

This node does not command the robot directly. It turns raw 6-axis wrench data
into the one value needed first for painting: force along the active wall normal.
MoveIt can then use the status for contact/over-force interlocks, and a later
servo controller can consume the published correction vector for true runtime
admittance.
"""
import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped, WrenchStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import Bool, Float64, String
from tf2_ros import Buffer, TransformException, TransformListener

from sketch_control.rotation_utils import quat_apply


WORK_AREA_PLANE_TOPIC = "/perception/work_area_plane"
WORK_AREA_REFINED_PLANE_TOPIC = "/perception/work_area_plane_refined"

STATUS_TOPIC = "/ft/status"
CONTACT_TOPIC = "/ft/contact"
NORMAL_FORCE_TOPIC = "/ft/normal_force"
CORRECTION_TOPIC = "/ft/admittance_correction"
ZERO_TOPIC = "/ft/zero"

LATCHED_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class FTNormalControllerNode(Node):
    def __init__(self):
        super().__init__("ft_normal_controller_node")

        self.declare_parameter("wrench_topic", "/aft200/ft")
        self.declare_parameter("base_frame", "link0")
        self.declare_parameter("sensor_frame", "tcp")
        self.declare_parameter("auto_zero_samples", 50)
        self.declare_parameter("filter_alpha", 0.15)
        self.declare_parameter("force_sign", 1.0)
        self.declare_parameter("contact_threshold_n", 3.0)
        self.declare_parameter("target_force_n", 10.0)
        self.declare_parameter("warn_force_n", 20.0)
        self.declare_parameter("abort_force_n", 30.0)
        self.declare_parameter("admittance_gain_m_per_n", 0.00025)
        self.declare_parameter("max_correction_m", 0.006)
        self.declare_parameter("refined_surface_fresh_s", 2.0)

        self.wrench_topic = str(self.get_parameter("wrench_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.sensor_frame = str(self.get_parameter("sensor_frame").value)
        self.auto_zero_samples = int(self.get_parameter("auto_zero_samples").value)
        self.filter_alpha = float(self.get_parameter("filter_alpha").value)
        self.force_sign = float(self.get_parameter("force_sign").value)
        self.contact_threshold_n = float(self.get_parameter("contact_threshold_n").value)
        self.target_force_n = float(self.get_parameter("target_force_n").value)
        self.warn_force_n = float(self.get_parameter("warn_force_n").value)
        self.abort_force_n = float(self.get_parameter("abort_force_n").value)
        self.admittance_gain_m_per_n = float(
            self.get_parameter("admittance_gain_m_per_n").value)
        self.max_correction_m = float(self.get_parameter("max_correction_m").value)
        self.refined_surface_fresh_s = float(
            self.get_parameter("refined_surface_fresh_s").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.surface_point = None
        self.surface_normal = None
        self.surface_source = "none"
        self.surface_time = 0.0
        self.filtered_force_base = None
        self.bias_force_base = np.zeros(3, dtype=float)
        self.zero_samples = []
        self.bias_ready = self.auto_zero_samples <= 0
        self.last_wrench_time = 0.0

        self.create_subscription(
            WrenchStamped, self.wrench_topic, self._on_wrench, 10)
        self.create_subscription(
            PoseStamped, WORK_AREA_PLANE_TOPIC, self._on_surface, LATCHED_QOS)
        self.create_subscription(
            PoseStamped, WORK_AREA_REFINED_PLANE_TOPIC,
            self._on_refined_surface, LATCHED_QOS)
        self.create_subscription(Bool, ZERO_TOPIC, self._on_zero, 10)

        self.status_pub = self.create_publisher(String, STATUS_TOPIC, 10)
        self.contact_pub = self.create_publisher(Bool, CONTACT_TOPIC, 10)
        self.normal_force_pub = self.create_publisher(
            Float64, NORMAL_FORCE_TOPIC, 10)
        self.correction_pub = self.create_publisher(
            Vector3Stamped, CORRECTION_TOPIC, 10)

        self.get_logger().info(
            "FT normal controller 시작\n"
            f"  wrench : {self.wrench_topic}\n"
            f"  surface: {WORK_AREA_REFINED_PLANE_TOPIC} 우선, "
            f"fallback={WORK_AREA_PLANE_TOPIC}\n"
            f"  zero   : {ZERO_TOPIC} 또는 startup {self.auto_zero_samples} samples\n"
            f"  target : {self.target_force_n:.1f}N "
            f"(warn={self.warn_force_n:.1f}, abort={self.abort_force_n:.1f})"
        )

    def _on_zero(self, msg):
        if not msg.data:
            return
        self.bias_ready = False
        self.zero_samples = []
        self.filtered_force_base = None
        self.get_logger().warn(
            "[FT] zero requested. 센서를 접촉 없는 상태로 유지하세요.")

    def _on_surface(self, msg):
        if (
            self.surface_source == "d405_refined"
            and time.monotonic() - self.surface_time <= self.refined_surface_fresh_s
        ):
            return
        self._set_surface(msg, "zed")

    def _on_refined_surface(self, msg):
        self._set_surface(msg, "d405_refined")

    def _set_surface(self, msg, source):
        frame = msg.header.frame_id or self.base_frame
        point = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=float)
        q = msg.pose.orientation
        normal = quat_apply([q.x, q.y, q.z, q.w], [0.0, 0.0, 1.0])
        try:
            point, normal = self._transform_point_vector_to_base(point, normal, frame)
        except TransformException as exc:
            self.get_logger().warn(
                f"[FT] surface TF 실패 ({self.base_frame}<-{frame}): {exc}",
                throttle_duration_sec=2.0)
            return
        normal = np.asarray(normal, dtype=float)
        normal /= np.linalg.norm(normal) + 1e-12
        self.surface_point = point
        self.surface_normal = normal
        self.surface_source = source
        self.surface_time = time.monotonic()
        self.get_logger().info(
            f"[FT] active surface={source}, normal=({normal[0]:+.2f},"
            f"{normal[1]:+.2f},{normal[2]:+.2f})",
            throttle_duration_sec=2.0)

    def _on_wrench(self, msg):
        frame = msg.header.frame_id or self.sensor_frame
        f_sensor = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
        ], dtype=float)
        try:
            force_base = self._transform_vector_to_base(f_sensor, frame)
        except TransformException as exc:
            self._publish_status(False, "wrench_tf_missing", detail=str(exc))
            return

        if self.filtered_force_base is None:
            self.filtered_force_base = force_base
        else:
            a = float(np.clip(self.filter_alpha, 0.0, 1.0))
            self.filtered_force_base = (
                (1.0 - a) * self.filtered_force_base + a * force_base
            )

        self.last_wrench_time = time.monotonic()
        if not self.bias_ready:
            self.zero_samples.append(self.filtered_force_base.copy())
            need = max(self.auto_zero_samples, 1)
            if len(self.zero_samples) < need:
                self._publish_status(
                    False, "zeroing", samples=len(self.zero_samples), need=need)
                return
            self.bias_force_base = np.mean(np.asarray(self.zero_samples), axis=0)
            self.bias_ready = True
            self.zero_samples = []
            self.get_logger().info(
                f"[FT] bias set: ({self.bias_force_base[0]:+.2f},"
                f"{self.bias_force_base[1]:+.2f},{self.bias_force_base[2]:+.2f})N")

        if self.surface_normal is None:
            self._publish_status(False, "waiting_for_surface")
            return

        force = self.force_sign * (self.filtered_force_base - self.bias_force_base)
        normal_force = float(np.dot(force, self.surface_normal))
        contact = normal_force >= self.contact_threshold_n

        correction_along_normal = self.admittance_gain_m_per_n * (
            normal_force - self.target_force_n
        )
        correction_along_normal = float(np.clip(
            correction_along_normal,
            -self.max_correction_m,
            self.max_correction_m,
        ))
        correction = self.surface_normal * correction_along_normal

        self._publish_numeric(normal_force, contact, correction)
        self._publish_status(
            True,
            "ok",
            normal_force_n=normal_force,
            contact=contact,
            target_force_n=self.target_force_n,
            warn_force_n=self.warn_force_n,
            abort_force_n=self.abort_force_n,
            surface_source=self.surface_source,
            correction_m=correction_along_normal,
            correction_xyz=correction.tolist(),
            bias_ready=self.bias_ready,
        )

        if normal_force >= self.abort_force_n:
            self.get_logger().error(
                f"[FT] OVER FORCE {normal_force:.1f}N >= {self.abort_force_n:.1f}N",
                throttle_duration_sec=0.5)
        elif normal_force >= self.warn_force_n:
            self.get_logger().warn(
                f"[FT] high force {normal_force:.1f}N >= {self.warn_force_n:.1f}N",
                throttle_duration_sec=0.5)

    def _publish_numeric(self, normal_force, contact, correction):
        force_msg = Float64()
        force_msg.data = float(normal_force)
        self.normal_force_pub.publish(force_msg)

        contact_msg = Bool()
        contact_msg.data = bool(contact)
        self.contact_pub.publish(contact_msg)

        corr_msg = Vector3Stamped()
        corr_msg.header.stamp = self.get_clock().now().to_msg()
        corr_msg.header.frame_id = self.base_frame
        corr_msg.vector.x = float(correction[0])
        corr_msg.vector.y = float(correction[1])
        corr_msg.vector.z = float(correction[2])
        self.correction_pub.publish(corr_msg)

    def _publish_status(self, ok, state, **fields):
        payload = {
            "ok": bool(ok),
            "state": state,
            "bias_ready": bool(self.bias_ready),
        }
        payload.update(fields)
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)

    def _transform_point_vector_to_base(self, point, vector, frame):
        if frame in (self.base_frame, "world", "World"):
            return point, vector
        tf = self.tf_buffer.lookup_transform(
            self.base_frame, frame, Time(), timeout=Duration(seconds=0.05))
        t = tf.transform.translation
        q = tf.transform.rotation
        q_tf = [q.x, q.y, q.z, q.w]
        p_base = quat_apply(q_tf, point) + np.array([t.x, t.y, t.z], dtype=float)
        v_base = quat_apply(q_tf, vector)
        return p_base, v_base

    def _transform_vector_to_base(self, vector, frame):
        if frame in (self.base_frame, "world", "World"):
            return vector
        tf = self.tf_buffer.lookup_transform(
            self.base_frame, frame, Time(), timeout=Duration(seconds=0.05))
        q = tf.transform.rotation
        return quat_apply([q.x, q.y, q.z, q.w], vector)


def main(args=None):
    rclpy.init(args=args)
    node = FTNormalControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
