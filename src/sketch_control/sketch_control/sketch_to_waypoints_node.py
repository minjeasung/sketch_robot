"""
sketch_to_waypoints_node — 브라우저 (u, v) → world 3D waypoints.

입력:
  /sketch_pixels          (geometry_msgs/PoseArray)
                          header.frame_id = "wall_front" | "zed_raw"
                          poses[i].position.{x=u, y=v, z=0}
  /perception/wall_plane  (geometry_msgs/PoseStamped, in zed_left_camera_frame)
                          최신 wall plane (centroid + normal via +Z) — 캐시
  TF                      world ← zed_left_camera_frame

출력:
  /sketch_waypoints       (geometry_msgs/PoseArray, frame_id="World")

알고리즘 (wall_front 모드만 — zed_raw 는 TODO):
  1) wall plane 의 centroid/normal 을 world frame 으로 변환 (TF + +Z axis rotation)
  2) wall plane 위 right/up 직교 기저 (world frame) 계산
  3) (u, v) → wall plane 위 점:
       x_plane = (u/VIEW_W - 0.5) * WALL_W
       y_plane = (v/VIEW_H - 0.5) * WALL_H   (v 증가 = down → up 부호 반전)
       p_world = centroid_world + right*x_plane - up*y_plane
  4) EE orientation: forward = -normal_world (EOAT 가 벽 접근), world_up = +Z
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

from tf2_ros import Buffer, TransformListener, TransformException
from geometry_msgs.msg import Pose, PoseArray, PoseStamped


# ---- 파라미터 ----------------------------------------------------------------
SKETCH_PIXELS_TOPIC = "/sketch_pixels"
WALL_PLANE_TOPIC = "/perception/wall_plane"
WAYPOINTS_TOPIC = "/sketch_waypoints"

WORLD_FRAME = "World"
CAM_FRAME = "zed_left_camera_frame"

# wall_projector_node 와 일치해야 함 (벽 평면 영역 + 가상 view 해상도)
WALL_W = 1.0     # m
WALL_H = 1.0     # m
VIEW_W = 800     # px
VIEW_H = 800     # px


def _quat_to_rot(qx, qy, qz, qw):
    return R.from_quat([qx, qy, qz, qw]).as_matrix()


def _rotation_from_forward_up(forward, world_up=np.array([0.0, 0.0, 1.0])):
    """forward (목표 방향) + world_up → 3x3 rotation matrix.
    EE local convention: +X = forward, +Y = right, +Z = up.
    moveit_executor 의 TORCH_MOUNT_AXIS 와 일치 (rod 가 tcp 의 -Y 방향이지만
    여기서는 단순화하여 +X = forward 의 generic EE pose 발행. 차후 매핑은
    moveit_executor 의 _brush_tip_to_tcp 가 처리).
    """
    f = np.asarray(forward, dtype=float)
    f /= np.linalg.norm(f) + 1e-12
    if abs(float(np.dot(f, world_up))) > 0.99:
        world_up = np.array([1.0, 0.0, 0.0])
    right = np.cross(f, np.asarray(world_up, dtype=float))
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, f)
    up /= np.linalg.norm(up) + 1e-12
    # local +X=forward, +Y=right, +Z=up
    return np.column_stack([f, right, up])


class SketchToWaypointsNode(Node):
    def __init__(self):
        super().__init__("sketch_to_waypoints_node")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_wall = None  # PoseStamped in CAM_FRAME

        self.create_subscription(
            PoseStamped, WALL_PLANE_TOPIC, self._on_wall, 10)
        self.create_subscription(
            PoseArray, SKETCH_PIXELS_TOPIC, self._on_sketch, 10)
        self.pub = self.create_publisher(PoseArray, WAYPOINTS_TOPIC, 10)

        self.get_logger().info(
            f"sketch_to_waypoints_node 시작\n"
            f"  wall  : {WALL_PLANE_TOPIC}\n"
            f"  sketch: {SKETCH_PIXELS_TOPIC}\n"
            f"  out   : {WAYPOINTS_TOPIC} (frame={WORLD_FRAME})\n"
            f"  wall rect {WALL_W}×{WALL_H} m ↔ view {VIEW_W}×{VIEW_H} px")

    # ---- callbacks ----
    def _on_wall(self, msg: PoseStamped):
        self.latest_wall = msg

    def _on_sketch(self, msg: PoseArray):
        view = msg.header.frame_id or ""
        if view != "wall_front":
            # TODO: zed_raw 모드 — 원본 카메라 픽셀에서 ray + plane intersection
            self.get_logger().warn(
                f"sketch frame_id='{view}' — wall_front 만 지원. skip")
            return
        if self.latest_wall is None:
            self.get_logger().warn(
                f"{WALL_PLANE_TOPIC} 미수신 — wall_plane 캐시 없음. skip")
            return
        if not msg.poses:
            self.get_logger().warn("빈 sketch — skip")
            return

        # TF: world ← zed_left_camera_frame
        try:
            tf = self.tf_buffer.lookup_transform(
                WORLD_FRAME, CAM_FRAME, Time(),
                timeout=Duration(seconds=0.5))
        except TransformException as e:
            self.get_logger().warn(
                f"TF lookup 실패 ({WORLD_FRAME}←{CAM_FRAME}): {e}")
            return

        T_wc = np.eye(4)
        t = tf.transform.translation
        q = tf.transform.rotation
        T_wc[:3, :3] = _quat_to_rot(q.x, q.y, q.z, q.w)
        T_wc[:3, 3] = [t.x, t.y, t.z]

        # wall_plane centroid (camera frame → world)
        wp = self.latest_wall.pose
        cent_cam_h = np.array(
            [wp.position.x, wp.position.y, wp.position.z, 1.0])
        cent_world = (T_wc @ cent_cam_h)[:3]

        # wall normal — pose.orientation 의 local +Z (wall_detector convention)
        R_plane_cam = _quat_to_rot(
            wp.orientation.x, wp.orientation.y,
            wp.orientation.z, wp.orientation.w)
        normal_cam = R_plane_cam @ np.array([0.0, 0.0, 1.0])
        normal_world = T_wc[:3, :3] @ normal_cam
        normal_world /= np.linalg.norm(normal_world) + 1e-12

        # wall plane 의 right/up (world frame)
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(world_up, normal_world))) > 0.95:
            world_up = np.array([1.0, 0.0, 0.0])
        right = np.cross(normal_world, world_up)
        right /= np.linalg.norm(right) + 1e-12
        up = np.cross(right, normal_world)
        up /= np.linalg.norm(up) + 1e-12

        # EE orientation — forward = -normal (벽 접근)
        Rm = _rotation_from_forward_up(-normal_world,
                                       np.array([0.0, 0.0, 1.0]))
        qx, qy, qz, qw = R.from_matrix(Rm).as_quat()

        # PoseArray 생성
        out = PoseArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = WORLD_FRAME

        for p in msg.poses:
            u = float(p.position.x)
            v = float(p.position.y)
            x_plane = (u / VIEW_W - 0.5) * WALL_W
            y_plane = (v / VIEW_H - 0.5) * WALL_H
            # v 픽셀 ↓ = wall up axis 와 반대
            p_world = cent_world + right * x_plane - up * y_plane

            pose = Pose()
            pose.position.x = float(p_world[0])
            pose.position.y = float(p_world[1])
            pose.position.z = float(p_world[2])
            pose.orientation.x = float(qx)
            pose.orientation.y = float(qy)
            pose.orientation.z = float(qz)
            pose.orientation.w = float(qw)
            out.poses.append(pose)

        self.pub.publish(out)
        self.get_logger().info(
            f"{len(out.poses)} waypoints published "
            f"(view={view}, centroid_world="
            f"({cent_world[0]:+.3f},{cent_world[1]:+.3f},{cent_world[2]:+.3f}), "
            f"normal_world=({normal_world[0]:+.2f},{normal_world[1]:+.2f},{normal_world[2]:+.2f}))")


def main(args=None):
    rclpy.init(args=args)
    node = SketchToWaypointsNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
