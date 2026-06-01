"""
sketch_to_waypoints_node — 브라우저 (u, v) → world 3D waypoints.

입력:
  /sketch_pixels          (geometry_msgs/PoseArray)
                          header.frame_id = "wall_front" | "zed_raw"
                          poses[i].position.{x=u, y=v, z=0}
  /perception/work_area_plane
                          (geometry_msgs/PoseStamped, in zed_left_camera_frame)
                          yellow border 로 검출한 작업영역 중심 + normal — 캐시
  /perception/work_area_corners
                          (geometry_msgs/PoseArray, TL/TR/BR/BL, same camera frame)
  TF                      world ← zed_left_camera_frame

출력:
  /sketch_waypoints       (geometry_msgs/PoseArray, frame_id="World")

알고리즘 (wall_front 모드만 — zed_raw 는 TODO):
  1) work area plane 의 centroid/normal 을 world frame 으로 변환 (TF + +Z axis rotation)
  2) wall plane 위 right/up 직교 기저 (world frame) 계산
  3) (u, v) → wall plane 위 롤러 중심점:
       x_plane = (u/VIEW_W - 0.5) * WALL_W
       y_plane = (v/VIEW_H - 0.5) * WALL_H   (v 증가 = down → up 부호 반전)
       p_world = centroid_world + normal*(roller_radius+clearance)
                 + right*x_plane - up*y_plane
  4) EE orientation: tcp local -Y = -normal_world (AFT200+roller EOAT 가 벽 접근)
"""
import numpy as np
import time
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile

from tf2_ros import Buffer, TransformListener, TransformException
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from sketch_control.rotation_utils import quat_from_matrix, quat_to_matrix


# ---- 파라미터 ----------------------------------------------------------------
SKETCH_PIXELS_TOPIC = "/sketch_pixels"
WORK_AREA_TOPIC = "/perception/work_area_plane"
WORK_AREA_REFINED_TOPIC = "/perception/work_area_plane_refined"
WORK_AREA_CORNERS_TOPIC = "/perception/work_area_corners"
WAYPOINTS_TOPIC = "/sketch_waypoints"
MARKERS_TOPIC = "/sketch_markers"

WORLD_FRAME = "World"
CAM_FRAME = "zed_left_camera_frame"

LATCHED_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# wall_projector_node 와 일치해야 함 (벽 평면 영역 + 가상 view 해상도)
WALL_W = 0.5     # m
WALL_H = 0.4     # m
VIEW_W = 800     # px
VIEW_H = 800     # px

# Sketch waypoint semantic: roller rotation-axis center.
# 실제 벽 표면점이 아니라, 롤러 반지름 + 소량 clearance 만큼 free-space 쪽에 둔다.
# 이렇게 해야 MoveIt collision world 에서 벽을 뚫지 않고 "거의 접촉" 경로가 된다.
ROLLER_RADIUS = 0.025
CONTACT_CLEARANCE = 0.002
EOAT_SURFACE_OFFSET = ROLLER_RADIUS + CONTACT_CLEARANCE


def _quat_to_rot(qx, qy, qz, qw):
    return quat_to_matrix([qx, qy, qz, qw])


def _rotation_from_rod_forward_up(forward, world_up=np.array([0.0, 0.0, 1.0])):
    """forward (목표 방향) + world_up → 3x3 rotation matrix.
    EOAT convention: tcp local -Y = AFT200/roller approach direction.
    따라서 local -Y 를 forward 에 맞추고 local +Z 는 가능한 한 world_up 에 둔다.
    """
    f = np.asarray(forward, dtype=float)
    f /= np.linalg.norm(f) + 1e-12
    if abs(float(np.dot(f, world_up))) > 0.99:
        world_up = np.array([1.0, 0.0, 0.0])
    y_axis = -f
    z_axis = np.asarray(world_up, dtype=float)
    z_axis = z_axis - y_axis * float(np.dot(z_axis, y_axis))
    z_axis /= np.linalg.norm(z_axis) + 1e-12
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis) + 1e-12
    return np.column_stack([x_axis, y_axis, z_axis])


class SketchToWaypointsNode(Node):
    def __init__(self):
        super().__init__("sketch_to_waypoints_node")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_work_area = None  # PoseStamped from ZED/global perception
        self.latest_refined_work_area = None  # PoseStamped refined by wrist D405
        self.latest_refined_work_area_time = 0.0
        self.latest_work_area_corners = None

        self.create_subscription(
            PoseStamped, WORK_AREA_TOPIC, self._on_work_area, LATCHED_QOS)
        self.create_subscription(
            PoseStamped, WORK_AREA_REFINED_TOPIC,
            self._on_refined_work_area, LATCHED_QOS)
        self.create_subscription(
            PoseArray, WORK_AREA_CORNERS_TOPIC, self._on_work_area_corners, LATCHED_QOS)
        self.create_subscription(
            PoseArray, SKETCH_PIXELS_TOPIC, self._on_sketch, 10)
        self.pub = self.create_publisher(PoseArray, WAYPOINTS_TOPIC, 10)
        # latch-like: RViz 가 늦게 켜져도 마지막 marker 보이게 transient_local 도 좋지만
        # 매 sketch 마다 갱신하므로 일반 10 depth 로 충분.
        self.marker_pub = self.create_publisher(MarkerArray, MARKERS_TOPIC, 10)

        self.get_logger().info(
            f"sketch_to_waypoints_node 시작\n"
            f"  work  : {WORK_AREA_TOPIC}\n"
            f"          {WORK_AREA_REFINED_TOPIC} (D405 우선, fresh only)\n"
            f"          {WORK_AREA_CORNERS_TOPIC}\n"
            f"  sketch: {SKETCH_PIXELS_TOPIC}\n"
            f"  out   : {WAYPOINTS_TOPIC} (frame={WORLD_FRAME})\n"
            f"          {MARKERS_TOPIC} (MarkerArray, 같은 frame)\n"
            f"  wall rect {WALL_W}×{WALL_H} m ↔ view {VIEW_W}×{VIEW_H} px")

    # ---- callbacks ----
    def _on_work_area(self, msg: PoseStamped):
        self.latest_work_area = msg

    def _on_refined_work_area(self, msg: PoseStamped):
        self.latest_refined_work_area = msg
        self.latest_refined_work_area_time = time.monotonic()

    def _current_work_area(self):
        if (
            self.latest_refined_work_area is not None
            and time.monotonic() - self.latest_refined_work_area_time <= 2.0
        ):
            return self.latest_refined_work_area, "d405_refined"
        return self.latest_work_area, "zed"

    def _on_work_area_corners(self, msg: PoseArray):
        if len(msg.poses) >= 4:
            self.latest_work_area_corners = msg

    def _on_sketch(self, msg: PoseArray):
        view = msg.header.frame_id or ""
        if view != "wall_front":
            # TODO: zed_raw 모드 — 원본 카메라 픽셀에서 ray + plane intersection
            self.get_logger().warn(
                f"sketch frame_id='{view}' — wall_front 만 지원. skip")
            return
        work_area, work_area_source = self._current_work_area()
        if work_area is None:
            self.get_logger().warn(
                f"{WORK_AREA_TOPIC} 미수신 — yellow work area 캐시 없음. skip")
            return
        if not msg.poses:
            self.get_logger().warn("빈 sketch — skip")
            return

        source_frame = (
            work_area.header.frame_id
            or CAM_FRAME
        )

        # TF: world ← camera/surface frame
        T_wc = np.eye(4)
        if source_frame != WORLD_FRAME:
            try:
                tf = self.tf_buffer.lookup_transform(
                    WORLD_FRAME, source_frame, Time(),
                    timeout=Duration(seconds=0.5))
            except TransformException as e:
                self.get_logger().warn(
                    f"TF lookup 실패 ({WORLD_FRAME}←{source_frame}): {e}")
                return

            t = tf.transform.translation
            q = tf.transform.rotation
            T_wc[:3, :3] = _quat_to_rot(q.x, q.y, q.z, q.w)
            T_wc[:3, 3] = [t.x, t.y, t.z]

        # work_area centroid (camera frame → world)
        wp = work_area.pose
        cent_cam_h = np.array(
            [wp.position.x, wp.position.y, wp.position.z, 1.0])
        cent_world = (T_wc @ cent_cam_h)[:3]

        # work_area normal — pose.orientation 의 local +Z
        R_plane_cam = _quat_to_rot(
            wp.orientation.x, wp.orientation.y,
            wp.orientation.z, wp.orientation.w)
        normal_cam = R_plane_cam @ np.array([0.0, 0.0, 1.0])
        normal_world = T_wc[:3, :3] @ normal_cam
        normal_world /= np.linalg.norm(normal_world) + 1e-12

        corners_world = self._corners_world(T_wc, source_frame)
        if corners_world is not None and work_area_source == "d405_refined":
            corners_world = self._project_points_to_plane(
                corners_world, cent_world, normal_world)

        # wall plane 의 right/up (world frame)
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(world_up, normal_world))) > 0.95:
            world_up = np.array([1.0, 0.0, 0.0])
        right = np.cross(normal_world, world_up)
        right /= np.linalg.norm(right) + 1e-12
        up = np.cross(right, normal_world)
        up /= np.linalg.norm(up) + 1e-12

        # EE orientation — tcp local -Y = -normal (벽 접근)
        Rm = _rotation_from_rod_forward_up(-normal_world,
                                           np.array([0.0, 0.0, 1.0]))
        qx, qy, qz, qw = quat_from_matrix(Rm)

        # PoseArray 생성
        out = PoseArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = WORLD_FRAME

        # Roller center: wall surface 에서 normal/free-space 방향으로
        # ROLLER_RADIUS + CONTACT_CLEARANCE 만큼 떨어진 평면.
        surface_offset = normal_world * EOAT_SURFACE_OFFSET
        for p in msg.poses:
            u = float(p.position.x)
            v = float(p.position.y)
            if corners_world is not None:
                p_surface = self._bilinear_point(corners_world, u, v)
            else:
                x_plane = (u / VIEW_W - 0.5) * WALL_W
                y_plane = (v / VIEW_H - 0.5) * WALL_H
                # v 픽셀 ↓ = wall up axis 와 반대
                p_surface = cent_world + right * x_plane - up * y_plane
            p_world = p_surface + surface_offset

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

        # ---- MarkerArray (RViz 시각 검증) ----
        marker_msg = MarkerArray()

        # 0) 이전 marker 모두 삭제 (DELETEALL)
        clear = Marker()
        clear.header.frame_id = WORLD_FRAME
        clear.header.stamp = out.header.stamp
        clear.ns = "sketch"
        clear.action = Marker.DELETEALL
        marker_msg.markers.append(clear)

        # 좌표 points (PoseArray 와 동일 순서)
        pts = [Point(x=ps.position.x, y=ps.position.y, z=ps.position.z)
               for ps in out.poses]

        # 1) SPHERE_LIST — waypoint 위치 (cyan 점)
        spheres = Marker()
        spheres.header.frame_id = WORLD_FRAME
        spheres.header.stamp = out.header.stamp
        spheres.ns = "sketch"
        spheres.id = 0
        spheres.type = Marker.SPHERE_LIST
        spheres.action = Marker.ADD
        spheres.pose.orientation.w = 1.0
        spheres.scale.x = 0.02
        spheres.scale.y = 0.02
        spheres.scale.z = 0.02
        spheres.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)  # cyan
        spheres.points = pts
        marker_msg.markers.append(spheres)

        # 2) LINE_STRIP — waypoint 연결선 (yellow)
        line = Marker()
        line.header.frame_id = WORLD_FRAME
        line.header.stamp = out.header.stamp
        line.ns = "sketch"
        line.id = 1
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.005  # LINE_STRIP: x = 두께
        line.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)  # yellow
        line.points = pts
        marker_msg.markers.append(line)

        self.marker_pub.publish(marker_msg)

        self.get_logger().info(
            f"{len(out.poses)} waypoints published "
            f"(view={view}, plane={work_area_source}, centroid_world="
            f"({cent_world[0]:+.3f},{cent_world[1]:+.3f},{cent_world[2]:+.3f}), "
            f"normal_world=({normal_world[0]:+.2f},{normal_world[1]:+.2f},{normal_world[2]:+.2f}))")

    def _corners_world(self, T_wc, source_frame):
        msg = self.latest_work_area_corners
        if msg is None or len(msg.poses) < 4:
            return None
        if (msg.header.frame_id or source_frame) != source_frame:
            return None
        pts = []
        for pose in msg.poses[:4]:
            p = np.array([
                pose.position.x,
                pose.position.y,
                pose.position.z,
                1.0,
            ])
            pts.append((T_wc @ p)[:3])
        return np.asarray(pts, dtype=float)

    @staticmethod
    def _bilinear_point(corners, u, v):
        # corners: TL, TR, BR, BL. wall_front pixel: u right, v down.
        su = float(np.clip(u / max(VIEW_W - 1, 1), 0.0, 1.0))
        sv = float(np.clip(v / max(VIEW_H - 1, 1), 0.0, 1.0))
        tl, tr, br, bl = corners
        top = tl + (tr - tl) * su
        bottom = bl + (br - bl) * su
        return top + (bottom - top) * sv

    @staticmethod
    def _project_points_to_plane(points, plane_point, normal):
        pts = np.asarray(points, dtype=float)
        n = np.asarray(normal, dtype=float)
        n /= np.linalg.norm(n) + 1e-12
        signed = (pts - np.asarray(plane_point, dtype=float)) @ n
        return pts - signed[:, None] * n


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
