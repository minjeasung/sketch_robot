"""
wall_projector_node — ZED RGB + 선택된 target surface → 작업영역 정면 view 생성.

입력:
  /zed/zed_node/rgb/color/rect/image  (sensor_msgs/Image, rgb8 | bgr8)
  /zed/zed_node/rgb/color/rect/camera_info       (sensor_msgs/CameraInfo)  — K 매트릭스
  /perception/target_surface            (geometry_msgs/PoseStamped)
                                         — sketch 기반 target surface
  /perception/wall_plane                fallback target surface
  /work_area_pixels                     optional sketch 기반 작업영역

출력:
  /perception/wall_front_view          (sensor_msgs/Image, rgb8)
  /perception/work_area_plane          (geometry_msgs/PoseStamped)
  /perception/work_area_corners        (geometry_msgs/PoseArray, TL/TR/BR/BL)

알고리즘:
  1. wall plane parameters (centroid, normal — zed_left_camera_frame)
     normal = quaternion 이 +Z 를 회전시킨 vector
  2. wall plane 위 right/up axes 정의 (camera +Y down 기준 horizontal/vertical)
  3. 4 꼭짓점 (centroid ± W/2 right ± H/2 up) — 작업 영역
  4. K 로 카메라 픽셀 projection (u = fx·X/Z + cx, v = fy·Y/Z + cy)
  5. cv2.getPerspectiveTransform + warpPerspective → 정면 view
  6. /perception/wall_front_view 발행
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, qos_profile_sensor_data

from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from sensor_msgs.msg import Image, CameraInfo


# ---- 파라미터 ----------------------------------------------------------------
INPUT_IMAGE_TOPIC = "/zed/zed_node/rgb/color/rect/image"
INPUT_INFO_TOPIC = "/zed/zed_node/rgb/color/rect/camera_info"
INPUT_WALL_TOPIC = "/perception/wall_plane"
INPUT_TARGET_TOPIC = "/perception/target_surface"
WORK_AREA_PIXELS_TOPIC = "/work_area_pixels"
OUTPUT_TOPIC = "/perception/wall_front_view"
WORK_AREA_TOPIC = "/perception/work_area_plane"
WORK_AREA_CORNERS_TOPIC = "/perception/work_area_corners"

OUTPUT_W = 800   # 가상 정면 view 픽셀
OUTPUT_H = 800

WALL_RECT_W = 0.5  # 벽 평면 위 작업 영역 (m)
WALL_RECT_H = 0.4
DEFAULT_TARGET_VIEW_W = 0.5
DEFAULT_TARGET_VIEW_H = 0.4

YELLOW_HSV_LOWER = np.array([18, 80, 80], dtype=np.uint8)
YELLOW_HSV_UPPER = np.array([45, 255, 255], dtype=np.uint8)
MIN_YELLOW_AREA_PX = 800
MIN_YELLOW_SIDE_PX = 20

LATCHED_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def _quat_z_axis(q):
    """quaternion (x,y,z,w) 가 local +Z 를 회전시킨 vector. wall_detector 의 convention."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
    ], dtype=float)


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


def _order_quad_points(pts):
    """Return points as TL, TR, BR, BL in image coordinates."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def _detect_yellow_quad(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < MIN_YELLOW_AREA_PX:
        return None, area

    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    if min(w, h) < MIN_YELLOW_SIDE_PX:
        return None, area
    return _order_quad_points(cv2.boxPoints(rect)), area


def _intersect_pixel_with_plane(pixel, K, plane_point, normal):
    u, v = float(pixel[0]), float(pixel[1])
    ray = np.linalg.inv(K) @ np.array([u, v, 1.0], dtype=float)
    ray = ray / (np.linalg.norm(ray) + 1e-12)
    denom = float(np.dot(ray, normal))
    if abs(denom) < 1e-9:
        return None
    t = float(np.dot(plane_point, normal)) / denom
    if t <= 0.0:
        return None
    return ray * t


def _plane_axes(normal):
    camera_up_ref = np.array([0.0, -1.0, 0.0])
    if abs(float(np.dot(camera_up_ref, normal))) > 0.99:
        camera_up_ref = np.array([1.0, 0.0, 0.0])
    right = np.cross(camera_up_ref, normal)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(normal, right)
    up /= np.linalg.norm(up) + 1e-12
    return right, up


def _project_points(points_3d, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    pts = np.asarray(points_3d, dtype=float)
    if np.any(pts[:, 2] <= 1e-6):
        return None
    out = np.zeros((pts.shape[0], 2), dtype=np.float32)
    out[:, 0] = fx * pts[:, 0] / pts[:, 2] + cx
    out[:, 1] = fy * pts[:, 1] / pts[:, 2] + cy
    return out


def _decode_image(msg):
    """sensor_msgs/Image (rgb8 | bgr8) → numpy HxWx3 (RGB)."""
    h, w = msg.height, msg.width
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding == "rgb8":
        return arr.reshape(h, w, 3).copy()
    if msg.encoding == "bgra8":
        return cv2.cvtColor(arr.reshape(h, w, 4), cv2.COLOR_BGRA2RGB)
    if msg.encoding == "rgba8":
        return arr.reshape(h, w, 4)[:, :, :3].copy()
    if msg.encoding == "bgr8":
        return cv2.cvtColor(arr.reshape(h, w, 3), cv2.COLOR_BGR2RGB)
    raise ValueError(f"unsupported encoding: {msg.encoding}")


def _encode_rgb(rgb, frame_id, stamp):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = rgb.shape[:2]
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = rgb.tobytes()
    return msg


class WallProjectorNode(Node):
    def __init__(self):
        super().__init__("wall_projector_node")

        self.create_subscription(
            Image, INPUT_IMAGE_TOPIC, self._on_image, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, INPUT_INFO_TOPIC, self._on_info, qos_profile_sensor_data)
        self.create_subscription(
            PoseStamped, INPUT_WALL_TOPIC, self._on_wall, 10)
        self.create_subscription(
            PoseStamped, INPUT_TARGET_TOPIC, self._on_target_surface, LATCHED_QOS)
        self.create_subscription(
            PoseArray, WORK_AREA_PIXELS_TOPIC, self._on_work_area_pixels, 10)
        self.front_pub = self.create_publisher(Image, OUTPUT_TOPIC, 10)
        self.work_area_pub = self.create_publisher(
            PoseStamped, WORK_AREA_TOPIC, LATCHED_QOS)
        self.work_area_corners_pub = self.create_publisher(
            PoseArray, WORK_AREA_CORNERS_TOPIC, LATCHED_QOS)

        self.K = None
        self.latest_surface = None  # (centroid, normal, frame_id, source)
        self.latest_work_area_pixels = None
        self.locked_work_area = None
        self._warned_behind_camera = False
        self._warned_K_missing = False
        self._yellow_warn_count = 0

        self.get_logger().info(
            f"wall_projector_node 시작\n"
            f"  in : {INPUT_IMAGE_TOPIC}\n"
            f"       {INPUT_INFO_TOPIC}\n"
            f"       {INPUT_TARGET_TOPIC} (fallback={INPUT_WALL_TOPIC})\n"
            f"       {WORK_AREA_PIXELS_TOPIC}\n"
            f"  out: {OUTPUT_TOPIC}  ({OUTPUT_W}×{OUTPUT_H}, "
            f"yellow/sketch work area)\n"
            f"       {WORK_AREA_TOPIC}, {WORK_AREA_CORNERS_TOPIC}")

    def _on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    def _on_wall(self, msg: PoseStamped):
        if self.latest_surface is not None and self.latest_surface[3] == "target":
            return
        self._cache_surface(msg, "wall")

    def _on_target_surface(self, msg: PoseStamped):
        self._clear_locked_work_area("target surface updated")
        self._cache_surface(msg, "target")

    def _cache_surface(self, msg: PoseStamped, source: str):
        centroid = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=float)
        normal = _quat_z_axis(msg.pose.orientation)
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-6:
            return
        normal = normal / n_norm
        self.latest_surface = (
            centroid, normal,
            msg.header.frame_id or "zed_left_camera_frame",
            source,
        )

    def _on_work_area_pixels(self, msg: PoseArray):
        if (msg.header.frame_id or "") != "zed_raw":
            self.get_logger().warn(
                f"work_area frame_id='{msg.header.frame_id}' — zed_raw 만 지원")
            return
        if not msg.poses:
            return
        self.latest_work_area_pixels = msg
        self._clear_locked_work_area("new work area sketch")
        self.get_logger().info(
            f"work_area sketch 수신: {len(msg.poses)} px")

    def _clear_locked_work_area(self, reason: str):
        if self.locked_work_area is not None:
            self.get_logger().info(f"work_area lock 해제: {reason}")
        self.locked_work_area = None

    def _on_image(self, msg: Image):
        if self.K is None:
            if not self._warned_K_missing:
                self._warned_K_missing = True
                self.get_logger().warn(
                    f"{INPUT_INFO_TOPIC} 미수신 — projection 보류")
            return
        if self.latest_surface is None:
            return

        try:
            rgb = _decode_image(msg)
        except Exception as e:
            self.get_logger().warn(f"image decode 실패: {e}")
            return

        centroid, normal, _frame, source = self.latest_surface

        if self.locked_work_area is not None:
            lock = self.locked_work_area
            src_pts = lock["src_pts"].copy()
            corners_3d = lock["corners_3d"].copy()
            normal = lock["normal"].copy()
            _frame = lock["frame_id"]
            mode = lock["mode"]
        else:
            src_pts, corners_3d, mode = self._choose_work_area(
                rgb, centroid, normal)
            if (
                src_pts is not None
                and corners_3d is not None
                and self.latest_work_area_pixels is not None
            ):
                self.locked_work_area = {
                    "src_pts": np.asarray(src_pts, dtype=np.float32).copy(),
                    "corners_3d": np.asarray(corners_3d, dtype=float).copy(),
                    "normal": np.asarray(normal, dtype=float).copy(),
                    "frame_id": _frame,
                    "mode": f"locked:{mode}",
                }
                mode = self.locked_work_area["mode"]
                self.get_logger().info(
                    f"work_area lock 설정 ({mode}) — 이후 robot occlusion 에도 "
                    "plane/corners 재검출 안 함")

        if src_pts is None or corners_3d is None:
            if source == "target":
                # target 은 선택됐지만 작업영역 sketch/yellow 가 아직 없는 상태.
                self._yellow_warn_count += 1
                if self._yellow_warn_count % 30 == 1:
                    self.get_logger().warn(
                        "work area 미지정 — ZED Raw 에서 Work Area 를 그리거나 "
                        "노란 사각형을 보여주세요")
            else:
                self._yellow_warn_count += 1
                if self._yellow_warn_count % 30 == 1:
                    self.get_logger().warn(
                        "yellow/work area 미검출 — front_view publish 보류")
            return
        self._yellow_warn_count = 0
        work_center = corners_3d.mean(axis=0)

        dst_pts = np.array([
            [0.0,            0.0],
            [OUTPUT_W - 1.0, 0.0],
            [OUTPUT_W - 1.0, OUTPUT_H - 1.0],
            [0.0,            OUTPUT_H - 1.0],
        ], dtype=np.float32)

        try:
            H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            front = cv2.warpPerspective(
                rgb, H, (OUTPUT_W, OUTPUT_H),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        except Exception as e:
            self.get_logger().warn(f"warpPerspective 실패: {e}")
            return

        out = _encode_rgb(front, "wall_front_view", msg.header.stamp)
        self.front_pub.publish(out)

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = _frame
        pose.pose.position.x = float(work_center[0])
        pose.pose.position.y = float(work_center[1])
        pose.pose.position.z = float(work_center[2])
        qx, qy, qz, qw = _normal_to_quaternion(normal)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        self.work_area_pub.publish(pose)
        self._publish_corners(corners_3d, _frame, msg.header.stamp)

        self.get_logger().info(
            f"front_view/work_area publish ({mode}, surface={source}) "
            f"center=({work_center[0]:+.3f},{work_center[1]:+.3f},"
            f"{work_center[2]:+.3f})")

    def _choose_work_area(self, rgb, centroid, normal):
        if self.latest_work_area_pixels is not None:
            src_pts, area = _detect_yellow_quad(rgb)
            if src_pts is not None and self._yellow_matches_work_area_sketch(src_pts):
                corners = self._corners_from_pixels(src_pts, centroid, normal)
                if corners is not None:
                    return src_pts, corners, f"yellow+sketch(area={area:.0f}px)"

            result = self._work_area_from_sketch(centroid, normal)
            if result[0] is not None:
                return result[0], result[1], "sketch-fallback"

        src_pts, area = _detect_yellow_quad(rgb)
        if src_pts is not None:
            corners = self._corners_from_pixels(src_pts, centroid, normal)
            if corners is not None:
                return src_pts, corners, f"yellow(area={area:.0f}px)"

        return None, None, "none"

    def _yellow_matches_work_area_sketch(self, yellow_pts):
        msg = self.latest_work_area_pixels
        if msg is None or not msg.poses:
            return True
        pts = np.array([[p.position.x, p.position.y] for p in msg.poses], dtype=float)
        u0, v0 = pts.min(axis=0)
        u1, v1 = pts.max(axis=0)
        margin = 30.0
        center = np.asarray(yellow_pts, dtype=float).mean(axis=0)
        return (
            u0 - margin <= center[0] <= u1 + margin
            and v0 - margin <= center[1] <= v1 + margin
        )

    def _work_area_from_sketch(self, centroid, normal):
        msg = self.latest_work_area_pixels
        pts_px = np.array([
            [p.position.x, p.position.y] for p in msg.poses
        ], dtype=float)
        u0 = float(np.clip(pts_px[:, 0].min(), 0, None))
        u1 = float(np.clip(pts_px[:, 0].max(), 0, None))
        v0 = float(np.clip(pts_px[:, 1].min(), 0, None))
        v1 = float(np.clip(pts_px[:, 1].max(), 0, None))
        if (u1 - u0) < MIN_YELLOW_SIDE_PX or (v1 - v0) < MIN_YELLOW_SIDE_PX:
            return None, None, "sketch-too-small"

        src_pts = np.array([
            [u0, v0],
            [u1, v0],
            [u1, v1],
            [u0, v1],
        ], dtype=np.float32)
        corners = self._corners_from_pixels(src_pts, centroid, normal)
        if corners is None:
            return None, None, "sketch-intersection-failed"
        return src_pts, corners, "sketch"

    def _corners_from_pixels(self, src_pts, centroid, normal):
        corners = []
        for px in src_pts:
            p = _intersect_pixel_with_plane(px, self.K, centroid, normal)
            if p is None:
                return None
            if p[2] <= 1e-6:
                if not self._warned_behind_camera:
                    self._warned_behind_camera = True
                    self.get_logger().warn(
                        "work area corner 중 Z<=0 — surface 가 카메라 뒤. skip")
                return None
            corners.append(p)
        return np.asarray(corners, dtype=float)

    def _publish_corners(self, corners_3d, frame_id, stamp):
        pa = PoseArray()
        pa.header.stamp = stamp
        pa.header.frame_id = frame_id
        for p in corners_3d:
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.position.z = float(p[2])
            pose.orientation.w = 1.0
            pa.poses.append(pose)
        self.work_area_corners_pub.publish(pa)


def main(args=None):
    rclpy.init(args=args)
    node = WallProjectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
