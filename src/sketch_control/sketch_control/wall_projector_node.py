"""
wall_projector_node — ZED RGB + 검출된 wall_plane → 벽 정면 가상 view 생성.

입력:
  /zed/zed_node/rgb/color/rect/image  (sensor_msgs/Image, rgb8 | bgr8)
  /zed/zed_node/rgb/color/rect/camera_info       (sensor_msgs/CameraInfo)  — K 매트릭스
  /perception/wall_plane               (geometry_msgs/PoseStamped)
                                        — pose.position = wall centroid
                                        — pose.orientation = +Z 가 wall normal

출력:
  /perception/wall_front_view          (sensor_msgs/Image, rgb8)

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
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo


# ---- 파라미터 ----------------------------------------------------------------
INPUT_IMAGE_TOPIC = "/zed/zed_node/rgb/color/rect/image"
INPUT_INFO_TOPIC = "/zed/zed_node/rgb/color/rect/camera_info"
INPUT_WALL_TOPIC = "/perception/wall_plane"
OUTPUT_TOPIC = "/perception/wall_front_view"

OUTPUT_W = 800   # 가상 정면 view 픽셀
OUTPUT_H = 800

WALL_RECT_W = 0.6  # 벽 평면 위 작업 영역 (m)
WALL_RECT_H = 0.6


def _quat_z_axis(q):
    """quaternion (x,y,z,w) 가 local +Z 를 회전시킨 vector. wall_detector 의 convention."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
    ], dtype=float)


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
        self.front_pub = self.create_publisher(Image, OUTPUT_TOPIC, 10)

        self.K = None
        self.latest_wall = None  # (centroid: ndarray(3), normal: ndarray(3), frame_id: str)
        self._warned_behind_camera = False
        self._warned_K_missing = False

        self.get_logger().info(
            f"wall_projector_node 시작\n"
            f"  in : {INPUT_IMAGE_TOPIC}\n"
            f"       {INPUT_INFO_TOPIC}\n"
            f"       {INPUT_WALL_TOPIC}\n"
            f"  out: {OUTPUT_TOPIC}  ({OUTPUT_W}×{OUTPUT_H}, "
            f"wall rect {WALL_RECT_W}×{WALL_RECT_H} m)")

    def _on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    def _on_wall(self, msg: PoseStamped):
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
        # zed_left_camera_frame (X-fwd) → zed_left_camera_frame_optical (Z-fwd) 변환
        # optical.x = -frame.y, optical.y = -frame.z, optical.z = +frame.x
        R_frame_to_optical = np.array([
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ])
        centroid = R_frame_to_optical @ centroid
        normal = R_frame_to_optical @ normal
        self.latest_wall = (
            centroid, normal,
            msg.header.frame_id or "zed_left_camera_frame",
        )

    def _on_image(self, msg: Image):
        if self.K is None:
            if not self._warned_K_missing:
                self._warned_K_missing = True
                self.get_logger().warn(
                    f"{INPUT_INFO_TOPIC} 미수신 — projection 보류")
            return
        if self.latest_wall is None:
            return

        try:
            rgb = _decode_image(msg)
        except Exception as e:
            self.get_logger().warn(f"image decode 실패: {e}")
            return

        centroid, normal, _frame = self.latest_wall

        # ZED camera frame: +X right, +Y down, +Z forward (OpenCV convention).
        # 벽 평면의 right/up 정의: camera +Y down 을 reference 로 horizontal axis.
        # right = normal × (camera up reference).
        camera_up_ref = np.array([0.0, -1.0, 0.0])  # camera frame 의 world-up 근사
        if abs(float(np.dot(camera_up_ref, normal))) > 0.99:
            # normal 이 거의 vertical 이면 fallback
            camera_up_ref = np.array([1.0, 0.0, 0.0])
        right = np.cross(camera_up_ref, normal)
        right /= np.linalg.norm(right) + 1e-12
        up = np.cross(normal, right)
        up /= np.linalg.norm(up) + 1e-12

        # 4 꼭짓점 — TL, TR, BR, BL (카메라 시점에서 본 위→아래)
        hw = WALL_RECT_W / 2.0
        hh = WALL_RECT_H / 2.0
        corners_3d = np.array([
            centroid - hw * right + hh * up,   # TL
            centroid + hw * right + hh * up,   # TR
            centroid + hw * right - hh * up,   # BR
            centroid - hw * right - hh * up,   # BL
        ])

        # 카메라 frame 에서 픽셀 projection
        K = self.K
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        src_pts = np.zeros((4, 2), dtype=np.float32)
        for i, P in enumerate(corners_3d):
            if P[2] <= 1e-6:
                if not self._warned_behind_camera:
                    self._warned_behind_camera = True
                    self.get_logger().warn(
                        "wall corner 중 Z<=0 — wall 이 카메라 뒤. skip")
                return
            src_pts[i, 0] = fx * P[0] / P[2] + cx
            src_pts[i, 1] = fy * P[1] / P[2] + cy

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
