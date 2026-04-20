"""
Sketch UI - 카메라 이미지 위에 스케치 → 3D 웨이포인트 퍼블리시 (벽에 쓰기)

- /camera/image_raw 구독 → tkinter Canvas에 실시간 표시
- /camera/camera_info 구독 → intrinsics (K) 추출
- TF lookup (world → SketchCamera) → extrinsics
  - USD 카메라 convention (-Z forward) → OpenCV optical (+Z forward) 변환:
    TF quaternion q 에 q_x180=(1,0,0,0) 을 오른쪽에서 곱함
- 픽셀 (u,v) → K⁻¹ → optical ray → world ray → x_plane(벽) ���점 → 3D 웨이포인트
- /sketch_waypoints (PoseArray) 퍼블리시, /execute_trajectory (Bool) 퍼블리시
"""
import threading
import numpy as np
from scipy.interpolate import CubicSpline
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool

import tf2_ros
from rclpy.duration import Duration

import tkinter as tk
from PIL import Image as PILImage, ImageTk

from sketch_control.targets import (
    load_objects_config, get_surface_plane, get_target,
    list_enabled_targets, ee_quat_for_target,
)


def quat_multiply(q1, q2):
    """쿼터니언 곱 q1*q2. 각각 [x, y, z, w] 형식."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def quat_rotate(q_xyzw, v):
    """쿼터니언으로 벡터 회전. q = [x, y, z, w]"""
    x, y, z, w = q_xyzw
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


class SketchUI(Node):
    def __init__(self):
        super().__init__("sketch_ui")

        # Publishers
        self.waypoint_pub = self.create_publisher(PoseArray, "/sketch_waypoints", 10)
        self.execute_pub = self.create_publisher(Bool, "/sketch_execute", 10)

        # Subscribers
        self.create_subscription(Image, "/camera/image_raw", self.on_image, 10)
        self.create_subscription(CameraInfo, "/camera/camera_info", self.on_caminfo, 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State
        self.latest_image = None
        self.K = None
        self.img_w = 640
        self.img_h = 480
        self.cam_pos = None       # world 좌표계 카메라 위치
        self.cam_quat_optical = None  # OpenCV optical frame quaternion [x,y,z,w]

        # TF frame name — Isaac Sim이 "World" (대문자) 로 퍼블리시, 자동 탐지
        self.camera_frame = None
        self.world_frame = None
        self._world_frame_candidates = ["World", "world"]
        self._camera_frame_candidates = ["SketchCamera", "sketch_camera", "World/SketchCamera"]

        # Sketch state
        self.points_px = []
        self.is_drawing = False

        # ---- Target registry ----
        self.cfg = load_objects_config()
        self.active_target_name = self.cfg.get("active_target", "wall")
        self.surface_point = None
        self.surface_normal = None
        self.ee_quat = None
        self._update_active_target()

        # 언프로젝션 실패 카운터
        self._fail_counts = {"K_missing": 0, "TF_missing": 0, "parallel": 0, "t_negative": 0}

        # 1초 뒤 사용 가능한 TF 프레임 목록 출력
        self.create_timer(1.0, self._log_available_frames, callback_group=None)
        self._frames_logged = False

        self.get_logger().info("Sketch UI 노드 시작 (TF lookup, 벽에 쓰기 모드)")

    # ---- ROS 콜백 -----------------------------------------------------------
    def on_image(self, msg: Image):
        if msg.encoding not in ("rgb8", "bgr8", "rgba8"):
            return
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        arr = buf.reshape(msg.height, msg.width, -1)
        if msg.encoding == "bgr8":
            arr = arr[:, :, ::-1]
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        self.latest_image = arr.copy()
        self.img_w, self.img_h = msg.width, msg.height

    def on_caminfo(self, msg: CameraInfo):
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        # fx/fy 비대칭이 의심스러우면 fy=fx fallback
        fx, fy = K[0, 0], K[1, 1]
        if abs(fx - fy) / max(fx, fy, 1e-9) > 0.3:
            self.get_logger().warn(f"fx={fx:.1f}, fy={fy:.1f} 비대칭 -> fy=fx fallback")
            K[1, 1] = fx
        self.K = K
        self.img_w, self.img_h = msg.width, msg.height

    # ---- TF 프레임 디버그 ----------------------------------------------------
    def _log_available_frames(self):
        if self._frames_logged:
            return
        frames_str = self.tf_buffer.all_frames_as_string()
        if frames_str.strip():
            self.get_logger().info(f"[TF] 사용 가능한 프레임:\n{frames_str}")
            self._frames_logged = True
        else:
            self.get_logger().info("[TF] 아직 수신된 프레임 없음 — 1초 뒤 재시도")

    # ---- TF lookup ----------------------------------------------------------
    def update_camera_tf(self):
        """world_frame → camera_frame 자동 탐지 후 TF lookup."""
        if self.world_frame is not None and self.camera_frame is not None:
            return self._try_tf_lookup(self.world_frame, self.camera_frame)

        # (world, camera) 조합 모두 시도
        for wf in self._world_frame_candidates:
            for cf in self._camera_frame_candidates:
                if self._try_tf_lookup(wf, cf):
                    self.world_frame = wf
                    self.camera_frame = cf
                    self.get_logger().info(
                        f"[TF] 프레임 확정: '{wf}' -> '{cf}'")
                    return True

        self.get_logger().warn(
            f"[TF] 프레임 못 찾음. frames: "
            f"{self.tf_buffer.all_frames_as_string()[:300]}")
        return False

    def _try_tf_lookup(self, source_frame, target_frame):
        """단일 프레임 조합으로 TF lookup 시도."""
        try:
            tf = self.tf_buffer.lookup_transform(
                source_frame, target_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return False

        t = tf.transform.translation
        self.cam_pos = np.array([t.x, t.y, t.z])

        r = tf.transform.rotation
        # Isaac Sim TF는 이미 [0,0,1]이 forward 방향이므로 q_x180 불필요
        self.cam_quat_optical = np.array([r.x, r.y, r.z, r.w])
        return True

    # ---- 마우스 입력 스무딩 ---------------------------------------------------
    @staticmethod
    def _smooth_mouse_points(points_px, target_spacing_px=5):
        """원본 픽셀 점들을 cubic spline 으로 보간 후 등간격 재샘플링."""
        if len(points_px) < 4:
            return points_px
        pts = np.array(points_px, dtype=float)
        # 중복/매우 가까운 점 제거 (CubicSpline 은 strictly increasing t 필요)
        diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.concatenate([[True], diffs > 0.5])
        pts = pts[keep]
        if len(pts) < 4:
            return list(map(tuple, pts))
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        t = np.concatenate([[0], np.cumsum(dists)])
        total_len = t[-1]
        if total_len < 1e-6:
            return list(map(tuple, pts))
        cs_x = CubicSpline(t, pts[:, 0], bc_type='natural')
        cs_y = CubicSpline(t, pts[:, 1], bc_type='natural')
        n_samples = max(10, min(60, int(total_len / target_spacing_px)))
        t_new = np.linspace(0, total_len, n_samples)
        return list(zip(cs_x(t_new).tolist(), cs_y(t_new).tolist()))

    # ---- Target 관리 --------------------------------------------------------
    def _update_active_target(self):
        """active target 의 surface plane + EE orientation 캐시."""
        obj = get_target(self.cfg, self.active_target_name)
        self.surface_point, self.surface_normal = get_surface_plane(obj)
        self.ee_quat = ee_quat_for_target(obj)
        self.get_logger().info(
            f"[Target] {self.active_target_name} | face={obj['sketch_face']} "
            f"pt={self.surface_point} n={self.surface_normal} "
            f"ee_q={self.ee_quat}"
        )

    # ---- 픽셀 -> 3D 언프로젝션 ---------------------------------------------
    def pixel_to_world(self, u, v):
        """이미지 픽셀 (u, v) → 월드 좌표 3D 점. active target 의 표면 평면과 교차."""
        if self.K is None:
            self._fail_counts["K_missing"] += 1
            return None
        if self.cam_pos is None or self.cam_quat_optical is None:
            self._fail_counts["TF_missing"] += 1
            return None

        K_inv = np.linalg.inv(self.K)
        d_cam = K_inv @ np.array([u, v, 1.0])
        d_world = quat_rotate(self.cam_quat_optical, d_cam)
        d_world = d_world / (np.linalg.norm(d_world) + 1e-9)

        # 평면 교차: (cam_pos + t*d_world - surface_point) · normal = 0
        denom = float(np.dot(d_world, self.surface_normal))
        if abs(denom) < 1e-6:
            self._fail_counts["parallel"] += 1
            return None
        t = float(np.dot(self.surface_point - self.cam_pos, self.surface_normal)) / denom
        if t <= 0:
            self._fail_counts["t_negative"] += 1
            return None
        return self.cam_pos + t * d_world

    # ---- Tkinter 콜백 -------------------------------------------------------
    def on_press(self, event):
        self.is_drawing = True
        self.points_px = [(event.x, event.y)]
        self.canvas.delete("sketch")

    def on_drag(self, event):
        if not self.is_drawing:
            return
        self.points_px.append((event.x, event.y))
        if len(self.points_px) > 1:
            x1, y1 = self.points_px[-2]
            x2, y2 = self.points_px[-1]
            self.canvas.create_line(x1, y1, x2, y2, fill="red", width=3, tags="sketch")

    def on_release(self, event):
        self.is_drawing = False
        self.get_logger().info(f"on_release: 캔버스 포인트 {len(self.points_px)}개")
        if len(self.points_px) < 2:
            return

        # TF 갱신
        if not self.update_camera_tf():
            self.get_logger().warn("on_release: TF lookup 실패 (world -> SketchCamera)")
            return
        if self.K is None:
            self.get_logger().warn("on_release: CameraInfo 미수신 — K 없음")
            return

        # Catmull-Rom 스플라인 스무딩 + 등간격 재샘플링
        sampled = self._smooth_mouse_points(self.points_px)

        pa = PoseArray()
        pa.header.frame_id = "world"
        pa.header.stamp = self.get_clock().now().to_msg()
        valid = 0
        # EE orientation: active target 의 surface normal 에 맞춰 계산됨
        ee_qx, ee_qy, ee_qz, ee_qw = map(float, self.ee_quat)
        for (u, v) in sampled:
            p = self.pixel_to_world(float(u), float(v))
            if p is None:
                continue
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.position.z = float(p[2])
            pose.orientation.x = ee_qx
            pose.orientation.y = ee_qy
            pose.orientation.z = ee_qz
            pose.orientation.w = ee_qw
            pa.poses.append(pose)
            valid += 1

        fc = self._fail_counts
        self.get_logger().info(
            f"언프로젝션: ok={valid}/{len(sampled)} "
            f"[K없음={fc['K_missing']} TF없음={fc['TF_missing']} "
            f"광선평행={fc['parallel']} t음수={fc['t_negative']}]")
        # 카운터 리셋
        self._fail_counts = {"K_missing": 0, "TF_missing": 0, "parallel": 0, "t_negative": 0}

        if valid < 2:
            self.get_logger().warn("유효한 3D 점이 부족합니다 — 퍼블리시 안 함")
            return

        self.waypoint_pub.publish(pa)
        p0 = pa.poses[0]
        self.get_logger().info(
            f"{valid}개 웨이포인트 퍼블리시 → target={self.active_target_name} | "
            f"첫점=({p0.position.x:.3f},{p0.position.y:.3f},{p0.position.z:.3f})")

    def _on_target_change(self, new_name):
        self.active_target_name = new_name
        self._update_active_target()
        self.clear_canvas()

    def send_execute(self):
        msg = Bool()
        msg.data = True
        self.execute_pub.publish(msg)
        self.get_logger().info("실행 명령 전송")

    def clear_canvas(self):
        self.canvas.delete("sketch")
        self.points_px = []

    # ---- GUI 루프 -----------------------------------------------------------
    def update_image(self):
        if self.latest_image is not None:
            pil = PILImage.fromarray(self.latest_image)
            tk_img = ImageTk.PhotoImage(pil)
            self.canvas.delete("img")
            self.canvas.create_image(0, 0, anchor="nw", image=tk_img, tags="img")
            self.canvas.tag_lower("img")
            self._tk_img_ref = tk_img
            if (self.canvas.winfo_width() != self.img_w
                    or self.canvas.winfo_height() != self.img_h):
                self.canvas.config(width=self.img_w, height=self.img_h)
        self.root.after(33, self.update_image)

    def run_gui(self):
        self.root = tk.Tk()
        self.root.title("Sketch Robot Control (camera un-projection)")

        self.canvas = tk.Canvas(self.root, width=self.img_w, height=self.img_h, bg="black")
        self.canvas.pack(pady=6)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack()
        tk.Button(btn_frame, text="실행", command=self.send_execute,
                  bg="#2e7d32", fg="white", width=10).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="지우기", command=self.clear_canvas,
                  bg="#616161", fg="white", width=10).pack(side=tk.LEFT, padx=4)

        # ---- 작업 대상 드롭다운 ----
        target_frame = tk.Frame(self.root)
        target_frame.pack(pady=6)
        tk.Label(target_frame, text="작업 대상:").pack(side=tk.LEFT, padx=4)
        target_names = list_enabled_targets(self.cfg)
        self.target_var = tk.StringVar(value=self.active_target_name)
        tk.OptionMenu(target_frame, self.target_var, *target_names,
                      command=self._on_target_change).pack(side=tk.LEFT)

        self.root.after(200, self.update_image)
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = SketchUI()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    node.run_gui()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
