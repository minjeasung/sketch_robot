#!/usr/bin/env python3
"""Phase 5 일감 2.5 — Hand-eye calibration script (eye-to-hand).

CALIB_POSE 기준 ±N variation 으로 RB10 을 ~20 pose 이동시키며 매번 (FK from /tf,
AprilTag corners from /detections, K matrix from /zed/.../camera_info) 캡처 →
cv2.solvePnP(ITERATIVE) 으로 카메라 frame 의 tag pose 산출 →
known TCP→tag offset 기반 pose 평균으로 T_camera→robot_base 추정.
ground_truth.json 의 (camera_optical_world_pose, robot_base_world_pose) 와 비교해
translation_error_mm + rotation_error_deg sanity check.

Run:
    source /opt/ros/jazzy/setup.bash
    source ~/sketch_robot_ws/install/setup.bash
    python3 ~/sketch_robot_ws/src/sketch_control/sketch_control/calibration_handeye.py

Prerequisites:
    1. Isaac Sim 실행 + Play (isaac_sim_rb10.py 로 CALIB_POSE 적용 상태)
    2. apriltag_ros detector 실행 (/detections 발행 중)
    3. /zed/zed_node/rgb/color/rect/camera_info, /joint_states, /tf 모두 활성
"""

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, JointState
from tf2_ros import Buffer, TransformListener, TransformException

try:
    from apriltag_msgs.msg import AprilTagDetectionArray
except ImportError:
    print("[FATAL] apriltag_msgs not found — "
          "sudo apt install ros-jazzy-apriltag-msgs ros-jazzy-apriltag-ros",
          file=sys.stderr)
    sys.exit(1)


# --- 파일 위치 ----------------------------------------------------------------
GROUND_TRUTH_PATH = Path.home() / "sketch_robot_ws" / "ground_truth.json"
OUTPUT_PATH = Path.home() / "sketch_robot_ws" / "calibration_results.json"

# --- ROS 토픽 / 프레임 --------------------------------------------------------
JOINT_COMMAND_TOPIC = "/joint_command"
JOINT_STATES_TOPIC = "/joint_states"
DETECTIONS_TOPIC = "/detections"
CAMERA_INFO_TOPIC = "/zed/zed_node/rgb/color/rect/camera_info"
ROBOT_BASE_FRAME = "link0"
ROBOT_TCP_FRAME = "tcp"

# --- Joint 순서 (URDF kinematic chain) ----------------------------------------
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
JOINT_IDX = {n: i for i, n in enumerate(JOINT_NAMES)}

# --- Safe variation 범위 (CALIB_POSE 기준 ±rad) -------------------------------
# 위 / shoulder / elbow 는 충돌 위험 (wall, mount, self) 으로 작게.
# Wrist 3 개는 tag 방향 다양화 위해 크게.
SAFE_DELTA = {
    "base":     0.4,
    "shoulder": 0.3,
    "elbow":    0.3,
    "wrist1":   0.5,
    "wrist2":   0.4,
    "wrist3":   0.5,
}

TARGET_TAG_ID = 0


# --- Calibration 동작 파라미터 ------------------------------------------------
@dataclass
class PoseConfig:
    safe_delta_per_joint: dict = field(default_factory=lambda: dict(SAFE_DELTA))
    num_poses: int = 40                    # 총 pose 수
    min_valid_poses: int = 10              # 이 미만이면 abort
    ready_settle_s: float = 1.0            # CALIB_POSE 경유 settling
    settling_time_s: float = 1.5           # target pose 후 정착 대기
    detection_timeout_s: float = 2.0       # detection 캡처 총 timeout
    samples_per_pose: int = 10             # multi-sample 평균 frame
    high_skip_warn_ratio: float = 0.30     # skip 비율 경고 임계값


# =============================================================================
# 노드
# =============================================================================
class HandEyeCalibrationNode(Node):

    def __init__(self, config: Optional[PoseConfig] = None):
        super().__init__("handeye_calibration")

        self.config = config or PoseConfig()

        # Ground truth -------------------------------------------------------
        if not GROUND_TRUTH_PATH.exists():
            raise FileNotFoundError(f"ground_truth.json not found: {GROUND_TRUTH_PATH}")
        with open(GROUND_TRUTH_PATH) as _f:
            self.gt = json.load(_f)
        self.tag_size = float(self.gt["apriltag"]["size_m"])
        mesh_size = self.gt["apriltag"].get("mesh_size_m")
        if mesh_size is None:
            self.get_logger().info(f"Tag size (from GT): {self.tag_size} m")
        else:
            self.get_logger().info(
                f"Tag detected size (from GT): {self.tag_size} m "
                f"(mesh {float(mesh_size)} m)"
            )
        self.get_logger().info(
            f"Config: num_poses={self.config.num_poses}, "
            f"min_valid={self.config.min_valid_poses}, "
            f"settling={self.config.settling_time_s}s, "
            f"detection_timeout={self.config.detection_timeout_s}s"
        )

        # TF -----------------------------------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # State for callbacks ------------------------------------------------
        self.K = None
        self.D = None
        self.cam_size = None
        self.latest_detection = None
        self.latest_joint_state = None

        # Subscribers --------------------------------------------------------
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self._cam_info_cb, 10)
        self.create_subscription(AprilTagDetectionArray, DETECTIONS_TOPIC,
                                 self._detection_cb, 10)
        self.create_subscription(JointState, JOINT_STATES_TOPIC,
                                 self._joint_state_cb, 10)

        # Publisher (Isaac Sim native ZedROS2Graph 와 무관 — JointGraph 의 SubJC) -
        self.joint_cmd_pub = self.create_publisher(JointState, JOINT_COMMAND_TOPIC, 10)

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _cam_info_cb(self, msg: CameraInfo):
        self.K = np.array(msg.k).reshape(3, 3)
        # ZED sim 은 distortion 없음 — 길이 부족하면 zeros.
        self.D = np.array(msg.d) if len(msg.d) >= 4 else np.zeros(5)
        self.cam_size = (msg.width, msg.height)

    def _detection_cb(self, msg: AprilTagDetectionArray):
        for det in msg.detections:
            if int(det.id) == TARGET_TAG_ID:
                self.latest_detection = det
                return

    def _joint_state_cb(self, msg: JointState):
        d = dict(zip(msg.name, msg.position))
        if all(j in d for j in JOINT_NAMES):
            self.latest_joint_state = [float(d[j]) for j in JOINT_NAMES]

    # ------------------------------------------------------------------
    # Pose generation (CALIB_POSE 기준 SAFE_DELTA 범위 내 40 pose)
    # ------------------------------------------------------------------
    def _generate_poses(self, base_pose) -> List[List[float]]:
        """3-bucket strategy — 모든 variation 이 SAFE_DELTA 내.

        1) Wrist axis-aligned (12 pose) — tag 방향 다양화에 가장 큰 효과
        2) Base/shoulder/elbow axis-aligned (6 pose) — 카메라 view 안에서
           큰 viewpoint 변화 (translation diversity)
        3) Multi-joint random uniform (남은 22 pose) — 6 joint 동시 variation
        """
        cfg = self.config
        safe = cfg.safe_delta_per_joint
        poses: List[List[float]] = []
        rng = np.random.default_rng(seed=42)

        # 1) Wrist axis-aligned: 각 wrist 마다 [-Δ, -Δ/2, +Δ/2, +Δ] (4 × 3 = 12)
        for jname in ("wrist1", "wrist2", "wrist3"):
            jidx = JOINT_IDX[jname]
            d_max = safe[jname]
            for d in (-d_max, -d_max / 2.0, +d_max / 2.0, +d_max):
                p = list(base_pose)
                p[jidx] += d
                poses.append(p)

        # 2) Base/shoulder/elbow axis-aligned: 각 ±Δ (2 × 3 = 6)
        for jname in ("base", "shoulder", "elbow"):
            jidx = JOINT_IDX[jname]
            d_max = safe[jname]
            for d in (-d_max, +d_max):
                p = list(base_pose)
                p[jidx] += d
                poses.append(p)

        # 3) Multi-joint random uniform within SAFE_DELTA (나머지)
        n_random = max(0, cfg.num_poses - len(poses))
        for _ in range(n_random):
            p = list(base_pose)
            for jname, jidx in JOINT_IDX.items():
                d_max = safe[jname]
                p[jidx] += float(rng.uniform(-d_max, +d_max))
            poses.append(p)

        return poses[:cfg.num_poses]

    # ------------------------------------------------------------------
    # Capture helpers
    # ------------------------------------------------------------------
    def _spin_for(self, sec):
        end = time.time() + sec
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _publish_joints(self, joint_positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = [float(p) for p in joint_positions]
        self.joint_cmd_pub.publish(msg)

    def _move_and_capture(self, target_pose, ready_pose):
        """Ready → settle → target → settle → capture.

        Pose 간 ready (= CALIB_POSE) 경유 — wall/mount 충돌 회피 + 매 pose 도착이
        같은 starting state (detection 일관성). 반환: (corners_avg, None) on success
        or (None, "reason") on fail.
        """
        cfg = self.config
        # 1) Ready pose 경유
        self._publish_joints(ready_pose)
        self._spin_for(cfg.ready_settle_s)
        # 2) Target pose
        self._publish_joints(target_pose)
        self._spin_for(cfg.settling_time_s)
        # 3) Capture
        corners_avg = self._capture_corners_avg(
            cfg.samples_per_pose, total_timeout=cfg.detection_timeout_s
        )
        if corners_avg is None:
            return None, "no detection within timeout (시야 밖 / 충돌 / 가림)"
        return corners_avg, None

    def _capture_corners_avg(self, n, total_timeout):
        """total_timeout 초 안에 최대 n 개 fresh detection 캡처 → (4,2) px 평균.

        실패 (>= n/2 개 미만) 시 None. tag 가 시야 밖이거나 detector 가 인식 X 면
        timeout 안에 충분히 못 모음 → skip 신호."""
        corners_list = []
        deadline = time.time() + total_timeout
        while rclpy.ok() and len(corners_list) < n and time.time() < deadline:
            self.latest_detection = None
            # 남은 시간 안에서 다음 detection 까지 wait
            while (rclpy.ok() and self.latest_detection is None
                   and time.time() < deadline):
                rclpy.spin_once(self, timeout_sec=0.02)
            if self.latest_detection is None:
                break
            det = self.latest_detection
            corners = np.array([[c.x, c.y] for c in det.corners], dtype=np.float64)
            corners_list.append(corners)
        if len(corners_list) < max(1, n // 2):
            return None
        return np.mean(np.stack(corners_list, axis=0), axis=0)

    def _lookup_base_tcp(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                ROBOT_BASE_FRAME, ROBOT_TCP_FRAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
        except TransformException as e:
            self.get_logger().warn(f"TF {ROBOT_BASE_FRAME}->{ROBOT_TCP_FRAME} fail: {e}")
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        R = _quat_xyzw_to_R([q.x, q.y, q.z, q.w])
        return R, np.array([t.x, t.y, t.z])

    # ------------------------------------------------------------------
    # PnP — tag corners (px) → R, t in camera frame
    # ------------------------------------------------------------------
    def _solve_pnp(self, corners_px):
        half = self.tag_size / 2.0
        # apriltag_ros 3.3.0 corners 순서 = TR / TL / BL / BR (pose 0 캡처로 확인).
        # Tag frame: +X right, +Y up, +Z out of tag.
        obj_pts = np.array([
            [ half,  half, 0.0],   # corners[0] = top-right
            [-half,  half, 0.0],   # corners[1] = top-left
            [-half, -half, 0.0],   # corners[2] = bottom-left
            [ half, -half, 0.0],   # corners[3] = bottom-right
        ], dtype=np.float64)
        # SOLVEPNP_ITERATIVE — IPPE_SQUARE 와 달리 corner 순서에 strict 하지 X.
        # IPPE_SQUARE 는 "BL 시작 CCW" 강제 → apriltag_ros 의 "TR 시작 CCW" 와
        # 불일치 시 z magnitude 0.028m 같은 garbage 출력. ITERATIVE 는 obj/img
        # 대응만 일관되면 OK.
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, corners_px.astype(np.float64),
            self.K, self.D,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        # apriltag_ros corner order + flipV texture mapping makes the detected tag
        # frame differ from the USD mesh physical frame by local Rz(180deg).
        # Convert PnP output to the physical tag frame stored in ground_truth.json.
        R = R @ np.diag([-1.0, -1.0, 1.0])
        return R, tvec.flatten()

    # ------------------------------------------------------------------
    # Main flow
    # ------------------------------------------------------------------
    def run(self):
        # Wait for topics + TF
        self.get_logger().info(
            "Waiting for camera_info / joint_states / detections / TF ..."
        )
        deadline = time.time() + 20.0
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            tf_ok = self.tf_buffer.can_transform(
                ROBOT_BASE_FRAME, ROBOT_TCP_FRAME, rclpy.time.Time()
            )
            if (self.K is not None
                    and self.latest_joint_state is not None
                    and tf_ok):
                break
        missing = []
        if self.K is None:
            missing.append(f"CameraInfo({CAMERA_INFO_TOPIC})")
        if self.latest_joint_state is None:
            missing.append(f"JointState({JOINT_STATES_TOPIC})")
        if not self.tf_buffer.can_transform(ROBOT_BASE_FRAME, ROBOT_TCP_FRAME,
                                            rclpy.time.Time()):
            missing.append(f"TF({ROBOT_BASE_FRAME}->{ROBOT_TCP_FRAME})")
        if missing:
            self.get_logger().error("Missing topics/TF: " + ", ".join(missing))
            return False

        base_pose = list(self.latest_joint_state)
        self.get_logger().info(
            f"Base pose (현재 /joint_states): "
            f"[{', '.join(f'{v:.4f}' for v in base_pose)}]"
        )

        poses = self._generate_poses(base_pose)
        self.get_logger().info(f"Generated {len(poses)} calibration poses")

        # ---- Collect ------------------------------------------------------
        # Ready pose = 시작 시점 /joint_states 의 값 (= isaac_sim_rb10.py 의
        # CALIB_POSE). 각 target 가기 전에 ready 경유 → 안전 + detection 일관성.
        cfg = self.config
        ready_pose = list(base_pose)
        per_pose_sec = (cfg.ready_settle_s + cfg.settling_time_s
                        + cfg.detection_timeout_s)
        self.get_logger().info(
            f"Per-pose budget ≈ {per_pose_sec:.1f}s (ready {cfg.ready_settle_s}s + "
            f"target {cfg.settling_time_s}s + capture {cfg.detection_timeout_s}s). "
            f"Total ~{per_pose_sec * len(poses):.0f}s for {len(poses)} poses."
        )
        samples = []
        skipped = []  # list[(pose_idx, reason)] — abort report 용
        for i, joint_target in enumerate(poses):
            self.get_logger().info(
                f"[{i+1:02d}/{len(poses)}] ready → target → "
                f"[{', '.join(f'{v:+.3f}' for v in joint_target)}]"
            )
            corners_avg, fail_reason = self._move_and_capture(joint_target, ready_pose)
            if corners_avg is None:
                skipped.append((i, fail_reason))
                self.get_logger().warn(f"    skip — {fail_reason}")
                continue

            pnp = self._solve_pnp(corners_avg)
            tcp = self._lookup_base_tcp()
            if pnp is None or tcp is None:
                reason = "PnP fail" if pnp is None else "TF lookup fail"
                skipped.append((i, reason))
                self.get_logger().warn(f"    skip — {reason}")
                continue
            R_cam_tag, t_cam_tag = pnp
            R_base_tcp, t_base_tcp = tcp
            samples.append({
                "joints": joint_target,
                "R_base_tcp": R_base_tcp.tolist(),
                "t_base_tcp": t_base_tcp.tolist(),
                "R_cam_tag": R_cam_tag.tolist(),
                "t_cam_tag": t_cam_tag.tolist(),
                "corners_avg_px": corners_avg.tolist(),
            })
            self.get_logger().info(
                f"    ✓ cam→tag t = "
                f"[{t_cam_tag[0]:+.3f}, {t_cam_tag[1]:+.3f}, {t_cam_tag[2]:+.3f}] m "
                f"(|t|={float(np.linalg.norm(t_cam_tag)):.3f}m)"
            )

        # ---- Skip rate / abort 검사 ---------------------------------------
        skip_rate = (len(skipped) / len(poses)) if poses else 1.0
        self.get_logger().info(
            f"Captured {len(samples)}/{len(poses)} (skipped {len(skipped)}, "
            f"rate {skip_rate:.0%})"
        )
        if len(samples) < cfg.min_valid_poses:
            self.get_logger().error(
                f"Only {len(samples)} valid poses — need ≥{cfg.min_valid_poses}. "
                f"Aborting calibration."
            )
            print()
            print(f"Skipped poses ({len(skipped)}):")
            for idx, reason in skipped:
                print(f"  #{idx+1:02d}: {reason}")
            return False
        if skip_rate > cfg.high_skip_warn_ratio:
            self.get_logger().warn(
                f"High skip rate {skip_rate:.0%} > {cfg.high_skip_warn_ratio:.0%} — "
                f"SAFE_DELTA 줄이거나 CALIB_POSE 재설정 권장"
            )

        # ---- Calibration (known tcp->tag absolute solve) ------------------
        # 이 셋업은 고정 카메라 + TCP 에 부착된 AprilTag 이고, tcp->tag offset 을
        # ground_truth/실측으로 알고 있다. 따라서 각 pose 에서 바로
        #   T_cam_base = T_cam_tag * inv(T_tcp_tag) * inv(T_base_tcp)
        # 를 계산한 뒤 평균내는 편이 handEye workaround 보다 모호성이 적다.
        T_tcp_tag = _pose_dict_to_T(self.gt["apriltag"]["tcp_local_pose"])
        T_cam_base_samples = []
        for s in samples:
            R_b2t = np.array(s["R_base_tcp"])
            t_b2t = np.array(s["t_base_tcp"])
            T_base_tcp = np.eye(4)
            T_base_tcp[:3, :3] = R_b2t
            T_base_tcp[:3, 3] = t_b2t

            T_cam_tag = np.eye(4)
            T_cam_tag[:3, :3] = np.array(s["R_cam_tag"])
            T_cam_tag[:3, 3] = np.array(s["t_cam_tag"])

            T_cam_base_samples.append(
                T_cam_tag @ np.linalg.inv(T_tcp_tag) @ np.linalg.inv(T_base_tcp)
            )

        T_cam_base = _average_transforms(T_cam_base_samples)
        R_cam2base = T_cam_base[:3, :3]
        t_cam2base = T_cam_base[:3, 3]

        # ---- Sanity check (ground truth) ----------------------------------
        T_world_cam = _pose_dict_to_T(self.gt["camera_optical_world_pose"])
        gt_base = self.gt.get("robot_base_world_pose")
        if gt_base is None:
            T_world_base = np.eye(4)
            self.get_logger().info("GT robot_base_world_pose=null → identity 사용")
        else:
            T_world_base = _pose_dict_to_T(gt_base)
        T_cam2base_gt = np.linalg.inv(T_world_cam) @ T_world_base

        T_cam2base_est = np.eye(4)
        T_cam2base_est[:3, :3] = R_cam2base
        T_cam2base_est[:3, 3] = t_cam2base

        t_err_mm = float(np.linalg.norm(
            T_cam2base_est[:3, 3] - T_cam2base_gt[:3, 3]
        )) * 1000.0
        R_diff = T_cam2base_est[:3, :3] @ T_cam2base_gt[:3, :3].T
        trace = float(np.clip(np.trace(R_diff), -1.0, 3.0))
        R_err_deg = float(np.degrees(
            np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        ))

        # ---- Console summary ----------------------------------------------
        q_est = _R_to_quat_xyzw(R_cam2base)
        q_gt = _R_to_quat_xyzw(T_cam2base_gt[:3, :3])
        print()
        print("=" * 60)
        print("Hand-eye Calibration Result")
        print("=" * 60)
        print(f"Num poses captured: {len(samples)} / {len(poses)}")
        print(f"Tag size:           {self.tag_size} m")
        print(f"Method:             DIRECT_KNOWN_TCP_TAG_AVERAGE")
        print()
        print(f"Estimated T_cam→base:")
        print(f"  translation [m]:   "
              f"[{t_cam2base[0]:+.4f}, {t_cam2base[1]:+.4f}, {t_cam2base[2]:+.4f}]")
        print(f"  rotation (xyzw):   "
              f"[{q_est[0]:+.4f}, {q_est[1]:+.4f}, {q_est[2]:+.4f}, {q_est[3]:+.4f}]")
        print()
        gt_t = T_cam2base_gt[:3, 3]
        print(f"Ground truth T_cam→base:")
        print(f"  translation [m]:   "
              f"[{gt_t[0]:+.4f}, {gt_t[1]:+.4f}, {gt_t[2]:+.4f}]")
        print(f"  rotation (xyzw):   "
              f"[{q_gt[0]:+.4f}, {q_gt[1]:+.4f}, {q_gt[2]:+.4f}, {q_gt[3]:+.4f}]")
        print()
        print(f"Calibration accuracy:")
        print(f"  Translation error: {t_err_mm:.3f} mm")
        print(f"  Rotation error:    {R_err_deg:.4f} deg")
        if t_err_mm < 1.0:
            print("✅ Calibration 정확도 OK (< 1mm) — sketch robot mm 정밀도 충족")
        elif t_err_mm < 5.0:
            print("⚠️  borderline (1-5mm) — pose 다양화 / multi-sample 증가 권장")
        else:
            print("❌ 정확도 부족 (> 5mm) — setup 점검 (intrinsic / tag size / corners 순서)")
        print("=" * 60)

        # ---- Save JSON ----------------------------------------------------
        out = {
            "method": "DIRECT_KNOWN_TCP_TAG_AVERAGE",
            "num_poses": len(samples),
            "num_attempted": len(poses),
            "num_skipped": len(skipped),
            "skip_rate": skip_rate,
            "tag_size_m": self.tag_size,
            "config": {
                "num_poses": cfg.num_poses,
                "min_valid_poses": cfg.min_valid_poses,
                "ready_settle_s": cfg.ready_settle_s,
                "settling_time_s": cfg.settling_time_s,
                "detection_timeout_s": cfg.detection_timeout_s,
                "samples_per_pose": cfg.samples_per_pose,
                "safe_delta_per_joint": cfg.safe_delta_per_joint,
            },
            "T_cam_to_base": {
                "translation": t_cam2base.tolist(),
                "rotation_xyzw": list(q_est),
            },
            "ground_truth_T_cam_to_base": {
                "translation": T_cam2base_gt[:3, 3].tolist(),
                "rotation_xyzw": list(q_gt),
            },
            "accuracy": {
                "translation_error_mm": t_err_mm,
                "rotation_error_deg": R_err_deg,
            },
            "skipped_poses": [
                {"pose_index": idx, "reason": reason} for idx, reason in skipped
            ],
            "poses_used": samples,
        }
        with open(OUTPUT_PATH, "w") as _f:
            json.dump(out, _f, indent=2)
        self.get_logger().info(f"Results saved → {OUTPUT_PATH}")

        # ---- Static TF launch arg suggestion ------------------------------
        print()
        print("[참고용] calibration 결과를 TF tree 에 영구 등록:")
        print(f"ros2 run tf2_ros static_transform_publisher \\")
        print(f"  --x {t_cam2base[0]:.6f} --y {t_cam2base[1]:.6f} --z {t_cam2base[2]:.6f} \\")
        print(f"  --qx {q_est[0]:.6f} --qy {q_est[1]:.6f} --qz {q_est[2]:.6f} --qw {q_est[3]:.6f} \\")
        print(f"  --frame-id zed_left_camera_frame_optical \\")
        print(f"  --child-frame-id {ROBOT_BASE_FRAME}")

        return True


# =============================================================================
# 수학 유틸
# =============================================================================
def _quat_xyzw_to_R(q):
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _R_to_quat_xyzw(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def _pose_dict_to_T(pose):
    T = np.eye(4)
    T[:3, :3] = _quat_xyzw_to_R(pose["rotation_xyzw"])
    T[:3, 3] = pose["translation"]
    return T


def _average_transforms(transforms):
    """Average SE(3) samples with arithmetic mean translation + SVD rotation."""
    if not transforms:
        return np.eye(4)
    T_avg = np.eye(4)
    T_avg[:3, 3] = np.mean([T[:3, 3] for T in transforms], axis=0)
    R_sum = np.sum([T[:3, :3] for T in transforms], axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1.0
        R_avg = U @ Vt
    T_avg[:3, :3] = R_avg
    return T_avg


# =============================================================================
# Entry
# =============================================================================
def main():
    rclpy.init()
    node = HandEyeCalibrationNode()
    rc = 1
    try:
        ok = node.run()
        rc = 0 if ok else 2
    except KeyboardInterrupt:
        node.get_logger().info("interrupted")
    except Exception as e:
        node.get_logger().error(f"unhandled: {e}")
        import traceback
        traceback.print_exc()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()
