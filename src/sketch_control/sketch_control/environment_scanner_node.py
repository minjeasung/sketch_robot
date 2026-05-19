"""
environment_scanner_node — ZED point cloud 한 번 스캔 → 다중 평면 + 잔여 장애물 voxel grid.

입력
  /zed/zed_node/point_cloud/cloud_registered  (sensor_msgs/PointCloud2)
  /perception/scan_trigger                    (std_msgs/Empty)   — 재스캔 요청

출력 (모두 입력 cloud 의 frame_id 그대로, 보통 "zed_left_camera_frame")
  /perception/planes         (geometry_msgs/PoseArray)
      pose.position    = 평면 inlier centroid
      pose.orientation = +Z 축이 plane normal 을 향하도록 회전한 quaternion
  /perception/plane_labels   (std_msgs/String, JSON)
      {"planes":[{"id":0,"type":"wall","size":[w,h],"n_inliers":N}, ...]}
  /perception/obstacles      (visualization_msgs/MarkerArray)
      평면 제거 후 남은 점들을 voxel grid 화. 각 voxel = CUBE marker.

알고리즘:
  1. trigger (시작 시 1회 자동 + /perception/scan_trigger 수신 시) 직후 도착하는 cloud 1개 캐시
  2. voxel_down_sample(0.02), crop (5m 이내)
  3. 반복 RANSAC (최대 5회): segment_plane(d=0.01, n=3, iter=2000)
     - inliers > 1000 점 인 평면만 인정, 그 외엔 중단
  4. 평면 normal.z 로 분류:
       |nz| < 0.3   → vertical (벽 후보)
       |nz| > 0.7   → horizontal (바닥/테이블 후보)
       그 외        → other
     수직 중 최대 → "wall", 수평 중 가장 낮은 Z(centroid.z 작음) → "floor", 나머지 → "other"
  5. 모든 평면 inliers 제거 후 남은 점 → voxel grid (0.05 m), 각 voxel 위치 → obstacle CUBE marker

규칙:
  ~/sketch_robot_ws/perception/ransac_multiplane.py 의 segment_plane wrapping 패턴을 참고.
"""

import json
from typing import Optional

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Empty, String
from visualization_msgs.msg import Marker, MarkerArray


# ── 입출력 토픽 ─────────────────────────────────────────────
INPUT_CLOUD_TOPIC = "/zed/zed_node/point_cloud/cloud_registered"
TRIGGER_TOPIC = "/perception/scan_trigger"
PLANES_TOPIC = "/perception/planes"
LABELS_TOPIC = "/perception/plane_labels"
OBSTACLES_TOPIC = "/perception/obstacles"

# ── 처리 파라미터 ────────────────────────────────────────────
VOXEL_DOWN = 0.02         # m, 입력 down-sample
CROP_MAX_DIST = 5.0       # m, 이 거리 초과 점은 폐기
RANSAC_DIST = 0.01        # m
RANSAC_N = 3
RANSAC_ITERS = 2000
MAX_PLANES = 5
MIN_PLANE_INLIERS = 1000  # 점 수 < 이 값이면 평면으로 인정 안 함
OBSTACLE_VOXEL = 0.05     # m, 잔여 점들의 voxel grid 크기

# 분류 기준
VERT_NZ_MAX = 0.3
HORIZ_NZ_MIN = 0.7


def _normal_to_quaternion(normal: np.ndarray):
    """+Z = (0,0,1) 을 normal 단위벡터로 회전시키는 quaternion (x,y,z,w)."""
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
    return (float(axis[0] * s), float(axis[1] * s), float(axis[2] * s),
            float(np.cos(angle / 2.0)))


def _plane_size_on_axes(inlier_pts: np.ndarray, normal: np.ndarray):
    """평면 inlier 점들을 평면 내 2개 직교축에 투영해 (w, h) 측정."""
    n = normal / (np.linalg.norm(normal) + 1e-12)
    # n 과 가장 평행하지 않은 기준축으로 직교벡터 잡기
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v /= (np.linalg.norm(v) + 1e-12)
    centered = inlier_pts - inlier_pts.mean(axis=0)
    pu = centered @ u
    pv = centered @ v
    return float(pu.max() - pu.min()), float(pv.max() - pv.min())


class EnvironmentScannerNode(Node):
    def __init__(self):
        super().__init__("environment_scanner_node")

        self._pending_scan = True   # 시작 시 1회 자동 스캔
        self._latest_frame_id = "zed_left_camera_frame"

        self.create_subscription(
            PointCloud2, INPUT_CLOUD_TOPIC, self._on_cloud, qos_profile_sensor_data)
        self.create_subscription(
            Empty, TRIGGER_TOPIC, self._on_trigger, 10)

        self.planes_pub = self.create_publisher(PoseArray, PLANES_TOPIC, 10)
        self.labels_pub = self.create_publisher(String, LABELS_TOPIC, 10)
        self.obstacles_pub = self.create_publisher(MarkerArray, OBSTACLES_TOPIC, 10)

        self.get_logger().info(
            f"environment_scanner 시작 — 자동 1회 스캔 대기 (trigger: {TRIGGER_TOPIC})")

    # ── trigger 처리 ────────────────────────────────────────
    def _on_trigger(self, _msg: Empty):
        self.get_logger().info("scan_trigger 수신 — 다음 cloud 캐시")
        self._pending_scan = True

    # ── cloud 콜백 (pending 시에만 처리) ───────────────────
    def _on_cloud(self, msg: PointCloud2):
        if not self._pending_scan:
            return
        self._pending_scan = False  # one-shot

        pts = self._cloud_to_numpy(msg)
        if pts is None or pts.shape[0] < 100:
            self.get_logger().warn("scan: 점 수 부족 — 재시도 대기")
            self._pending_scan = True
            return

        self._latest_frame_id = msg.header.frame_id or "zed_left_camera_frame"
        self.get_logger().info(
            f"scan 시작: {pts.shape[0]} pts (frame={self._latest_frame_id})")

        # ── voxel down-sample + crop ────────────────────────
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd = pcd.voxel_down_sample(VOXEL_DOWN)
        arr = np.asarray(pcd.points)
        dist = np.linalg.norm(arr, axis=1)
        keep = np.where(dist < CROP_MAX_DIST)[0]
        pcd = pcd.select_by_index(keep)
        if len(pcd.points) < 100:
            self.get_logger().warn("crop 후 점 수 부족 — abort")
            return
        self.get_logger().info(
            f"  preprocess: {len(pcd.points)} pts (voxel={VOXEL_DOWN}, crop<{CROP_MAX_DIST}m)")

        # ── 반복 RANSAC: 최대 MAX_PLANES 개 추출 ───────────
        remaining = pcd
        planes = []  # list of dict {normal, centroid, inlier_pts, n_inliers}
        for i in range(MAX_PLANES):
            if len(remaining.points) < MIN_PLANE_INLIERS:
                break
            try:
                model, inliers = remaining.segment_plane(
                    distance_threshold=RANSAC_DIST,
                    ransac_n=RANSAC_N,
                    num_iterations=RANSAC_ITERS,
                )
            except Exception as e:
                self.get_logger().warn(f"segment_plane 실패: {e}")
                break
            if len(inliers) < MIN_PLANE_INLIERS:
                self.get_logger().info(
                    f"  plane#{i}: inliers {len(inliers)} < {MIN_PLANE_INLIERS} — 종료")
                break

            a, b, c, _d = model
            n = np.array([a, b, c], dtype=float)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-6:
                break
            n /= n_norm

            inlier_pcd = remaining.select_by_index(inliers)
            inlier_pts = np.asarray(inlier_pcd.points)
            centroid = inlier_pts.mean(axis=0)

            planes.append({
                "normal": n,
                "centroid": centroid,
                "inlier_pts": inlier_pts,
                "n_inliers": int(len(inliers)),
            })
            self.get_logger().info(
                f"  plane#{i}: n=({n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f}) "
                f"inliers={len(inliers)} centroid=({centroid[0]:+.2f},"
                f"{centroid[1]:+.2f},{centroid[2]:+.2f})")

            remaining = remaining.select_by_index(inliers, invert=True)

        # ── 평면 분류 ───────────────────────────────────────
        labels = self._classify(planes)

        # ── 잔여 점 → voxel grid ────────────────────────────
        residual = np.asarray(remaining.points)
        voxel_centers = self._voxelize(residual, OBSTACLE_VOXEL)
        self.get_logger().info(
            f"  residual={residual.shape[0]} pts → obstacle voxels={voxel_centers.shape[0]}")

        # ── publish ──────────────────────────────────────────
        self._publish_planes(planes, labels, msg.header.stamp)
        self._publish_obstacles(voxel_centers, msg.header.stamp)

        self.get_logger().info(
            f"scan 완료: planes={len(planes)} (labels={[l['type'] for l in labels]}) "
            f"obstacles={voxel_centers.shape[0]}")

    # ── PointCloud2 → Nx3 numpy ─────────────────────────────
    def _cloud_to_numpy(self, msg: PointCloud2) -> Optional[np.ndarray]:
        try:
            raw = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            try:
                points = np.column_stack(
                    [raw["x"], raw["y"], raw["z"]]).astype(np.float32)
            except (TypeError, IndexError, ValueError):
                points = np.array(
                    [(p[0], p[1], p[2]) for p in raw], dtype=np.float32)
            points = points[np.isfinite(points).all(axis=1)]
            return points
        except Exception as e:
            self.get_logger().warn(f"PointCloud2 parse 실패: {e}")
            return None

    # ── 평면 라벨링 ─────────────────────────────────────────
    def _classify(self, planes):
        """
        normal.z 기반 분류 + 'wall' / 'floor' 1개씩 선정.
        labels[i] = {"id":i, "type":"wall"|"floor"|"other", "size":[w,h], "n_inliers":N}
        """
        labels = []
        verts, horizs = [], []
        for i, p in enumerate(planes):
            nz = abs(float(p["normal"][2]))
            if nz < VERT_NZ_MAX:
                kind = "vertical"
                verts.append((i, p))
            elif nz > HORIZ_NZ_MIN:
                kind = "horizontal"
                horizs.append((i, p))
            else:
                kind = "other"
            w, h = _plane_size_on_axes(p["inlier_pts"], p["normal"])
            labels.append({"id": i, "type": "other",
                           "size": [round(w, 3), round(h, 3)],
                           "n_inliers": p["n_inliers"], "_kind": kind})

        # wall = 수직 중 최대 inlier
        if verts:
            best_id = max(verts, key=lambda x: x[1]["n_inliers"])[0]
            labels[best_id]["type"] = "wall"
        # floor = 수평 중 centroid.z 가장 낮은 것
        if horizs:
            best_id = min(horizs, key=lambda x: float(x[1]["centroid"][2]))[0]
            labels[best_id]["type"] = "floor"
        # 내부 키 정리
        for lbl in labels:
            lbl.pop("_kind", None)
        return labels

    # ── 잔여 점 voxel grid ─────────────────────────────────
    def _voxelize(self, pts: np.ndarray, voxel: float) -> np.ndarray:
        if pts.shape[0] == 0:
            return np.empty((0, 3), dtype=float)
        idx = np.floor(pts / voxel).astype(np.int64)
        # 고유 voxel 인덱스 추출 후 중심 좌표로 환산
        uniq = np.unique(idx, axis=0)
        return (uniq.astype(float) + 0.5) * voxel

    # ── 발행: planes (PoseArray) + labels (String) ─────────
    def _publish_planes(self, planes, labels, stamp):
        pa = PoseArray()
        pa.header.stamp = stamp
        pa.header.frame_id = self._latest_frame_id
        for p in planes:
            pose = Pose()
            c = p["centroid"]
            pose.position.x = float(c[0])
            pose.position.y = float(c[1])
            pose.position.z = float(c[2])
            qx, qy, qz, qw = _normal_to_quaternion(p["normal"])
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
            pa.poses.append(pose)
        self.planes_pub.publish(pa)

        msg = String()
        msg.data = json.dumps({"planes": labels})
        self.labels_pub.publish(msg)

    # ── 발행: obstacles (MarkerArray) ──────────────────────
    def _publish_obstacles(self, voxel_centers: np.ndarray, stamp):
        ma = MarkerArray()
        # 이전 마커 클리어
        clear = Marker()
        clear.header.frame_id = self._latest_frame_id
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        for i, c in enumerate(voxel_centers):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self._latest_frame_id
            m.ns = "obstacle_voxels"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(c[0])
            m.pose.position.y = float(c[1])
            m.pose.position.z = float(c[2])
            m.pose.orientation.w = 1.0
            m.scale.x = OBSTACLE_VOXEL
            m.scale.y = OBSTACLE_VOXEL
            m.scale.z = OBSTACLE_VOXEL
            m.color.r = 0.9
            m.color.g = 0.3
            m.color.b = 0.1
            m.color.a = 0.6
            ma.markers.append(m)

        self.obstacles_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = EnvironmentScannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
