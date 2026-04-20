"""용접 비드 시각화 - torch_tip TF 를 구독해서 RViz Marker 로 비드 표시.

Phase 1.6: 토치 TCP 가 벽에 접촉 범위 안에 들어오면 비드 Marker 생성.
새 비드는 주황 (열) → 3초에 걸쳐 회색 (냉각) 으로 색상 전이.
Isaac Sim 은 전혀 건드리지 않음 (Marker 는 RViz 에만 표시).
"""
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from sketch_control.targets import (
    load_objects_config, get_surface_plane, get_target,
)


CONTACT_RANGE = 0.015       # ±15mm (접촉 판정)
MIN_GAP = 0.003             # 3mm (비드 간 최소 간격)
MAX_GAP = 0.05              # 5cm (이 이상이면 경로 끊김으로 간주)
BEAD_RADIUS = 0.004         # 4mm 구
HOT_COLOR = (1.0, 0.5, 0.0)     # 주황
COOL_COLOR = (0.3, 0.3, 0.3)    # 회색
COOL_DURATION = 3.0         # 초


class WeldVisualizer(Node):
    def __init__(self):
        super().__init__("weld_visualizer")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_pub = self.create_publisher(
            MarkerArray, "/weld_beads", 10)

        # active target 의 표면 평면 로드
        cfg = load_objects_config()
        target_name = cfg.get("active_target", "wall")
        target = get_target(cfg, target_name)
        self.plane_pt, self.plane_n = get_surface_plane(target)
        self.plane_pt = np.asarray(self.plane_pt, dtype=float)
        self.plane_n = np.asarray(self.plane_n, dtype=float)

        # 비드 저장
        self.beads = []          # [{pos, created, id}, ...]
        self.last_contact = None
        self.next_id = 0

        # source frame 자동 탐지 (world/World/base_link)
        self._source_frame = None
        self._frame_candidates = ["world", "World", "base_link"]

        self.create_timer(0.05, self.tick)            # 20Hz 추적
        self.create_timer(0.2, self.publish_markers)  # 5Hz 색 업데이트

        self.get_logger().info(
            f"Weld Visualizer 시작 | target={target_name} "
            f"plane_pt={self.plane_pt} normal={self.plane_n}"
        )

    def _detect_source_frame(self):
        """후보 프레임 중 torch_tip 으로 TF 가능한 것 선택. 성공 시 True."""
        for f in self._frame_candidates:
            try:
                self.tf_buffer.lookup_transform(
                    f, "torch_tip", rclpy.time.Time(),
                    timeout=Duration(seconds=0.1))
                self._source_frame = f
                self.get_logger().info(f"[OK] TF source frame 자동 탐지: '{f}'")
                return True
            except Exception:
                continue
        return False

    def tick(self):
        if self._source_frame is None:
            if not self._detect_source_frame():
                # 2초에 한 번만 안내 로그
                self.get_logger().warn(
                    "TF source frame 못 찾음 "
                    f"(후보: {self._frame_candidates}). Isaac Sim /tf 확인",
                    throttle_duration_sec=2.0,
                )
                return

        try:
            tf = self.tf_buffer.lookup_transform(
                self._source_frame, "torch_tip", rclpy.time.Time(),
                timeout=Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup 실패: {self._source_frame} -> torch_tip ({e})",
                throttle_duration_sec=2.0,
            )
            return

        tip = np.array([
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ])

        # 평면까지 부호 있는 거리
        signed = float(np.dot(tip - self.plane_pt, self.plane_n))

        # 1초에 한 번씩 상태 로그
        self.get_logger().info(
            f"[DBG] tip=({tip[0]:+.3f},{tip[1]:+.3f},{tip[2]:+.3f}) "
            f"signed_dist={signed*1000:+.1f}mm "
            f"(range=±{CONTACT_RANGE*1000:.0f}mm) beads={len(self.beads)}",
            throttle_duration_sec=1.0,
        )

        if not (-CONTACT_RANGE <= signed <= CONTACT_RANGE):
            self.last_contact = None
            return

        # 평면 위로 투영
        projected = tip - self.plane_n * signed

        last = self.last_contact
        if last is None:
            self.last_contact = projected
            self._add_bead(projected)
            self.get_logger().info(
                f"[BEAD] 첫 접촉: ({projected[0]:.3f},"
                f"{projected[1]:.3f},{projected[2]:.3f})"
            )
            return

        gap = float(np.linalg.norm(projected - last))
        if gap < MIN_GAP:
            return
        if gap > MAX_GAP:
            self.last_contact = projected
            return

        self._add_bead(projected)
        self.last_contact = projected

    def _add_bead(self, pos):
        self.beads.append({
            "pos": pos.copy(),
            "created": time.time(),
            "id": self.next_id,
        })
        self.next_id += 1

    def publish_markers(self):
        if not self.beads:
            return

        arr = MarkerArray()
        now = time.time()
        frame = self._source_frame or "world"
        for b in self.beads:
            age = now - b["created"]
            t = min(1.0, age / COOL_DURATION)
            r = HOT_COLOR[0] + (COOL_COLOR[0] - HOT_COLOR[0]) * t
            g = HOT_COLOR[1] + (COOL_COLOR[1] - HOT_COLOR[1]) * t
            bl = HOT_COLOR[2] + (COOL_COLOR[2] - HOT_COLOR[2]) * t

            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "weld_beads"
            m.id = b["id"]
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(b["pos"][0])
            m.pose.position.y = float(b["pos"][1])
            m.pose.position.z = float(b["pos"][2])
            m.pose.orientation.w = 1.0
            m.scale.x = BEAD_RADIUS * 2
            m.scale.y = BEAD_RADIUS * 2
            m.scale.z = BEAD_RADIUS * 2
            m.color.r = float(r)
            m.color.g = float(g)
            m.color.b = float(bl)
            m.color.a = 1.0
            arr.markers.append(m)

        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = WeldVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
