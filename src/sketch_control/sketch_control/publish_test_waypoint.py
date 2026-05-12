#!/usr/bin/env python3
"""
Phase 4 Session 4 — Stage 1 단독 검증용 테스트 waypoint publisher.

용도:
- 로봇이 READY_POSE 인 상태에서 실행.
- tf2 로 link0 → tcp 실측해 orientation 추출 (RB10 base/EE frame).
- 단일 Pose 의 PoseArray 를 /sketch_waypoints 에 publish.
- 그 후 별도로 /debug_trigger_stage1 publish 하면 Stage 1 단독 실행됨.

사용:
    ros2 run sketch_control publish_test_waypoint
"""

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray
from tf2_ros import Buffer, TransformListener


# RB10 frame 이름 (moveit_executor.py 의 BASE_FRAME / EE_LINK 와 일치).
BASE_FRAME = 'link0'
EE_FRAME = 'tcp'

# 보드 표면 위 한 점 (link0 기준).
# 보드 가시영역: y ∈ [-0.60, +0.30], z ∈ [0, 0.90], 평면 x = 0.80.
TARGET_X = 0.80
TARGET_Y = 0.0
TARGET_Z = 0.45


class TestWaypointPublisher(Node):
    def __init__(self):
        super().__init__('test_waypoint_publisher')
        self.pub = self.create_publisher(PoseArray, '/sketch_waypoints', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def lookup_ee_orientation(self, timeout_sec: float = 3.0):
        """BASE_FRAME → EE_FRAME transform 의 rotation 부분을 quaternion 으로 반환."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                trans = self.tf_buffer.lookup_transform(
                    BASE_FRAME, EE_FRAME, rclpy.time.Time())
                return trans.transform.rotation
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.1)
        return None

    def wait_for_subscriber(self, timeout_sec: float = 3.0):
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self.pub.get_subscription_count() > 0:
                return True
            self.get_logger().info('Waiting for /sketch_waypoints subscriber...')
            rclpy.spin_once(self, timeout_sec=0.2)
        return False

    def publish_test_waypoint(self):
        rot = self.lookup_ee_orientation()
        if rot is None:
            self.get_logger().error(
                f'{BASE_FRAME} -> {EE_FRAME} transform 못 받음. '
                'robot_state_publisher / joint_states 활성 확인.')
            return False

        pose = Pose()
        pose.position.x = TARGET_X
        pose.position.y = TARGET_Y
        pose.position.z = TARGET_Z
        pose.orientation = rot  # READY 시점의 EE orientation 그대로

        msg = PoseArray()
        msg.header.frame_id = BASE_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.poses = [pose]

        if not self.wait_for_subscriber():
            self.get_logger().warn(
                'subscriber 미발견. moveit_executor 활성 확인. 그래도 publish 시도.')

        for _ in range(5):
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().info(
            f'[OK] published /sketch_waypoints  '
            f'pos=({TARGET_X:.3f}, {TARGET_Y:.3f}, {TARGET_Z:.3f})  '
            f'quat=({rot.x:.4f}, {rot.y:.4f}, {rot.z:.4f}, {rot.w:.4f})')
        return True


def main():
    rclpy.init()
    node = TestWaypointPublisher()
    try:
        node.publish_test_waypoint()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
