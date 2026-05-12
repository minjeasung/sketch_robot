# Phase 4 Session 4 — Stage 1 단독 트리거 추가

## 목적

Stage 1 (현재 → safe_pose, OMPL/PTP 자유공간 이동) 을 실로봇에서 단독 실행/검증할 수 있도록 디버그 트리거 토픽을 추가한다. Session 3 에 추가된 `/debug_trigger_stage5` 와 동일한 패턴.

토치 미장착 + 작업면 미접촉 조건이라 보수적인 offset (0.15 m) 을 **별도 상수**로 두고, production 의 stage1 offset 은 건드리지 않는다.

테스트 waypoint 의 EE orientation 은 README 시점의 `base_link → tool0` 실측 quaternion 을 그대로 사용 → IK 도달 가능성 보장 + RB10 의 wall-facing quat 계산 의존성 회피.

---

## 변경 사항 요약

| 파일 | 변경 |
|---|---|
| `src/sketch_control/sketch_control/moveit_executor.py` | 상수 1개 + subscriber 1개 + 콜백 1개 추가 |
| `src/sketch_control/sketch_control/publish_test_waypoint.py` | **신규 작성** |
| `src/sketch_control/setup.py` | `console_scripts` 에 entry_point 1개 추가 |

---

## 1. `moveit_executor.py` 변경

### 1.1 상수 추가

기존 상수 영역 (예: `READY_POSE_JOINTS` 정의 부근) 에 다음 추가:

```python
# Stage 1 단독 디버그용. production 의 approach offset 과 분리.
# 토치 미장착 + 첫 실로봇 검증이라 일반보다 보수적으로 잡음.
DEBUG_STAGE1_OFFSET = 0.15  # meters, surface normal 역방향으로 후퇴 거리

# RB10 setup: 벽이 base_link 의 +X 방향, 평면 x = 0.80
# → surface normal (벽 → 자유공간 방향) = (-1, 0, 0)
DEBUG_STAGE1_SURFACE_NORMAL = (-1.0, 0.0, 0.0)
```

### 1.2 Subscriber 추가

`__init__` 안, **현재 ~133번째 줄의 `/debug_trigger_stage5` subscriber 바로 아래** 에 동일 형식으로 추가:

```python
self.create_subscription(
    Bool, "/debug_trigger_stage1", self.on_debug_trigger_stage1, 10)
```

(Bool import 는 Stage 5 가 이미 쓰고 있으므로 추가 import 불필요.)

### 1.3 콜백 추가

`on_debug_trigger_stage5` 메서드 (현재 ~302번째 줄) **바로 위** 에 다음 메서드 추가. Stage 5 콜백의 구조와 사용 식별자를 그대로 따라가야 한다 (특히 "이미 실행 중" 플래그 이름과 trajectory 실행 함수).

```python
def on_debug_trigger_stage1(self, msg: Bool):
    """
    Stage 1 단독 실행 (디버그용).

    self.last_waypoints 의 첫 번째 Pose 를 surface waypoint 로 보고,
    surface normal 방향으로 DEBUG_STAGE1_OFFSET 만큼 떨어진 safe_pose 계산.
    현재 joint state → safe_pose 를 Stage 1 planner 로 plan, 성공 시 실행.
    Stage 2~5 는 트리거하지 않는다.

    publish_test_waypoint.py 로 /sketch_waypoints 먼저 publish 한 후 실행할 것.
    """
    # ★ "이미 실행 중" 플래그 이름은 on_debug_trigger_stage5 가 사용하는 것과 동일하게.
    if self._is_executing:  # ← Stage 5 콜백이 쓰는 플래그명으로 교체
        self.get_logger().warn(
            "이미 실행 중 -> /debug_trigger_stage1 무시")
        return

    self.get_logger().info(
        "[DEBUG] /debug_trigger_stage1 수신 -> Stage 1 단독 실행")

    # waypoint 존재 확인
    if not self.last_waypoints or len(self.last_waypoints) == 0:
        self.get_logger().error(
            "/sketch_waypoints 가 비어있음. "
            "publish_test_waypoint.py 먼저 실행 필요.")
        return

    waypoint = self.last_waypoints[0]

    # safe_pose = waypoint + DEBUG_STAGE1_OFFSET * surface_normal
    nx, ny, nz = DEBUG_STAGE1_SURFACE_NORMAL
    safe_pose = Pose()
    safe_pose.position.x = waypoint.position.x + DEBUG_STAGE1_OFFSET * nx
    safe_pose.position.y = waypoint.position.y + DEBUG_STAGE1_OFFSET * ny
    safe_pose.position.z = waypoint.position.z + DEBUG_STAGE1_OFFSET * nz
    safe_pose.orientation = waypoint.orientation  # waypoint 의 EE orientation 유지

    self.get_logger().info(
        f"[DEBUG] Stage 1 target safe_pose: "
        f"({safe_pose.position.x:.3f}, {safe_pose.position.y:.3f}, {safe_pose.position.z:.3f})"
    )
    self.get_logger().info(
        f"[DEBUG] safe_pose orientation (xyzw): "
        f"({safe_pose.orientation.x:.4f}, {safe_pose.orientation.y:.4f}, "
        f"{safe_pose.orientation.z:.4f}, {safe_pose.orientation.w:.4f})"
    )

    # 기존 Stage 1 planner 호출.
    # ★ 함수명은 코드 내 실제 메서드로 교체. 후보:
    #    self._plan_stage1, self.plan_stage1, self._plan_approach 등.
    #    on_sketch_execute (또는 동급 콜백) 에서 Stage 1 호출하는 줄 참고.
    try:
        traj = self._plan_stage1(safe_pose)  # ← 실제 함수명으로 교체
    except Exception as e:
        self.get_logger().error(f"Stage 1 planning 실패: {e}")
        return

    if traj is None:
        self.get_logger().error("Stage 1 planning 결과 없음 (None 반환)")
        return

    # 실행
    self._is_executing = True  # ← Stage 5 와 같은 플래그명
    try:
        # ★ Stage 5 콜백이 쓰는 trajectory 실행 함수와 동일한 함수 호출.
        #    후보: self._execute_trajectory, self._execute, self.send_trajectory 등.
        self._execute_trajectory(traj)
        self.get_logger().info("[DEBUG] Stage 1 단독 실행 완료")
    except Exception as e:
        self.get_logger().error(f"Stage 1 실행 실패: {e}")
    finally:
        self._is_executing = False
```

**Claude Code 에 부탁드리는 식별자 매칭** (`★` 표시 4곳):
1. 실행 중 플래그 (`self._is_executing` 자리) — `on_debug_trigger_stage5` 가 사용하는 것과 정확히 동일하게.
2. Stage 1 planner 함수명 (`self._plan_stage1` 자리) — `on_sketch_execute` 또는 동급의 production 콜백에서 Stage 1 을 호출하는 코드 보고 결정.
3. trajectory 실행 함수명 (`self._execute_trajectory` 자리) — 마찬가지로 production 콜백 / Stage 5 콜백 참고.
4. `Pose` import 가 모듈 상단에 없으면 추가: `from geometry_msgs.msg import Pose, PoseArray`.

---

## 2. `publish_test_waypoint.py` 신규 작성

경로: `~/sketch_robot_ws/src/sketch_control/sketch_control/publish_test_waypoint.py`

```python
#!/usr/bin/env python3
"""
Phase 4 Session 4 — Stage 1 단독 검증용 테스트 waypoint publisher.

용도:
- 로봇이 READY_POSE 인 상태에서 실행.
- tf2 로 base_link → tool0 실측해 orientation 추출.
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


# 보드 표면 위 한 점 (base_link 기준).
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

    def lookup_tool0_orientation(self, timeout_sec: float = 3.0):
        """base_link → tool0 transform 의 rotation 부분을 quaternion 으로 반환."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                trans = self.tf_buffer.lookup_transform(
                    'base_link', 'tool0', rclpy.time.Time())
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
        # 1) tool0 orientation 실측
        rot = self.lookup_tool0_orientation()
        if rot is None:
            self.get_logger().error(
                'base_link → tool0 transform 못 받음. '
                'robot_state_publisher / joint_states 활성 확인.')
            return False

        # 2) Pose 구성
        pose = Pose()
        pose.position.x = TARGET_X
        pose.position.y = TARGET_Y
        pose.position.z = TARGET_Z
        pose.orientation = rot  # READY 시점의 tool0 orientation 그대로

        msg = PoseArray()
        msg.header.frame_id = 'base_link'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.poses = [pose]

        # 3) subscriber 대기
        if not self.wait_for_subscriber():
            self.get_logger().warn(
                'subscriber 미발견. moveit_executor 활성 확인. 그래도 publish 시도.')

        # 4) publish (안전하게 5회 반복)
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
```

---

## 3. `setup.py` entry_point 추가

`src/sketch_control/setup.py` 의 `entry_points={'console_scripts': [...]}` 안에 다음 한 줄 추가:

```python
'publish_test_waypoint = sketch_control.publish_test_waypoint:main',
```

(기존 entry 들과 같은 형식으로 콤마 신경쓰면서.)

---

## 4. Build & 실행

### 4.1 Build

```bash
cd ~/sketch_robot_ws
colcon build --packages-select sketch_control --symlink-install
source install/setup.bash
```

### 4.2 실행 순서 (실로봇)

**안전 체크 먼저**:
- E-stop 손 닿는 위치
- 로봇 반경 1.3 m + 마진 클리어
- 펜던트 속도 30% 이하
- 첫 trigger 시 손은 E-stop 위에

```bash
# 터미널 1: rbpodo + MoveIt + moveit_executor 띄우기 (현재 사용 중인 실로봇 launch)
#   ★ 정확한 launch 파일은 본인 환경 확인 (Session 3 에서 쓰던 것)

# 펜던트로 READY_POSE 도달 확인 (base ≈ +179°)

# 터미널 2: 테스트 waypoint publish
ros2 run sketch_control publish_test_waypoint
# 기대 출력: [OK] published /sketch_waypoints  pos=(0.800, 0.000, 0.450)  quat=(...)

# 터미널 3: 로그 모니터
ros2 topic echo /joint_command   # 또는 실로봇 명령 토픽

# 터미널 4: trigger
ros2 topic pub --once /debug_trigger_stage1 std_msgs/Bool "{data: true}"
```

### 4.3 검증 포인트

- `[DEBUG] /debug_trigger_stage1 수신` 로그 출력
- `[DEBUG] Stage 1 target safe_pose: (0.650, 0.000, 0.450)` 로그 (= 0.80 - 0.15)
- Stage 1 planner trajectory 생성 성공
- 실로봇이 부드럽게 움직여 보드 앞 ~15 cm 위치에서 정지
- EE orientation 은 READY 와 거의 동일 유지 (Stage 1 은 위치만 옮기는 게 의도)

### 4.4 실패 시 디버깅 순서

1. `publish_test_waypoint` 가 `[OK] published ...` 안 뜨면 → tf2 lookup 실패. `ros2 run tf2_ros tf2_echo base_link tool0` 직접 확인.
2. trigger 후 콜백 로그 안 뜨면 → subscriber 등록 실패. `ros2 topic info /debug_trigger_stage1 --verbose` 로 subscriber 1개 확인.
3. "이미 실행 중" 출력되면 → 직전 실행 미종료. 노드 재시작.
4. planning 실패 → safe_pose IK 도달 불가. TARGET_Y, TARGET_Z 값을 보드 가시영역 내 다른 값으로 변경 (예: y=-0.15, z=0.5).
5. trajectory 생성됐는데 로봇 미동작 → `/joint_command` 또는 실로봇 명령 토픽 publish 됐는지 확인. rbpodo 드라이버 활성 상태 확인.

---

## 5. 작업 완료 보고 항목

다음 항목들 알려주세요:

1. 변경된 파일 목록 (예상: `moveit_executor.py`, `publish_test_waypoint.py` 신규, `setup.py`)
2. `colcon build --packages-select sketch_control --symlink-install` 결과 (성공 / 경고 / 에러)
3. `★` 표시 4곳에 실제로 어떤 식별자를 채웠는지 (특히 Stage 1 planner 함수명)
4. `ros2 topic info /debug_trigger_stage1 --verbose` 출력 (subscriber 1개 떠있는지)
5. 만약 build 후 첫 실행까지 했다면 첫 trigger 시점의 로그 일부 (특히 safe_pose 좌표 라인)
