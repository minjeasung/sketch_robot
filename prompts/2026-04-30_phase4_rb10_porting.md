# Phase 4 — RB10 포팅 작업 (Claude Code 용)

작성일: 2026-04-30
대상 파일: `src/sketch_control/sketch_control/moveit_executor.py` (1050줄)
원본 백업: `src/sketch_control/sketch_control/moveit_executor.py.bak_phase3` 로 먼저 복사할 것

## 0. 배경

이 코드는 시뮬(Isaac Sim + UR10)에서 검증 끝난 4-stage 파이프라인이다.
Phase 4에서는 동일한 코드를 실로봇(Rainbow Robotics RB10-1300, model_id `rb10_1300e_u`)에 옮긴다. 로봇 driver는 공식 `rbpodo_ros2`를 쓴다 (이미 `~/rb10_ws`에 빌드 완료, fake_hardware 테스트 통과).

`targets.py`는 robot-agnostic 이므로 **수정하지 말 것**.

## 1. 결정적 차이 — 두 개

### 1.1 Trajectory 실행 방식 통째 변경 (가장 중요)

시뮬에선 `JointState`를 `/joint_command` 토픽에 직접 publish해서 Isaac Sim articulation drive로 흘렸다 (`execute_trajectory_direct()` 함수 L345-383).

실로봇 RB10에는 `/joint_command` 토픽 없음. 대신 표준 ros2_control의 `joint_trajectory_controller`가 떠 있고, **`FollowJointTrajectory` action**으로 trajectory를 보내야 한다.

- Action name: `/joint_trajectory_controller/follow_joint_trajectory`
- Action type: `control_msgs.action.FollowJointTrajectory`
- Goal에 `trajectory` (`trajectory_msgs/JointTrajectory`) + tolerances 채우기
- Result로 `error_code` 받음

### 1.2 Tool 마운팅 축 변수화

원본은 `_brush_tip_to_tool0()` (L656-671)에서 토치가 `tool0`의 로컬 `+Y` 방향으로 뻗는다고 하드코딩. RB10 + 새 EoAT 마운팅에서는 어느 축이 될지 아직 미정. **변수화하되, 기본값은 `+y` 유지** (실로봇 캘리브 단계에서 실측 후 변경할 자리).

## 2. 변경 항목 — 정확한 매핑표

원본 → 새 값. 의미 못 박아두니 한 군데도 빠뜨리지 말 것.

| 항목 | 원본 (UR10) | 새 값 (RB10) |
|---|---|---|
| `PLANNING_GROUP` | `"ur_manipulator"` | `"mainpulation"` ⚠️ 오타 그대로 유지 (공식 SRDF 자체가 `mainpulation`) |
| `EE_LINK` | `"tool0"` | `"tcp"` |
| `BASE_FRAME` | `"base_link"` | `"link0"` |
| Robot URDF name | (UR) | `"rb"` (참고용, 코드 직접 영향 없음) |
| Joint names (운동학 순) | shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3 | `base, shoulder, elbow, wrist1, wrist2, wrist3` |
| `READY_POSE_JOINTS` 값 | `[-0.08, -1.6, 1.76, -1.76, -1.9, 3.14]` | **각 값 `0.0`으로 두고 `# TODO: RB10 실측 필요` 주석.** UR10 값을 RB10에 그대로 보내면 위험 |
| TF lookup `("world", "tool0")` | 그대로 | `("world", "tcp")` |
| Touch links | `["tool0", "wrist_3_link", "wrist_2_link", "flange"]` | `["tcp", "link6", "link5"]` (RB10 link 이름) |
| `/joint_command` publisher | 있음 | **삭제** |
| `execute_trajectory_direct()` | `/joint_command` publish loop | **`FollowJointTrajectory` action client로 재작성** (3절 참조) |
| `_brush_tip_to_tool0` | 로컬 `+Y` 하드코딩 | 마운팅 축 변수화 (4절 참조) |

### 변수명 일괄 rename

가독성을 위해 코드 전반의 `tool0` 표현을 `tcp`로 일괄 변경한다. 의미가 같으므로 안전한 변경이다.

| 원본 변수/식별자 | 새 이름 |
|---|---|
| `_safety_tool0_pose` | `_safety_tcp_pose` |
| `_retreat_tool0_pose` | `_retreat_tcp_pose` |
| `_stage3_tool0_wps` | `_stage3_tcp_wps` |
| `tool0_wps` (로컬 변수, 여러 곳) | `tcp_wps` |
| `safety_tool0` (로컬) | `safety_tcp` |
| `retreat_tool0` (로컬) | `retreat_tcp` |
| `_brush_tip_to_tool0` | `_brush_tip_to_tcp` |
| `_first_tool0` | `_first_tcp` |
| 로그 문자열 `"tool0=..."` | `"tcp=..."` |
| 주석 내 "tool0" | "tcp" (의미가 RB10 EE link면) |

⚠️ 단, 다음은 그대로 유지:
- 함수/변수가 **시뮬 잔재가 아니라 일반 의미의 EE pose**를 가리키면 OK — 이 케이스에선 모두 EE link 의미라 다 rename 대상
- docstring 내 "URDF tool0" 같은 역사적 언급은 "URDF tcp (RB10) / tool0 (UR10)" 식으로 명확히

## 3. `execute_trajectory_direct()` 재작성

기존 함수 시그니처 유지: `execute_trajectory_direct(self, traj, on_complete=None)`.
함수 이름도 유지 (호출처가 여러 곳).

내부 구현만 교체:

```python
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient

# __init__ 에서:
self.traj_action_client = ActionClient(
    self,
    FollowJointTrajectory,
    "/joint_trajectory_controller/follow_joint_trajectory",
)
# /joint_command 퍼블리셔는 제거.

def execute_trajectory_direct(self, traj, on_complete=None):
    """RB10 driver의 FollowJointTrajectory action으로 trajectory 전송.
    on_complete: action 성공 후 호출할 callback (다음 stage 트리거용)."""
    if not traj.joint_trajectory.points:
        self.get_logger().warn("빈 궤적")
        if on_complete is None:
            self.executing = False
        return

    if not self.traj_action_client.wait_for_server(timeout_sec=5.0):
        self.get_logger().error("FollowJointTrajectory action server 없음")
        self.executing = False
        return

    goal = FollowJointTrajectory.Goal()
    goal.trajectory = traj.joint_trajectory  # trajectory_msgs/JointTrajectory 그대로

    # path/goal tolerances 는 비워둠 (controller 디폴트 사용).
    # 필요시 차후에 GoalTolerance 추가.

    self.get_logger().info(
        f"FollowJointTrajectory 전송: {len(traj.joint_trajectory.points)} 포인트")

    send_future = self.traj_action_client.send_goal_async(goal)

    def _goal_response(fut):
        try:
            handle = fut.result()
        except Exception as e:
            self.get_logger().error(f"send_goal 실패: {e}")
            self.executing = False
            return
        if not handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            self.executing = False
            return
        result_future = handle.get_result_async()
        result_future.add_done_callback(_result_done)

    def _result_done(fut):
        try:
            res = fut.result().result
        except Exception as e:
            self.get_logger().error(f"trajectory result 실패: {e}")
            self.executing = False
            return
        # FollowJointTrajectory.Result.error_code: 0 = SUCCESSFUL
        if res.error_code != 0:
            self.get_logger().error(
                f"Trajectory 실행 에러 code={res.error_code}: {res.error_string}")
            self.executing = False
            return
        self.get_logger().info(">>> 궤적 실행 완료 (action SUCCESS)")
        if on_complete is not None:
            on_complete()
        else:
            self.executing = False

    send_future.add_done_callback(_goal_response)
```

### 주의

- 원본은 `threading.Thread`로 비동기 실행했지만, action 콜백 체인으로 자연스럽게 비동기가 되므로 `threading` 사용하지 말 것.
- `time.sleep(0.5)` (joint_state 안정화) 같은 시뮬 hack도 제거 — action이 실제 완료 시점을 신호로 줌.
- `import threading`, `import time` 이 다른 곳에서도 쓰이는지 확인 후 안전하면 import 제거.

## 4. `_brush_tip_to_tcp` (구 `_brush_tip_to_tool0`) 변수화

마운팅 축을 모듈 상수로 빼고, 함수가 그 값에 따라 동작하도록.

```python
# 모듈 상단 상수 영역
# 토치가 EE link (tcp) 의 로컬 어느 축으로 뻗는지.
# 값: "+x", "-x", "+y", "-y", "+z", "-z"
# 기본 +y 는 UR10 시뮬 시절 가정. RB10 실로봇 마운팅 후 실측해서 변경.
TORCH_MOUNT_AXIS = "+y"  # TODO: RB10 + 새 EoAT 마운팅 후 실측

# 함수
@staticmethod
def _local_axis_in_world(q, axis):
    """쿼터니언 q 기준 로컬 axis ('+x'/'-x'/...) 의 world 방향 단위벡터."""
    x, y, z, w = q.x, q.y, q.z, q.w
    if axis in ("+x", "-x"):
        v = np.array([1 - 2*(y*y + z*z), 2*(x*y + z*w), 2*(x*z - y*w)])
    elif axis in ("+y", "-y"):
        v = np.array([2*(x*y - z*w), 1 - 2*(x*x + z*z), 2*(y*z + x*w)])
    elif axis in ("+z", "-z"):
        v = np.array([2*(x*z + y*w), 2*(y*z - x*w), 1 - 2*(x*x + y*y)])
    else:
        raise ValueError(f"unknown axis: {axis}")
    if axis.startswith("-"):
        v = -v
    return v

@staticmethod
def _brush_tip_to_tcp(pose):
    """torch_tip 기준 좌표를 tcp 기준으로 변환.
    토치는 tcp 의 로컬 TORCH_MOUNT_AXIS 방향으로 TORCH_LENGTH 만큼 뻗음."""
    axis_world = MoveItExecutor._local_axis_in_world(pose.orientation, TORCH_MOUNT_AXIS)
    pos = np.array([pose.position.x, pose.position.y, pose.position.z])
    new_pos = pos - TORCH_LENGTH * axis_world
    new_pose = Pose()
    new_pose.position.x = float(new_pos[0])
    new_pose.position.y = float(new_pos[1])
    new_pose.position.z = float(new_pos[2])
    new_pose.orientation = copy.deepcopy(pose.orientation)
    return new_pose
```

또한 토치 attached collision (L284-310) 의 cylinder 회전도 `TORCH_MOUNT_AXIS`에 맞춰 계산해야 한다.
원본은 X-축 90도 회전으로 cylinder default `+z` 를 `+y` 로 돌렸음. 변수화:

```python
def _torch_attach_quat(axis):
    """SolidPrimitive.CYLINDER (default +z) 를 axis 방향으로 회전시키는 quaternion."""
    half = math.pi / 4.0  # 90도의 절반
    if axis == "+z":
        return (0.0, 0.0, 0.0, 1.0)
    if axis == "-z":
        return (1.0, 0.0, 0.0, 0.0)
    if axis == "+y":
        return (-math.sin(half), 0.0, 0.0, math.cos(half))  # X축 -90도
    if axis == "-y":
        return (math.sin(half), 0.0, 0.0, math.cos(half))   # X축 +90도
    if axis == "+x":
        return (0.0, math.sin(half), 0.0, math.cos(half))   # Y축 +90도
    if axis == "-x":
        return (0.0, -math.sin(half), 0.0, math.cos(half))  # Y축 -90도
    raise ValueError(f"unknown axis: {axis}")
```

⚠️ 위 quaternion 공식들은 **수학적으로 검증된 값으로 채워야 함** (LLM이 부호 실수 흔함). 만약 prompt 받는 측이 자신 없으면 `scipy.spatial.transform.Rotation` 으로 동적 생성하는 방식 쓸 것:

```python
from scipy.spatial.transform import Rotation as R
def _torch_attach_quat(axis):
    target = {"+x": [1,0,0], "-x": [-1,0,0], "+y": [0,1,0],
              "-y": [0,-1,0], "+z": [0,0,1], "-z": [0,0,-1]}[axis]
    src = np.array([0,0,1])  # cylinder default
    tgt = np.array(target, dtype=float)
    if np.allclose(src, tgt):
        return (0.0, 0.0, 0.0, 1.0)
    if np.allclose(src, -tgt):
        return (1.0, 0.0, 0.0, 0.0)
    rot, _ = R.align_vectors(tgt[None, :], src[None, :])
    q = rot.as_quat()  # [x, y, z, w]
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
```

cylinder pose의 `position` 도 마찬가지로 axis 방향에 따라 `TORCH_LENGTH/2` 만큼 옮겨져야 함. 원본 (`+y` 12.5cm) 패턴을 일반화:

```python
def _torch_attach_offset(axis):
    """cylinder 중심 위치 (tcp 기준), TORCH_LENGTH/2 만큼 axis 방향."""
    sign = -1.0 if axis.startswith("-") else 1.0
    a = axis[1]
    half = TORCH_LENGTH / 2.0 * sign
    if a == "x":
        return (half, 0.0, 0.0)
    if a == "y":
        return (0.0, half, 0.0)
    if a == "z":
        return (0.0, 0.0, half)
```

## 5. `READY_POSE_JOINTS` — 값 비우기

원본:
```python
READY_POSE_JOINTS = {
    "shoulder_pan_joint":   -0.08,
    ...
}
```

새:
```python
# RB10 joint 운동학 순서 (URDF 기준).
# 주의: /joint_states 토픽은 알파벳 순으로 발행됨 (base, elbow, shoulder, wrist1, wrist2, wrist3) —
# 이 dict는 이름 매핑이라 순서 무관, 안전.
# 각 값은 RB10 실로봇에서 teach pendant로 안전한 자세 만든 후 /joint_states 읽어 실측해야 함.
READY_POSE_JOINTS = {
    "base":     0.0,  # TODO: RB10 실측
    "shoulder": 0.0,  # TODO: RB10 실측
    "elbow":    0.0,  # TODO: RB10 실측
    "wrist1":   0.0,  # TODO: RB10 실측
    "wrist2":   0.0,  # TODO: RB10 실측
    "wrist3":   0.0,  # TODO: RB10 실측
}
```

## 6. `FIXED_TOOL0_QUAT` 처리 (있다면)

원본 grep해서 `FIXED_TOOL0_QUAT` 또는 비슷한 캘리브 quaternion 상수 있으면 비슷하게 처리: TODO 마킹 + UR10 값 제거 + 0,0,0,1 (identity) 로 두고 주석.
(grep으로 못 찾으면 무시 — 어쩌면 이 코드엔 없을 수도)

## 7. `ROBOT_ORIGIN` 검토

L65: `ROBOT_ORIGIN = (0.0, 0.0, 0.0)` 로 박혀 있고 L179-181에서 waypoint 변환에 쓰임.

RB10도 `link0`이 `world`에 fixed_joint로 붙어있고 (offset 0), launch에서 base를 origin에 두므로 그대로 `(0,0,0)` 유지해도 됨. 단 주석을 RB10 기준으로 갱신:

```python
# RB10 link0 가 world 원점에 fixed_joint로 박혀있음. 작업대 위에 따로 옮기면 변경 필요.
ROBOT_ORIGIN = (0.0, 0.0, 0.0)
```

## 8. Header docstring 갱신

L1-7의 파일 docstring이 시뮬 전제 (Isaac Sim, /joint_command) 로 적혀있음. 다음과 같이 교체:

```python
"""
MoveIt Executor - 4-stage 용접 모션 (Plan + FollowJointTrajectory 실행)

대상 로봇: Rainbow Robotics RB10-1300 (rb10_1300e_u, rbpodo_ros2 driver)
EE link: tcp / Base frame: link0 / Planning group: mainpulation (SRDF 오타)

4-stage 파이프라인:
  Stage 1: 자유 plan (현재 자세 → safety pose)  [OMPL via /move_action]
  Stage 2: cartesian (safety → 표면 첫 점)        [/compute_cartesian_path]
  Stage 3: cartesian path (표면 위 스케치 추종)
  Stage 4: cartesian (표면 끝 → retreat pose)
  Stage 5: 자유 plan (retreat → READY_POSE)

토치(EoAT)는 tcp 의 TORCH_MOUNT_AXIS 방향으로 뻗음. 마운팅 변경 시 그 상수만 갱신.
"""
```

## 9. 산출물 + 검증

### 변경 후 file 산출
- `src/sketch_control/sketch_control/moveit_executor.py` (수정본)
- `src/sketch_control/sketch_control/moveit_executor.py.bak_phase3` (원본 백업, 미리 cp 로 보존)

### Sanity check
포팅 끝나면 다음 항목을 작업자에게 reporting:

1. ✅ 모든 `tool0` / `base_link` / `ur_manipulator` / `shoulder_pan_joint` 등 UR10 식별자가 코드에서 사라졌는지 (grep으로 확인)
2. ✅ `READY_POSE_JOINTS` 의 6개 값이 모두 0.0 + TODO 주석인지
3. ✅ `execute_trajectory_direct` 가 더 이상 `/joint_command` publish 안 하고 action client 사용하는지
4. ✅ `TORCH_MOUNT_AXIS` 상수 정의됨 + 디폴트 `"+y"` + 사용처(`_brush_tip_to_tcp`, attached collision torch) 다 그 상수 따르는지
5. ✅ Python syntax 에러 없는지 (`python3 -m py_compile moveit_executor.py`)
6. ✅ 함수 호출처가 다 일관되게 rename 됐는지 (예: `_brush_tip_to_tool0` 호출이 한 군데도 안 남았는지)

### Sanity check 위반 항목 발견 시
TODO 주석 + 작업자에게 명시 reporting. 임의로 추측해 채우지 말 것.

## 10. 작업하지 말 것 (out of scope)

- `targets.py` 수정 ❌ (robot-agnostic)
- `READY_POSE_JOINTS` 값 추측해서 채우기 ❌ (실측 후 수동 갱신)
- `TORCH_MOUNT_AXIS` 디폴트 `+y` 외로 변경 ❌ (실측 후 수동 갱신)
- 4-stage 파이프라인 로직 변경 ❌ (이미 검증됨, 손대지 마)
- launch 파일 수정 ❌ (별도 단계에서 다룰 것)
- `__init__.py`, `setup.py` ❌ (변경 불필요)
- IK fallback / planner 종류 변경 ❌

## 11. 다음 세션 (이 작업 완료 후) 액션

이 prompt 따른 산출물이 만들어진 후, 사용자가 직접 할 일:

1. RB10 실로봇 켜고 use_fake_hardware:=false 로 driver launch
2. teach pendant 로 안전 자세 만들기 → `ros2 topic echo /joint_states --once` → 6개 값을 `READY_POSE_JOINTS`에 박아넣기
3. 토치 마운팅 보고 `TORCH_MOUNT_AXIS` 결정 + 박아넣기
4. 사용자 패키지 빌드 (`colcon build`) → 노드 띄우기 → 작은 동작부터 검증
