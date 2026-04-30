# Phase 3 — READY_POSE 경유 라우팅 (Stage 5 추가)

## 배경

Wall 이 PlanningScene 에 등록 안 되는 이슈 (ApplyPlanningScene success=True 인데
GetPlanningScene 결과 비어있음, frame_id base_link/world/World 모두 동일 증상) 가
ROS2 Jazzy + UR + Isaac Sim + ur_moveit_config 의 누적 셋업 문제로 추정됨.
TF tree 진단 결과:
- `world / World` 와 `base_link` 사이 TF 끊김
- `/move_group` 의 `robot_description_planning.frame_id` parameter 미설정
- UR chain (base_link → wrist_3_link) 자체는 정상

→ wall 등록 디버깅 포기. **wall 가까이 안 지나가는 경로 설계** 로 우회.

## 해결 전략

매 Submit 종료 후 **READY_POSE 로 자동 복귀** 하는 Stage 5 추가.

```
[변경 전 흐름]
1차: READY → safety_1 → surface_1 → bead_1 → retreat_1
2차: retreat_1 → safety_2 → ...   ← 이 Stage 1 에서 wall 닿음

[변경 후 흐름]
1차: READY → safety_1 → surface_1 → bead_1 → retreat_1 → READY ★
2차: READY → safety_2 → surface_2 → bead_2 → retreat_2 → READY ★
3차: READY → ...
```

매번 Stage 1 시작점이 READY (wall 에서 충분히 멀음) 이므로 OMPL/PTP 의 충돌 회피
약함과 무관하게 wall 안 닿음.

## 설계 — Stage 5 의 plan 방식

| 옵션 | 장점 | 단점 |
|---|---|---|
| **A. Joint goal (MotionPlanRequest + JointConstraint)** | READY_POSE 의 정확한 joint 값으로 복귀, IK ambiguity 없음 | 새 함수 필요 |
| B. Pose goal (현재 Stage 1 와 같은 방식, IK 로 변환) | 코드 재사용 | IK 가 다른 config 선택 가능 |
| C. 직접 joint command (MoveIt 우회) | 가장 단순 | 충돌 검사 안 됨, 안전성 낮음 |

→ **옵션 A** 채택. Stage 1 의 OMPL planning 인프라 재사용 (planner_id, planning_pipeline 동일),
goal 만 PositionConstraint+OrientationConstraint 대신 JointConstraint 로.

## 사전 확인 — READY_POSE 값 위치

먼저 코드에서 READY_POSE 가 어떻게 정의돼있는지 확인:

```bash
grep -rn -E "READY_POSE|ready_pose|HOME_POSE" \
  ~/sketch_robot_ws/src/sketch_control/sketch_control/ \
  ~/sketch_robot_ws/src/sketch_control/launch/ 2>/dev/null
```

기대 결과: `READY_POSE = [...]` 같은 list 가 어딘가에 있을 것. 보통 `moveit_executor.py`
또는 `phase1_python.launch.py` 또는 `core.launch.py` 의 controller initial state 옆.

찾으면 그 값 메모. 없으면 아래 기본값 사용 (UR5e 의 흔한 ready pose):
```python
READY_POSE_JOINTS = {
    "shoulder_pan_joint":   0.0,
    "shoulder_lift_joint": -1.5708,    # -90°
    "elbow_joint":          1.5708,    # +90°
    "wrist_1_joint":       -1.5708,    # -90°
    "wrist_2_joint":       -1.5708,    # -90°
    "wrist_3_joint":        0.0,
}
```

(만약 launch 의 initial state 가 다르면 그 값 그대로 사용. 위는 fallback.)

## 변경 사항

대상 파일: `~/sketch_robot_ws/src/sketch_control/sketch_control/moveit_executor.py`

### 1. import 추가

기존 `JointConstraint` 가 import 안 돼있으면 추가:

```python
from moveit_msgs.msg import (
    PositionConstraint, OrientationConstraint, JointConstraint,
    Constraints, MotionPlanRequest, ...
)
```

### 2. 상수 추가

파일 상단 (다른 상수 옆, `SAFETY_OFFSET`, `RETREAT_OFFSET` 근처):

```python
# Stage 5 — READY_POSE 복귀
RETURN_TO_READY = True   # False 면 Stage 5 비활성 (디버깅용)

READY_POSE_JOINTS = {
    "shoulder_pan_joint":   0.0,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint":          1.5708,
    "wrist_1_joint":       -1.5708,
    "wrist_2_joint":       -1.5708,
    "wrist_3_joint":        0.0,
}
# ↑ 사전 확인에서 찾은 진짜 READY_POSE 값으로 교체. 안 찾았으면 위 fallback 유지.
```

### 3. 새 함수 — `plan_to_joint_target`

기존 `plan_to_pose_target` 또는 비슷한 함수 옆에 추가:

```python
def plan_to_joint_target(self, joint_dict, planner_id=None, timeout=10.0):
    """Plan and execute movement to a target joint configuration.
    
    Args:
        joint_dict: {joint_name: position_rad, ...}
        planner_id: Override default. None 이면 self.PLANNER_ID 사용.
        timeout: Planning + execution timeout in seconds.
    Returns:
        bool — success
    """
    # MotionPlanRequest 빌드
    req = MotionPlanRequest()
    req.group_name = self.GROUP_NAME           # "ur_manipulator"
    req.planner_id = planner_id or self.PLANNER_ID
    req.num_planning_attempts = 5
    req.allowed_planning_time = 5.0
    req.max_velocity_scaling_factor = 0.3
    req.max_acceleration_scaling_factor = 0.3
    
    # Goal: JointConstraint 들로
    constraints = Constraints()
    for joint_name, target_pos in joint_dict.items():
        jc = JointConstraint()
        jc.joint_name = joint_name
        jc.position = target_pos
        jc.tolerance_above = 0.01    # 0.6° 허용
        jc.tolerance_below = 0.01
        jc.weight = 1.0
        constraints.joint_constraints.append(jc)
    req.goal_constraints = [constraints]
    
    # 현재 state 를 start state 로 설정 (필수)
    req.start_state.joint_state = self.current_joint_state
    req.start_state.is_diff = False
    
    # Stage 1 의 planning action client 와 동일한 인터페이스로 호출
    # 기존 코드의 plan_to_pose_target 의 action call 부분 참조해서
    # MotionPlanRequest 만 위 req 로 교체
    
    self.get_logger().info(
        f"=== STAGE 5: return to READY_POSE (joint goal, {req.planner_id}) ==="
    )
    
    # ↓ 아래 부분은 기존 plan_to_pose_target 의 action call 패턴 그대로 재사용
    #   (MoveGroupAction goal 빌드 → send_goal → wait → execute trajectory)
    #   기존 함수가 어떻게 plan + execute 하는지에 따라 정확한 줄 다름.
    #   핵심: MotionPlanRequest 를 위 req 로 set, response 의 trajectory 를 execute.
    
    # 실패 시 직접 joint command fallback (안전성 마지막 보루)
    # — 이 부분은 옵션. 너무 복잡하면 일단 빼고 그냥 plan 실패 시 False 반환.
    
    return success  # bool
```

**중요**: 이 함수의 실제 plan + execute 부분은 기존 `plan_to_pose_target` (또는 Stage 1
처리하는 함수) 의 코드 패턴을 그대로 따라 하는 게 안전. MotionPlanRequest 의 goal 부분만
JointConstraint 로 바꾸고 나머지 (action call, trajectory execute) 는 동일.

### 4. on_execute (또는 main pipeline 함수) 끝부분 수정

Stage 4 (retreat cartesian) 가 끝난 후 아래 추가:

기존 마지막 부분 (예시):
```python
# Stage 4 끝
self.get_logger().info(">>> 모든 stage 완료 (1→2→3→4)")
self.publish_status("done")
return
```

다음으로 변경:
```python
# Stage 4 끝
if RETURN_TO_READY:
    # === STAGE 5: return to READY_POSE (joint goal) ===
    success = self.plan_to_joint_target(READY_POSE_JOINTS)
    if not success:
        self.get_logger().warn(
            "Stage 5 (READY 복귀) 실패. 다음 Submit 의 시작 위치 부적합 가능."
        )
        # 실패해도 전체 프로세스 종료 — 사용자가 수동 복귀 시도
    else:
        self.get_logger().info("Stage 5 완료: READY_POSE 복귀")

self.get_logger().info(">>> 모든 stage 완료 (1→2→3→4→5)")
self.publish_status("done")
return
```

### 5. 첫 Submit 의 안전성 (옵션, 권장)

첫 Submit 시 로봇이 진짜로 READY_POSE 에 있는지 launch 가 보장 못 할 수 있음.
`on_execute` 의 시작 부분에서 `current_joint_state` 가 READY_POSE 와 가까운지 확인:

```python
# on_execute 시작 부분, Stage 1 시작 직전
if not self._is_at_ready_pose():
    self.get_logger().warn(
        "현재 자세가 READY_POSE 와 다름. 먼저 READY 복귀 필요."
    )
    # 자동 복귀
    if not self.plan_to_joint_target(READY_POSE_JOINTS):
        self.get_logger().error("READY 복귀 실패. Submit 중단.")
        self.publish_status("error: cannot reach READY")
        return

# (기존 Stage 1 로직 진행)
```

`_is_at_ready_pose` 헬퍼 추가:

```python
def _is_at_ready_pose(self, tol_rad=0.05):
    """Check if current joint state is near READY_POSE (within tol_rad per joint)."""
    if self.current_joint_state is None:
        return False
    cs = dict(zip(
        self.current_joint_state.name,
        self.current_joint_state.position
    ))
    for jn, target in READY_POSE_JOINTS.items():
        if jn not in cs:
            return False
        if abs(cs[jn] - target) > tol_rad:
            return False
    return True
```

이건 옵션이지만 권장. 첫 Submit 의 시작 자세가 random 이면 Stage 1 의 path 가 wall 닿을 수 있음.

## Build

```bash
cd ~/sketch_robot_ws
colcon build --packages-select sketch_control --symlink-install
source install/setup.bash
```

Build 에러 시:
- `JointConstraint` import 누락 → import 줄 확인
- `self.GROUP_NAME` / `self.PLANNER_ID` 가 다른 이름으로 정의됐을 수 있음 → grep 으로 확인

## 테스트 절차

### 1. 정리

```bash
pkill -9 -f isaac_sim
sleep 5
pkill -9 -f "ros2|moveit|rviz|tcp_endpoint|robot_state|world_to|move_group|weld"
sleep 5
```

### 2. Isaac Sim + launch

```bash
~/sketch_robot_ws/run_isaac_sim.sh
# 별도 터미널
source /opt/ros/jazzy/setup.bash
source ~/sketch_robot_ws/install/setup.bash
ros2 launch sketch_control phase2_unity.launch.py 2>&1 | tee /tmp/phase3_ready_test.log
```

### 3. Unity 연결

### 4. Stress test — 점 3개 다른 위치

벽 위 3 군데 (왼쪽 위, 가운데, 오른쪽 아래) 클릭 → Submit (각각 따로).

각 Submit 마다 launch 로그에서 확인:

```
=== STAGE 1: free-space approach ===
... (planning + execute)
=== STAGE 2: linear approach ===
=== STAGE 3: bead ===
=== STAGE 4: retreat ===
=== STAGE 5: return to READY_POSE ===
... (joint planning + execute)
Stage 5 완료: READY_POSE 복귀
>>> 모든 stage 완료 (1→2→3→4→5)
```

Isaac Sim 영상에서 확인:
1. 매 Submit 종료 시 로봇이 READY 자세로 돌아가는지 (팔 들고 토치 아래 향함)
2. 다음 Submit 시작 시 다시 READY 부터 출발
3. 어느 Stage 에서도 토치가 wall 안 닿음 (특히 Stage 1)

### 5. 결과 보고

- 영상 또는 사진 (가능하면)
- launch 로그 (각 stage 별 fraction / planning 결과)
- 토치가 wall 닿는지 여부 (이게 본 평가)
- Stage 5 의 평균 시간 (1초? 3초? 너무 길면 튜닝)

## 알려진 함정

1. **MotionPlanRequest 의 start_state**: 반드시 `current_joint_state` 로 설정.
   비어있으면 MoveIt 이 last known state 가정해서 잘못된 plan 생성.

2. **`max_velocity_scaling_factor`**: Stage 5 는 빈 공간 이동이라 0.5~1.0 까지 올려도
   안전. Stage 1 의 0.3 보다 빨라도 됨. 시간 절약.

3. **READY_POSE 가 singularity 근처**: shoulder_lift = -90° 는 UR 의 elbow up 자세.
   가끔 IK 가 elbow down 으로 풀려서 path 가 길어질 수 있음. JointConstraint 사용
   덕분에 이 문제는 없음 (정확한 config 강제).

4. **Stage 5 실패 시**: warn 만 찍고 done 처리. 다음 Submit 시 `_is_at_ready_pose()`
   에서 다시 시도. 두 번 연속 실패면 launch 재시작 권장.

## 복원

문제 시 RETURN_TO_READY = False 한 줄로 비활성:

```bash
sed -i 's/RETURN_TO_READY = True/RETURN_TO_READY = False/' \
   ~/sketch_robot_ws/src/sketch_control/sketch_control/moveit_executor.py
cd ~/sketch_robot_ws && colcon build --packages-select sketch_control --symlink-install
```

이러면 Phase 3 직전 4-stage 동작으로 즉시 복귀 (Stage 5 코드는 남아있지만 호출 안 됨).
