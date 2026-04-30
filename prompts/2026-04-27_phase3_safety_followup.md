# Phase 3 — approach 4-stage 파이프라인 안전성 보강

## 배경

직전 적용된 `2026-04-27_phase3_approach_4stage.md` 패치 후 Claude Code 가 보고한
3 개 안전 이슈를 처리.

대상 파일: `~/sketch_robot_ws/src/sketch_control/sketch_control/moveit_executor.py`

## 변경 사항

### 1. Stage 1 시작 전 `scene_confirmed` 체크

**문제**: `on_execute` → `stage1_approach_free` 가 호출될 때 PlanningScene 에
attached torch / wall 등록이 아직 안 끝났을 수 있음. 그러면 OMPL 이 충돌 객체 모르고
경로 생성 → 토치가 벽 통과하는 경로 가능.

**수정**: `stage1_approach_free` 진입 시 `self.scene_confirmed` 가 False 면 최대 5초간 대기.
타임아웃되면 경고 + 그래도 진행 (개발 편의 — 너무 strict 하면 디버그 어려움).

`stage1_approach_free` 함수의 맨 앞에 다음 블록 추가:

```python
def stage1_approach_free(self):
    # PlanningScene 검증 대기 (attached torch + wall 등록 확인)
    import time
    wait_start = time.time()
    while not self.scene_confirmed and (time.time() - wait_start) < 5.0:
        self.get_logger().info("PlanningScene 검증 대기 중...")
        time.sleep(0.5)
    if not self.scene_confirmed:
        self.get_logger().warn(
            "PlanningScene 미검증 (5초 타임아웃). 충돌 회피 약할 수 있음.")
    else:
        self.get_logger().info("PlanningScene 검증 OK")

    if not self.move_action_client.wait_for_server(timeout_sec=5.0):
        # ... (기존 코드 유지)
```

기존의 `if not self.move_action_client.wait_for_server(...)` 줄 앞에 위 블록을 삽입.

### 2. planner_id 동적 fallback

**문제**: `PLANNER_ID = "RRTConnect"` 가 OMPL 설정에서 등록 안 돼있을 수 있음.
MoveIt 1 시절 표기인 `"RRTConnectkConfigDefault"` 로 등록된 시스템도 있음.

**수정**: stage 1 의 result callback `_stage1_result` 에서 planner mismatch 에러
(MoveItErrorCodes.PLANNING_FAILED = 99999 또는 INVALID_MOTION_PLAN = -1 등)
감지 시 한 번 자동 재시도 with 빈 planner_id (서버 기본값 사용).

`_stage1_result` 의 error 처리 부분을 다음으로 교체:

기존:
```python
def _stage1_result(self, future):
    try:
        result = future.result().result
    except Exception as e:
        self.get_logger().error(f"STAGE 1 result 실패: {e}")
        self.executing = False
        return
    if result.error_code.val != 1:  # MoveItErrorCodes.SUCCESS = 1
        self.get_logger().error(
            f"STAGE 1 planning 실패 error_code={result.error_code.val}")
        self.executing = False
        return
    # ... (이후 코드 유지)
```

변경:
```python
def _stage1_result(self, future):
    try:
        result = future.result().result
    except Exception as e:
        self.get_logger().error(f"STAGE 1 result 실패: {e}")
        self.executing = False
        return
    if result.error_code.val != 1:  # MoveItErrorCodes.SUCCESS = 1
        self.get_logger().error(
            f"STAGE 1 planning 실패 error_code={result.error_code.val}")
        # planner_id mismatch 가능성 — 빈 planner_id 로 한 번 재시도
        if not getattr(self, "_stage1_retried", False):
            self._stage1_retried = True
            self.get_logger().warn(
                f"planner_id='{PLANNER_ID}' 실패 — 서버 기본 planner 로 재시도")
            self._retry_stage1_with_default_planner()
            return
        self._stage1_retried = False
        self.executing = False
        return

    # 성공 시 retry flag 리셋
    self._stage1_retried = False

    # ... (이후 코드 유지)
```

그리고 새 함수 `_retry_stage1_with_default_planner` 를 `stage1_approach_free` 바로 뒤에 추가:

```python
def _retry_stage1_with_default_planner(self):
    """planner_id 를 비워서 MoveGroup 의 기본 planner 로 재시도."""
    goal = MoveGroup.Goal()
    goal.request.group_name = PLANNING_GROUP
    rs = RobotState()
    rs.joint_state = self.current_joint_state
    rs.is_diff = False
    goal.request.start_state = rs
    goal.request.goal_constraints = [
        self._make_pose_constraints(self._safety_tool0_pose)
    ]
    # planner_id 비움 — 서버 기본 사용
    goal.request.planner_id = ""
    goal.request.allowed_planning_time = ALLOWED_PLANNING_TIME
    goal.request.num_planning_attempts = PLANNING_ATTEMPTS
    goal.request.max_velocity_scaling_factor = 0.3
    goal.request.max_acceleration_scaling_factor = 0.3
    goal.planning_options.plan_only = True
    goal.planning_options.planning_scene_diff.is_diff = True

    future = self.move_action_client.send_goal_async(goal)
    future.add_done_callback(self._stage1_goal_response)
```

`__init__` 에 retry flag 초기화 추가:

```python
self._stage1_retried = False
```

(기존 `_safety_tool0_pose`, `_retreat_tool0_pose`, `_stage3_tool0_wps` 와 같은 위치)

### 3. Stage 3 fail 시 executing flag 누수 수정 (기존 버그)

**문제**: `_cartesian_done` 의 `fraction < 0.5` 분기가 return 만 하고
`self.executing = False` 안 함. 한 번 fail 하면 다음 Submit 영원히 막힘.

**수정**: `_cartesian_done` 안의 fail 분기에 flag 리셋 추가.

기존:
```python
def _cartesian_done(self, future):
    try:
        resp = future.result()
    except Exception as e:
        self.get_logger().error(f"Cartesian 서비스 실패: {e}")
        return

    fraction = resp.fraction
    self.get_logger().info(f"Cartesian path: {fraction*100:.1f}% 달성")
    if fraction < 0.5:
        self.get_logger().error(
            "50% 미만 -> 스케치를 재생성하거나 타겟을 변경하세요.")
        return
    if fraction < 0.95:
        self.get_logger().warn(f"{fraction*100:.1f}% -> 부분 실행")

    traj = self._rescale_trajectory(resp.solution, scale=0.3)
    self.execute_trajectory_direct(
        traj, on_complete=self.stage4_retreat)
```

변경 (각 early return 에 `self.executing = False` 추가):
```python
def _cartesian_done(self, future):
    try:
        resp = future.result()
    except Exception as e:
        self.get_logger().error(f"Cartesian 서비스 실패: {e}")
        self.executing = False
        return

    fraction = resp.fraction
    self.get_logger().info(f"Cartesian path: {fraction*100:.1f}% 달성")
    if fraction < 0.5:
        self.get_logger().error(
            "50% 미만 -> 스케치를 재생성하거나 타겟을 변경하세요.")
        self.executing = False
        return
    if fraction < 0.95:
        self.get_logger().warn(f"{fraction*100:.1f}% -> 부분 실행")

    traj = self._rescale_trajectory(resp.solution, scale=0.3)
    self.execute_trajectory_direct(
        traj, on_complete=self.stage4_retreat)
```

## 검증

```bash
cd ~/sketch_robot_ws
colcon build --packages-select sketch_control --symlink-install
source install/setup.bash
python3 -c "from sketch_control.moveit_executor import MoveItExecutor; print('OK')"
```

## 작업 가이드

1. 변경 1, 2, 3 을 순서대로 적용.
2. Build + import 검증.
3. 변경 요약 출력 (각 변경의 적용 위치 라인 번호).
