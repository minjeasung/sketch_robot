# Phase 3 — moveit_executor.py 의 approach 단계를 4-stage 파이프라인으로 재구성

## 배경

`~/sketch_robot_ws/src/sketch_control/sketch_control/moveit_executor.py` 의 현재 동작:

1. `on_execute()` → `approach_to_first()` 호출
2. `approach_to_first()` 가 `compute_ik` service 로 첫 점에서 5cm 떨어진 곳의 IK 계산
3. `_approach_ik_done()` 이 현재 joint → 목표 joint 를 50 step 직선 보간 (joint space)
4. 도착 후 `plan_cartesian()` 으로 표면 위 비드 cartesian path 실행

문제: 3단계의 joint 직선 보간은 충돌 회피 0. 카르테시안 공간에서 곡선이 되어 토치가 벽을 휩쓸고 지나갈 수 있음. approach 시작점이 벽에서 5cm 밖에 안 떨어져있어 곡선이 벽 안으로 들어감.

## 목표

approach 단계를 4-stage 파이프라인으로 재구성:

- Stage 1: 현재 → 안전 진입점 (벽에서 15cm 후퇴) — MoveGroup action (OMPL/RRTConnect) 로 자유 경로
- Stage 2: 안전점 → 표면 첫 점 — cartesian path (짧은 직선 진입)
- Stage 3: 표면 위 비드 — cartesian path (기존 plan_cartesian 그대로)
- Stage 4: 마지막 점 → 후퇴점 (벽에서 15cm 후퇴) — cartesian path (짧은 직선 후퇴)

각 stage 가 다음 stage 의 callback chain. 어디서 fail 하면 명확한 에러 로그와 함께 중단.

## 변경 사항 상세

### 1. Import 추가

파일 상단 import 영역에 추가:

```python
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    PositionConstraint,  # 기존 import 라인에 추가
)
```

기존 `from moveit_msgs.msg import (...)` 안에 `PositionConstraint` 추가.

### 2. 상수 추가

`BRUSH_PRESS_DEPTH = 0.003` 라인 근처에 추가:

```python
SAFETY_OFFSET = 0.15   # Stage 1 안전 진입점: 표면 normal 방향 15cm
RETREAT_OFFSET = 0.15  # Stage 4 후퇴점: 표면 normal 방향 15cm
PLANNER_ID = "RRTConnect"
ALLOWED_PLANNING_TIME = 5.0
PLANNING_ATTEMPTS = 5
```

### 3. `__init__` 안에 ActionClient 추가

기존 `self.ik_client = self.create_client(...)` 라인 뒤에 추가:

```python
# MoveGroup action client (Stage 1: 자유 경로 planning)
self.move_action_client = ActionClient(self, MoveGroup, "/move_action")

# 다음 stage 로 넘기는 데 쓰는 상태
self._safety_tool0_pose = None
self._retreat_tool0_pose = None
self._stage3_tool0_wps = None  # Stage 2 끝났을 때 stage 3 가 쓸 waypoints
```

### 4. 헬퍼 함수 추가

클래스 안 적당한 위치 (예: `_brush_tip_to_tool0` 위) 에 새 헬퍼 메서드 3 개 추가:

```python
def _offset_along_normal(self, pose, distance):
    """pose 를 active target 의 표면 normal 방향으로 distance 만큼 후퇴."""
    target = get_target(self.cfg, self.active_target_name)
    _, n = get_surface_plane(target)
    out = copy.deepcopy(pose)
    out.position.x += distance * float(n[0])
    out.position.y += distance * float(n[1])
    out.position.z += distance * float(n[2])
    return out

def _make_pose_constraints(self, pose, link_name=EE_LINK, frame=BASE_FRAME):
    """Pose 를 MoveGroup goal 의 Constraints 로 변환."""
    c = Constraints()

    pc = PositionConstraint()
    pc.header.frame_id = frame
    pc.link_name = link_name
    pc.target_point_offset.x = 0.0
    pc.target_point_offset.y = 0.0
    pc.target_point_offset.z = 0.0
    sp = SolidPrimitive()
    sp.type = SolidPrimitive.SPHERE
    sp.dimensions = [0.005]  # 5mm tolerance
    pc.constraint_region.primitives.append(sp)
    region_pose = Pose()
    region_pose.position = pose.position
    region_pose.orientation.w = 1.0
    pc.constraint_region.primitive_poses.append(region_pose)
    pc.weight = 1.0
    c.position_constraints.append(pc)

    oc = OrientationConstraint()
    oc.header.frame_id = frame
    oc.link_name = link_name
    oc.orientation = pose.orientation
    oc.absolute_x_axis_tolerance = 0.05
    oc.absolute_y_axis_tolerance = 0.05
    oc.absolute_z_axis_tolerance = 0.05
    oc.weight = 1.0
    c.orientation_constraints.append(oc)

    return c

def _compute_snapped_tool0_waypoints(self):
    """current_waypoints 를 표면 스냅 + densify + tool0 변환한 결과 반환.
    Stage 2/3 양쪽이 같은 변환 결과를 써야 일관됨.
    plan_cartesian 의 기존 로직과 동일하지만 분리해서 재사용.
    Returns: (snapped_tip_wps, tool0_wps, target, n)
    """
    target = get_target(self.cfg, self.active_target_name)
    sp, n = get_surface_plane(target)
    fixed_q = ee_quat_for_target(target)
    snapped = []
    for wp in self.current_waypoints:
        p = np.array([wp.position.x, wp.position.y, wp.position.z])
        delta = float(np.dot(p - sp, n))
        projected = p - delta * n - n * BRUSH_PRESS_DEPTH
        sw = copy.deepcopy(wp)
        sw.position.x = float(projected[0])
        sw.position.y = float(projected[1])
        sw.position.z = float(projected[2])
        sw.orientation.x = float(fixed_q[0])
        sw.orientation.y = float(fixed_q[1])
        sw.orientation.z = float(fixed_q[2])
        sw.orientation.w = float(fixed_q[3])
        snapped.append(sw)
    densified = self._densify_waypoints(snapped, spacing_m=0.005)
    tool0_wps = [self._brush_tip_to_tool0(wp) for wp in densified]
    return densified, tool0_wps, target, n
```

### 5. `on_execute` 변경

기존:

```python
def on_execute(self, msg: Bool):
    if not msg.data or not self.current_waypoints:
        ...
    self.get_logger().info("1단계: 첫 점까지 approach (plan_only)")
    self.approach_to_first()
```

변경 후:

```python
def on_execute(self, msg: Bool):
    if not msg.data or not self.current_waypoints:
        self.get_logger().warn("실행할 웨이포인트가 없습니다")
        return
    if self.current_joint_state is None:
        self.get_logger().warn("joint_state 미수신 -> 실행 보류")
        return
    if self.executing:
        self.get_logger().warn("이미 실행 중")
        return

    # Stage 2/3 가 쓸 표면 스냅된 tool0 waypoints 미리 계산
    densified_tip, tool0_wps, target, n = self._compute_snapped_tool0_waypoints()
    self._stage3_tool0_wps = tool0_wps

    # Stage 1 의 목표: 첫 점 표면 위치 → normal 방향 SAFETY_OFFSET 후퇴
    fixed_q = ee_quat_for_target(target)
    first_tip = densified_tip[0]
    safety_tip = self._offset_along_normal(first_tip, SAFETY_OFFSET)
    safety_tool0 = self._brush_tip_to_tool0(safety_tip)
    safety_tool0.orientation.x = float(fixed_q[0])
    safety_tool0.orientation.y = float(fixed_q[1])
    safety_tool0.orientation.z = float(fixed_q[2])
    safety_tool0.orientation.w = float(fixed_q[3])
    self._safety_tool0_pose = safety_tool0

    # Stage 4 의 목표: 마지막 점 → normal 방향 RETREAT_OFFSET 후퇴
    last_tip = densified_tip[-1]
    retreat_tip = self._offset_along_normal(last_tip, RETREAT_OFFSET)
    retreat_tool0 = self._brush_tip_to_tool0(retreat_tip)
    retreat_tool0.orientation.x = float(fixed_q[0])
    retreat_tool0.orientation.y = float(fixed_q[1])
    retreat_tool0.orientation.z = float(fixed_q[2])
    retreat_tool0.orientation.w = float(fixed_q[3])
    self._retreat_tool0_pose = retreat_tool0

    self.get_logger().info("=" * 60)
    self.get_logger().info("=== STAGE 1: free-space approach (OMPL) ===")
    self.get_logger().info(
        f"safety_tool0=({safety_tool0.position.x:.3f},"
        f"{safety_tool0.position.y:.3f},{safety_tool0.position.z:.3f})")
    self.executing = True
    self.stage1_approach_free()
```

### 6. `approach_to_first`, `_approach_ik_done` 삭제

이 두 함수는 더 이상 사용 안 함. 파일에서 제거.

### 7. Stage 함수들 추가

`on_execute` 바로 아래쯤 적당한 위치에 추가:

```python
# ---- Stage 1: free-space approach via MoveGroup action -------------------
def stage1_approach_free(self):
    if not self.move_action_client.wait_for_server(timeout_sec=5.0):
        self.get_logger().error("MoveGroup action server 없음 (/move_action)")
        self.executing = False
        return

    goal = MoveGroup.Goal()
    goal.request.group_name = PLANNING_GROUP
    rs = RobotState()
    rs.joint_state = self.current_joint_state
    rs.is_diff = False
    goal.request.start_state = rs
    goal.request.goal_constraints = [
        self._make_pose_constraints(self._safety_tool0_pose)
    ]
    goal.request.planner_id = PLANNER_ID
    goal.request.allowed_planning_time = ALLOWED_PLANNING_TIME
    goal.request.num_planning_attempts = PLANNING_ATTEMPTS
    goal.request.max_velocity_scaling_factor = 0.3
    goal.request.max_acceleration_scaling_factor = 0.3

    goal.planning_options.plan_only = True
    goal.planning_options.planning_scene_diff.is_diff = True

    future = self.move_action_client.send_goal_async(goal)
    future.add_done_callback(self._stage1_goal_response)

def _stage1_goal_response(self, future):
    try:
        handle = future.result()
    except Exception as e:
        self.get_logger().error(f"STAGE 1 send_goal 실패: {e}")
        self.executing = False
        return
    if not handle.accepted:
        self.get_logger().error("STAGE 1 goal rejected")
        self.executing = False
        return
    self.get_logger().info("STAGE 1 goal accepted, planning...")
    handle.get_result_async().add_done_callback(self._stage1_result)

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

    traj = result.planned_trajectory
    n_points = len(traj.joint_trajectory.points)
    self.get_logger().info(f"STAGE 1 planning OK: {n_points} 포인트")

    # 0.3x 스케일링 (cartesian 과 동일)
    traj = self._rescale_trajectory(traj, scale=0.3)
    self.execute_trajectory_direct(
        traj, on_complete=self.stage2_approach_linear)


# ---- Stage 2: linear approach to first surface point (cartesian) --------
def stage2_approach_linear(self):
    self.get_logger().info("=== STAGE 2: linear approach (cartesian) ===")
    if not self.cartesian_client.wait_for_service(timeout_sec=5.0):
        self.get_logger().error("/compute_cartesian_path 서비스 없음 (Stage 2)")
        self.executing = False
        return

    target = get_target(self.cfg, self.active_target_name)
    _, n = get_surface_plane(target)

    req = GetCartesianPath.Request()
    req.header.frame_id = BASE_FRAME
    req.header.stamp = self.get_clock().now().to_msg()
    req.group_name = PLANNING_GROUP
    req.link_name = EE_LINK

    rs = RobotState()
    rs.joint_state = self.current_joint_state
    rs.is_diff = False
    req.start_state = rs

    # 목표: 표면 첫 점 (tool0 좌표)
    req.waypoints = [self._stage3_tool0_wps[0]]
    req.max_step = 0.005
    req.jump_threshold = 5.0
    req.avoid_collisions = True
    req.max_velocity_scaling_factor = 0.3
    req.max_acceleration_scaling_factor = 0.3

    # Orientation constraint (Stage 3 와 동일)
    ee_q = ee_quat_for_target(target)
    brush_dir_world = -np.asarray(n, dtype=float)
    free_axis = int(np.argmax(np.abs(brush_dir_world)))
    tol = [0.2, 0.2, 0.2]
    tol[free_axis] = 3.14
    oc = OrientationConstraint()
    oc.header.frame_id = "world"
    oc.link_name = EE_LINK
    oc.orientation.x = float(ee_q[0])
    oc.orientation.y = float(ee_q[1])
    oc.orientation.z = float(ee_q[2])
    oc.orientation.w = float(ee_q[3])
    oc.absolute_x_axis_tolerance = tol[0]
    oc.absolute_y_axis_tolerance = tol[1]
    oc.absolute_z_axis_tolerance = tol[2]
    oc.weight = 1.0
    req.path_constraints = Constraints()
    req.path_constraints.orientation_constraints.append(oc)

    future = self.cartesian_client.call_async(req)
    future.add_done_callback(self._stage2_done)

def _stage2_done(self, future):
    try:
        resp = future.result()
    except Exception as e:
        self.get_logger().error(f"STAGE 2 서비스 실패: {e}")
        self.executing = False
        return
    fraction = resp.fraction
    self.get_logger().info(f"STAGE 2 cartesian: {fraction*100:.1f}%")
    if fraction < 0.95:
        self.get_logger().error(
            f"STAGE 2 fraction {fraction*100:.0f}% < 95% -> 중단")
        self.executing = False
        return
    traj = self._rescale_trajectory(resp.solution, scale=0.3)
    self.execute_trajectory_direct(
        traj, on_complete=self.stage3_main_path)
```

### 8. `plan_cartesian` 의 stage 3 화

기존 `plan_cartesian` 함수의 동작은 그대로 유지하되:

- 함수 이름을 `stage3_main_path` 로 변경 (또는 `plan_cartesian` 유지하되 끝에 stage4 호출)
- 기존 `_cartesian_done` 의 `execute_trajectory_direct(traj)` 호출을 `execute_trajectory_direct(traj, on_complete=self.stage4_retreat)` 로 변경

가장 안전한 방법: 기존 함수 이름 유지하고 `_cartesian_done` 의 마지막 부분만 수정.

기존 `_cartesian_done` 끝:

```python
        traj = self._rescale_trajectory(resp.solution, scale=0.3)
        self.execute_trajectory_direct(traj)
```

변경:

```python
        traj = self._rescale_trajectory(resp.solution, scale=0.3)
        self.execute_trajectory_direct(
            traj, on_complete=self.stage4_retreat)
```

### 9. Stage 4 추가

```python
# ---- Stage 4: linear retreat (cartesian) --------------------------------
def stage4_retreat(self):
    self.get_logger().info("=== STAGE 4: linear retreat (cartesian) ===")
    if not self.cartesian_client.wait_for_service(timeout_sec=5.0):
        self.get_logger().error("/compute_cartesian_path 서비스 없음 (Stage 4)")
        self.executing = False
        return

    target = get_target(self.cfg, self.active_target_name)
    _, n = get_surface_plane(target)

    req = GetCartesianPath.Request()
    req.header.frame_id = BASE_FRAME
    req.header.stamp = self.get_clock().now().to_msg()
    req.group_name = PLANNING_GROUP
    req.link_name = EE_LINK

    rs = RobotState()
    rs.joint_state = self.current_joint_state
    rs.is_diff = False
    req.start_state = rs

    req.waypoints = [self._retreat_tool0_pose]
    req.max_step = 0.005
    req.jump_threshold = 5.0
    req.avoid_collisions = True
    req.max_velocity_scaling_factor = 0.3
    req.max_acceleration_scaling_factor = 0.3

    ee_q = ee_quat_for_target(target)
    brush_dir_world = -np.asarray(n, dtype=float)
    free_axis = int(np.argmax(np.abs(brush_dir_world)))
    tol = [0.2, 0.2, 0.2]
    tol[free_axis] = 3.14
    oc = OrientationConstraint()
    oc.header.frame_id = "world"
    oc.link_name = EE_LINK
    oc.orientation.x = float(ee_q[0])
    oc.orientation.y = float(ee_q[1])
    oc.orientation.z = float(ee_q[2])
    oc.orientation.w = float(ee_q[3])
    oc.absolute_x_axis_tolerance = tol[0]
    oc.absolute_y_axis_tolerance = tol[1]
    oc.absolute_z_axis_tolerance = tol[2]
    oc.weight = 1.0
    req.path_constraints = Constraints()
    req.path_constraints.orientation_constraints.append(oc)

    future = self.cartesian_client.call_async(req)
    future.add_done_callback(self._stage4_done)

def _stage4_done(self, future):
    try:
        resp = future.result()
    except Exception as e:
        self.get_logger().error(f"STAGE 4 서비스 실패: {e}")
        self.executing = False
        return
    fraction = resp.fraction
    self.get_logger().info(f"STAGE 4 cartesian: {fraction*100:.1f}%")
    if fraction < 0.5:
        self.get_logger().warn(
            f"STAGE 4 fraction {fraction*100:.0f}% 낮음 -> 중단 (토치 표면에 남음)")
        self.executing = False
        return
    traj = self._rescale_trajectory(resp.solution, scale=0.3)
    self.execute_trajectory_direct(traj, on_complete=self._all_stages_done)

def _all_stages_done(self):
    self.get_logger().info("=" * 60)
    self.get_logger().info(">>> 모든 stage 완료 (1→2→3→4)")
    self.get_logger().info("=" * 60)
    self.executing = False
```

### 10. `execute_trajectory_direct` 에 `on_complete` 인자 추가

기존:

```python
def execute_trajectory_direct(self, traj):
    """궤적 포인트를 시간에 맞춰 /joint_command 로 퍼블리시."""
    points = traj.joint_trajectory.points
    names = list(traj.joint_trajectory.joint_names)
    if not points:
        self.get_logger().warn("빈 궤적")
        self.executing = False
        return

    self.executing = True
    self.get_logger().info(f"궤적 실행 시작: {len(points)} 포인트")

    def _run():
        ...
        self.get_logger().info(">>> 궤적 실행 완료")
        self.executing = False

    threading.Thread(target=_run, daemon=True).start()
```

변경 (`on_complete` 인자 + flag 처리):

```python
def execute_trajectory_direct(self, traj, on_complete=None):
    """궤적 포인트를 시간에 맞춰 /joint_command 로 퍼블리시.
    on_complete: 실행 끝나고 호출할 callback. 다음 stage 트리거용."""
    points = traj.joint_trajectory.points
    names = list(traj.joint_trajectory.joint_names)
    if not points:
        self.get_logger().warn("빈 궤적")
        if on_complete is None:
            self.executing = False
        return

    self.get_logger().info(f"궤적 실행 시작: {len(points)} 포인트")

    def _run():
        import time
        prev_time = 0.0
        for i, pt in enumerate(points):
            t_sec = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            dt = t_sec - prev_time
            if dt > 0:
                time.sleep(dt)
            prev_time = t_sec

            cmd = JointState()
            cmd.name = names
            cmd.position = list(pt.positions)
            if pt.velocities:
                cmd.velocity = list(pt.velocities)
            self.joint_cmd_pub.publish(cmd)

        self.get_logger().info(">>> 궤적 실행 완료")
        # joint_state 안정화 짧게 대기 (다음 stage 가 current_joint_state 쓰니까)
        time.sleep(0.5)
        if on_complete is not None:
            on_complete()
        else:
            self.executing = False

    threading.Thread(target=_run, daemon=True).start()
```

### 11. 검증

수정 후:

```bash
cd ~/sketch_robot_ws
colcon build --packages-select sketch_control --symlink-install
source install/setup.bash
```

Build error 없어야 함. Import 오류, 변수 미정의 등 잡아주세요.

## 작업 가이드

1. 백업이 있는지 먼저 확인 (`moveit_executor.py.bak_phase2`).
2. 변경사항 1번부터 10번까지 순서대로 적용.
3. 11번 build.
4. Build 성공하면 끝. 실행은 사용자가 별도로 테스트.
5. 변경 요약을 마지막에 출력 (어떤 함수가 추가/수정/삭제됐는지).
