# Phase 4 Session 4-cont — Joint-space jog 디버그 트리거

## 배경

직전 시도에서 `/debug_trigger_stage1` (Cartesian goal + OMPL) 가 swooping 경로 생성 → 캐비닛 윗면 임팩트 → RB10 자체 안전정지 (controller deactivate, t=1.05s).

근본 원인 두 개:
1. Planning scene 에 wall 만 등록, 주변 환경 (캐비닛, 책상, 옆 장비) 미등록 → OMPL 자유공간으로 인식
2. Cartesian goal + READY 의 wrist-up orientation 강제 유지 → IK 해 공간 좁아 OMPL 이 우회 path 선택

근본 해결 = 환경 인지 (ZED) 도입이지만, ZED 통합 전에 **motion pipeline 자체** 가 정상인지 검증해두지 않으면 ZED 디버깅 마지막 단계에서 다중 미지수 합산. 그래서 사이에 "거의 안 움직이는 jog" 한 번 통과시킨 후 ZED 로 넘어감.

## 설계

새 디버그 trigger `/debug_trigger_jog` — 기존 `/debug_trigger_stage1`, `/debug_trigger_stage5` 와 분리. **OMPL / MoveIt planner / IK 모두 우회**, JointTrajectory 직접 빌드 → `execute_trajectory_direct` 호출.

동작:
1. `self.current_joint_state` 읽음
2. 한 joint (default: wrist3, index 5) 에 작은 delta (default: 0.05 rad ≈ 2.9°) 적용한 target joints 계산
3. 50점 보간 JointTrajectory 빌드 (10초 duration, 양 끝 v=0)
4. RobotTrajectory 로 감싸 `execute_trajectory_direct` 호출
5. on_complete = chain 차단 (Stage 2 진입 없음)

검증 의미:
- ROS → moveit_executor → controller (FollowJointTrajectory) → 실로봇 신호 경로 정상 작동
- joint_command publish + execute_trajectory_direct 동작
- 펜던트 override 가 ROS trajectory 에 적용되는지 (펜던트 100% / 30% 두 번 돌려서 시간 차 확인)
- trajectory 완료 콜백이 정상 호출되는지

**왜 wrist3 인가**: 가장 끝 joint, 회전만 함, 다른 link 위치 안 변함 → 충돌 위험 본질적으로 0. 회전 중심이 EE 자체라 어느 방향으로도 추가로 안 빠짐. 시각적으로 EE 가 살짝 도는 것만 보이면 통과.

---

## 변경 사항

| 파일 | 변경 |
|---|---|
| `src/sketch_control/sketch_control/moveit_executor.py` | 상수 4개 + subscriber 1개 + 콜백 2개 추가 |

다른 파일 (setup.py 포함) 변경 없음.

---

## 1. `moveit_executor.py` 변경

### 1.1 상수 추가

기존 `DEBUG_STAGE1_OFFSET` 상수 영역 부근에 추가:

```python
# Joint-space jog 디버그용. motion pipeline 단독 검증.
# 한 joint 만 작게 움직여 OMPL / IK / Cartesian goal 의존성 모두 우회.
JOG_JOINT_INDEX = 5      # wrist3 (가장 국소적, 충돌 위험 최소)
JOG_DELTA_RAD = 0.05     # ≈ 2.9°. 시각적으로 보이지만 무시할 수준
JOG_DURATION_SEC = 10.0  # 10초에 걸쳐 움직임 → 인간 반응 충분
JOG_NUM_POINTS = 50      # 0.2초 간격 보간
```

### 1.2 Subscriber 추가

`__init__` 안, 기존 stage1/stage5 trigger subscriber 옆에 추가:

```python
self.create_subscription(
    Bool, "/debug_trigger_jog", self.on_debug_trigger_jog, 10)
```

### 1.3 콜백 2개 추가

기존 `on_debug_trigger_stage1` 또는 `on_debug_trigger_stage5` 옆에 추가:

```python
def on_debug_trigger_jog(self, msg: Bool):
    """
    매우 작은 joint-space 동작으로 motion pipeline 검증.
    
    OMPL / IK / Cartesian goal / planning scene 의존성 모두 없음.
    한 joint (wrist3) 에 0.05 rad delta 를 10초에 걸쳐 적용.
    Trajectory 직접 빌드 → execute_trajectory_direct 호출.
    """
    if self.executing:
        self.get_logger().warn(
            "이미 실행 중 -> /debug_trigger_jog 무시")
        return

    self.get_logger().info(
        "[DEBUG] /debug_trigger_jog 수신 -> joint-space jog 단독 실행")

    if self.current_joint_state is None or \
       len(self.current_joint_state.position) == 0:
        self.get_logger().error(
            "current_joint_state 미수신. /joint_states 흐름 확인.")
        return

    n = len(self.current_joint_state.position)
    if JOG_JOINT_INDEX >= n:
        self.get_logger().error(
            f"JOG_JOINT_INDEX={JOG_JOINT_INDEX} 가 joint 수({n}) 초과")
        return

    current = list(self.current_joint_state.position)
    target = list(current)
    target[JOG_JOINT_INDEX] = current[JOG_JOINT_INDEX] + JOG_DELTA_RAD

    target_joint_name = self.current_joint_state.name[JOG_JOINT_INDEX]
    self.get_logger().info(
        f"[DEBUG] jog target joint: {target_joint_name} "
        f"({current[JOG_JOINT_INDEX]:.4f} -> {target[JOG_JOINT_INDEX]:.4f}, "
        f"delta={JOG_DELTA_RAD:+.4f} rad)")
    self.get_logger().info(
        f"[DEBUG] jog duration: {JOG_DURATION_SEC}s, "
        f"points: {JOG_NUM_POINTS}, "
        f"평균 각속도: {JOG_DELTA_RAD / JOG_DURATION_SEC:.4f} rad/s "
        f"(≈ {(JOG_DELTA_RAD / JOG_DURATION_SEC) * 57.3:.2f}°/s)")

    # 2점 + 사이 선형 보간 trajectory 빌드
    traj = JointTrajectory()
    traj.joint_names = list(self.current_joint_state.name)

    avg_v = JOG_DELTA_RAD / JOG_DURATION_SEC
    for i in range(JOG_NUM_POINTS):
        alpha = i / (JOG_NUM_POINTS - 1)  # 0.0 ~ 1.0
        point = JointTrajectoryPoint()
        point.positions = [
            current[j] + alpha * (target[j] - current[j])
            for j in range(n)
        ]
        # 양 끝점 정지, 중간은 일정 속도
        if i == 0 or i == JOG_NUM_POINTS - 1:
            point.velocities = [0.0] * n
        else:
            point.velocities = [0.0] * n
            point.velocities[JOG_JOINT_INDEX] = avg_v
        t = alpha * JOG_DURATION_SEC
        point.time_from_start.sec = int(t)
        point.time_from_start.nanosec = int((t - int(t)) * 1e9)
        traj.points.append(point)

    # RobotTrajectory 로 감싸 실행
    rt = RobotTrajectory()
    rt.joint_trajectory = traj

    self.executing = True
    try:
        # ★ execute_trajectory_direct 의 정확한 시그니처는 기존 호출처 (예: _stage1_result) 따라가기.
        #   on_complete 인자 위치/이름이 다르면 거기 맞춰서.
        self.execute_trajectory_direct(rt, on_complete=self._jog_done)
        self.get_logger().info("[DEBUG] jog trajectory sent.")
    except Exception as e:
        self.get_logger().error(f"jog trajectory 전송 실패: {e}")
        self.executing = False


def _jog_done(self):
    """Jog 완료 콜백 (Stage chain 진입 없음, executing 플래그만 해제)."""
    self.get_logger().info("[DEBUG] jog 단독 실행 완료")
    self.executing = False
```

**Claude Code 에 부탁**:
- 필요한 import 가 모듈 상단에 없으면 추가:
  - `from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint`
  - `from moveit_msgs.msg import RobotTrajectory`
  - 이 둘은 이미 다른 곳에서 쓰고 있을 가능성 높음 — 있으면 무시.
- `self.executing` / `self.current_joint_state` / `self.execute_trajectory_direct` 의 정확한 식별자는 기존 코드 (특히 `on_debug_trigger_stage5` / `_stage1_result`) 따라가기.
- `_jog_done` 의 시그니처 — `_debug_stage1_done` 이 어떤 인자 받는지 보고 동일하게. 인자 없는 콜백이면 위 코드 그대로, 인자 있으면 (예: trajectory 결과) 거기 맞춰서.

---

## 2. Build

```bash
cd ~/sketch_robot_ws
colcon build --packages-select sketch_control --symlink-install
source install/setup.bash
```

---

## 3. 실행 (실로봇)

### 3.0 사전 준비

- 로봇 fault 클리어 + 정상 모드 확인 (이전 사고 후 복구 상태)
- 펜던트로 READY_POSE 도달 (base ≈ +179°)
- E-stop 손 닿는 위치
- 펜던트 속도 30% (그러나 이 jog 은 trajectory 자체가 매우 느려서 펜던트 override 와 무관하게 안전)

### 3.1 5터미널 절차

각 터미널 코드블록 통째로 복붙. source 3줄 모두 동일.

**터미널 1 — MoveIt + 드라이버**

```bash
source /opt/ros/jazzy/setup.bash
source ~/rb10_ws/install/setup.bash
source ~/sketch_robot_ws/install/setup.bash
ros2 launch rbpodo_moveit_config moveit.launch.py model_id:=rb10_1300e_u robot_ip:=10.0.2.7 use_fake_hardware:=false 2>&1 | tee /tmp/phase4_s4_jog_moveit.log
```

**터미널 2 — moveit_executor**

```bash
source /opt/ros/jazzy/setup.bash
source ~/rb10_ws/install/setup.bash
source ~/sketch_robot_ws/install/setup.bash
ros2 run sketch_control moveit_executor 2>&1 | tee /tmp/phase4_s4_jog_executor.log
```

**터미널 3 — subscriber 등록 확인**

```bash
source /opt/ros/jazzy/setup.bash
source ~/rb10_ws/install/setup.bash
source ~/sketch_robot_ws/install/setup.bash
ros2 topic info /debug_trigger_jog --verbose
```

→ `Subscription count: 1` 보이면 OK.

**터미널 4 — joint_states sanity (선택, jog 직전 1번)**

```bash
source /opt/ros/jazzy/setup.bash
source ~/rb10_ws/install/setup.bash
source ~/sketch_robot_ws/install/setup.bash
ros2 topic echo /joint_states --once
```

→ 6개 joint 의 name + position 출력. position 값들이 READY_POSE_JOINTS 값 근처면 READY 상태 정상 확인. wrist3 (또는 마지막 joint) 가 거의 0 이어야 함.

**⚠ TRIGGER — 다음 명령 실행 시 EE 가 매우 천천히 ~3° 회전**

펜던트 속도 30% / 손 E-stop 위 / 마음의 준비.

**터미널 5 — TRIGGER**

```bash
source /opt/ros/jazzy/setup.bash
source ~/rb10_ws/install/setup.bash
source ~/sketch_robot_ws/install/setup.bash
ros2 topic pub --once /debug_trigger_jog std_msgs/Bool "{data: true}"
```

→ 터미널 2 에 `[DEBUG] /debug_trigger_jog 수신` 부터 `[DEBUG] jog 단독 실행 완료` 까지 출력.
→ 실로봇은 EE (wrist3 회전) 가 10초에 걸쳐 약 3° 도는 게 보임. 다른 부위는 안 움직임.

### 3.2 (선택) 펜던트 override 검증

위 trigger 이 한 번 정상 통과한 후, 같은 trigger 를 다음 조건들로 다시 실행:

- 펜던트 100% — 동일 시간 (10초) 걸리는지 확인
- 펜던트 10% — 동일 시간 vs 10배 길어지는지 (≈ 100초) 확인

같으면 → 펜던트 override 가 ROS trajectory 에 안 먹힘 (속도 통제는 코드 측 rescale + joint_limits 로만 가능). 다르면 → 먹힘 (다음부터 펜던트도 안전 layer 로 활용 가능).

각 trigger 사이엔 펜던트로 READY 로 복귀하거나, jog 한 번에 wrist3 가 +0.05 rad 옮겨졌으니 그대로 다음 trigger 누르면 또 +0.05 rad 더해짐 (누적 ≈ 0.15 rad ≈ 8.6°까지는 무해).

---

## 4. 검증 포인트

성공 조건:

- 터미널 2 에 `[DEBUG] /debug_trigger_jog 수신` ~ `[DEBUG] jog 단독 실행 완료` 출력 정상
- 실로봇 EE 가 약 10초간 천천히 회전 (육안 관찰)
- 다른 joint / link 는 움직이지 않음
- 회전 끝난 후 정지, controller deactivate 없음 (트럭 종료 없음)
- TCP_NOW heartbeat 의 quat 값이 회전 전후로 약간 변경됨 (wrist3 변화 반영)

실패 조건 / 디버깅:

- trigger 보냈는데 trigger 수신 로그 안 뜸 → subscriber 미등록 (터미널 3 에서 확인)
- "current_joint_state 미수신" 에러 → /joint_states 흐름 끊김 (rbpodo 드라이버 / robot_state_publisher 점검)
- "Trajectory 실행 에러" 또는 controller 거부 → trajectory 형식 문제. 로그에서 정확한 에러 메시지 확인 후 trajectory 빌드 부분 재검토 (특히 joint_names 순서, time_from_start)
- 로봇이 갑자기 빠르게 움직이거나 다른 방향 → **즉시 E-stop**. 코드 어딘가 다른 trigger 가 동시에 들어갔거나 trajectory 가 의도와 다르게 생성됐을 가능성. 로그 확보 후 보고

---

## 5. 작업 완료 보고 항목

다음 알려주세요:

1. 변경된 파일 목록 + colcon build 결과
2. ★ 4곳 (executing 플래그명, current_joint_state 속성명, execute_trajectory_direct 시그니처, _jog_done 인자) 어떻게 매칭했는지
3. 실행 시 터미널 2 의 trigger 이벤트 로그 (`/debug_trigger_jog 수신` ~ `jog 단독 실행 완료`)
4. 실로봇 거동 인상 — 정말 느리고 작은 회전이었는지, EE 만 도는 게 보였는지
5. (선택) 펜던트 override 검증 결과 — 100% vs 10% 시 동작 시간 차이

---

## 6. 다음 세션 계획

이 jog 통과하면 motion pipeline 변수 제거됨. ZED 통합 본격 시작:

- Session 5: ZED 마운팅 + SDK + ROS2 wrapper + static TF + 점군 받기 + RANSAC plane 분리
- Session 6: ZED 점군 → octomap → moveit planning scene + self-segmentation
- Session 7+: planning scene 완성된 상태에서 Cartesian Stage 1 재시도, 충돌 회피 검증
