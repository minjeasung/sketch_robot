# Phase 4 — Session 1: RB10 Driver Setup & Robot Info Extraction

세션 날짜: 2026-04-29
환경: 실로봇 연결용 새 컴퓨터 (`snucem`), Ubuntu 24.04, ROS 2 Jazzy
로봇 모델: **Rainbow Robotics RB10-1300 (rb10_1300e_u)**
세션 상태: ✅ 완료 — 로봇 전원 OFF, fake_hardware로만 검증
다음 세션: 포팅 prompt 작성 → moveit_executor.py UR10→RB10 변환

---

## 0. 결론 한 줄

`rbpodo_ros2`(공식 driver)가 Jazzy에서 빌드/실행 모두 성공.
URDF·MoveIt·controller 다 정상 로드. 포팅에 필요한 모든 frame/joint/group 정보 확보.

---

## 1. 셋업 결과

### 1.1 환경
- OS: Ubuntu 24.04.4 LTS (noble)
- ROS distro: Jazzy
- 워크스페이스: `~/rb10_ws/`

### 1.2 설치된 것
- `rbpodo` C++ 라이브러리 v0.16.10 → `/usr/local/lib/librbpodo.a`, `/usr/local/include/rbpodo/`
- `rbpodo_ros2` (5 packages): `rbpodo_msgs`, `rbpodo_description`, `rbpodo_moveit_config`, `rbpodo_hardware`, `rbpodo_bringup`
- ROS 의존성: `ros-jazzy-{moveit, ros2-control, ros2-controllers, robot-state-publisher, joint-state-publisher, rviz2, urdf-launch, xacro, pluginlib, ament-cmake}`

### 1.3 빌드
- `colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --symlink-install`
- 5/5 패키지 성공
- 경고 1건 (무해): `on_init` deprecated in `rbpodo_hardware` (Jazzy backward compatible)

### 1.4 검증 launch 명령
```bash
ros2 launch rbpodo_moveit_config moveit.launch.py \
  model_id:=rb10_1300e_u \
  use_fake_hardware:=true \
  cb_simulation:=false
```
RViz에 RB10 모델 표시됨. MoveIt "You can start planning now!" 출력 확인.

---

## 2. 추출한 로봇 정보 (포팅 핵심 자료)

### 2.1 비교표

| 항목 | UR10 (Phase 3 시뮬) | RB10 (Phase 4 실로봇) |
|---|---|---|
| `PLANNING_GROUP` | `ur_manipulator` | **`mainpulation`** ⚠️ 오타 |
| `BASE_FRAME` | `base_link` | **`link0`** |
| `EE_LINK` | `tool0` | **`tcp`** |
| Robot URDF name | (UR10) | **`rb`** |
| World root | base_link | **`world` → `link0`** (static TF) |
| `JOINT_NAMES` (운동학적 순) | shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3 | **`base, shoulder, elbow, wrist1, wrist2, wrist3`** |
| `joint_states` 발행 순서 | (UR 디폴트) | **알파벳 순**: `base, elbow, shoulder, wrist1, wrist2, wrist3` ⚠️ 다름 |
| Controller | UR 류 | **`joint_trajectory_controller`** + `joint_state_broadcaster` |
| Controller type | FollowJointTrajectory | FollowJointTrajectory (동일) |
| Update rate | (시뮬 가변) | **30 Hz** |
| Named home pose (SRDF) | UR 디폴트 있었음 | **없음 — 직접 정의 필요** |
| Joint limits | UR10 spec | **placeholder ±3.14 rad, vel 3.14, eff 10** ⚠️ |

### 2.2 TF tree (현재 fake_hardware 상태)
```
world
  └── link0     (static, 'fixed_joint')
        └── link1
              └── link2
                    └── link3
                          └── link4
                                └── link5
                                      └── link6
                                            └── tcp
```

### 2.3 SRDF 핵심
```xml
<robot name="rb">
  <group name="mainpulation">              <!-- 오타 그대로 사용 필수 -->
    <chain base_link="link0" tip_link="tcp"/>
  </group>
  <virtual_joint name="fixed_joint" type="fixed"
                 parent_frame="world" child_link="link0"/>
  <!-- group_state 없음 -->
</robot>
```

---

## 3. ⚠️ 다음 세션 시작 전 알아둘 트랩

### TRAP 1 — Planning group 이름 오타 `mainpulation`
공식 SRDF 자체에 박힌 오타. `manipulation`으로 적으면 group not found로 죽음.
포팅 코드에 `PLANNING_GROUP = "mainpulation"`으로 정확히 복사할 것.

### TRAP 2 — Joint state 순서 ≠ URDF 정의 순서
- URDF/MoveIt 내부: `base, shoulder, elbow, wrist1, wrist2, wrist3` (운동학)
- `/joint_states` 토픽: `base, elbow, shoulder, wrist1, wrist2, wrist3` (알파벳)

`joint_state_broadcaster`가 알파벳으로 sort해서 발행. trajectory msg 만들 때 인덱스로 접근하면 어깨↔팔꿈치 swap. 항상 **이름→값 dict 매핑**으로 처리할 것. MoveIt API 거치면 알아서 처리됨.

### TRAP 3 — Joint limits가 placeholder
6 joints 모두 `±3.14 rad, vel 3.14, effort 10` — 진짜 spec 아님. fake_hardware에선 collision/IK 정상이지만 실로봇에서 protective stop 가능. 실로봇 단계에서 데이터시트/teach pendant 실측값으로 URDF override 또는 별도 limits.yaml 필요.

### TRAP 4 — `joint_states.header.frame_id = base_link`
실제 TF에는 `base_link` frame이 없음 (`link0`만 존재). 이 header.frame_id는 호환용 라벨일 뿐. TF lookup용으로 쓰지 말 것 — 항상 `link0`로.

### TRAP 5 — Named home pose 부재
SRDF에 `<group_state>`가 없음. `READY_POSE_JOINTS` 6개 값을 RB10 좌표계에서 **직접** 정해야 함.
정석: 실로봇 켜고 teach pendant로 안전한 자세 → joint angle 6개 읽어서 코드에 박기.
임시: fake_hardware에서 RViz로 IK 풀어서 대략값 확인 → 실로봇에서 검증.

---

## 4. 시뮬→실로봇 포팅 시 변경 포인트 (코드 레벨)

`sketch_robot_ws/src/sketch_control/sketch_control/moveit_executor.py` 기준 수정 항목:

```python
# === 변경 필요 상수 ===
PLANNING_GROUP = "mainpulation"        # 오타 그대로
BASE_FRAME = "link0"                   # was "base_link"
EE_LINK = "tcp"                        # was "tool0"

JOINT_NAMES = [                        # URDF 운동학 순서
    "base", "shoulder", "elbow",
    "wrist1", "wrist2", "wrist3",
]

# === 새로 측정 필요 ===
READY_POSE_JOINTS = [...]              # RB10에서 실측 (TODO)
FIXED_TOOL0_QUAT = (...)               # RB10에서 hand-eye 후 재캘리브 (TODO)

# === 그대로 사용 가능 ===
# 4-stage executor 구조, FollowJointTrajectory action 인터페이스
# (controller 종류 동일하므로 action client 코드 그대로)
```

**그대로 두기**: 4-stage 파이프라인 구조, action client 패턴, planning scene 관리.
**바뀜**: 위 상수 5개 + READY_POSE 캘리브 + tool quat 캘리브.

---

## 5. 다음 세션 액션 플랜

### 우선순위 A — 코드 포팅 prompt 작성
이 문서를 입력 자료로 깨끗한 prompt 만들어 Claude Code에 던지기.
산출물: 포팅된 `moveit_executor.py` (READY_POSE/FIXED_TOOL0_QUAT은 TODO 마킹).

### 우선순위 B — 실로봇 첫 연결
- 컨트롤박스 IP 확인 (teach pendant 네트워크 메뉴)
- `ping <robot_ip>`
- E-stop 작동 확인
- 토치/EoAT 분리 (flange만)
- 주변 1.3m 클리어
- `use_fake_hardware:=false robot_ip:="x.x.x.x"`로 launch
- RViz에서 실로봇 자세가 teach pendant와 일치하는지 확인 (TF 검증)

### 우선순위 C — READY_POSE 실측
- 실로봇 연결 후, teach pendant로 안전한 시작 자세 만들기
- `ros2 topic echo /joint_states --once`로 6개 값 읽기
- ⚠️ joint_states는 알파벳 순이므로 운동학 순서로 재배열 후 코드에 박기

### 우선순위 D (보류) — Joint limits 실값
포팅 동작 검증 후. RB10 데이터시트 또는 teach pendant에서 받기.

### 우선순위 E (별도 트랙) — ZED 셋업
ZED SDK 설치 + `zed_ros2_wrapper` 토픽 확인만. 알고리즘은 hand-eye calib 후 착수.

---

## 6. 안전 메모

- 이번 세션 fake_hardware로만 작업, 로봇 전원 OFF 상태 유지함
- 다음 세션 실로봇 첫 연결 시:
  - 토치/EoAT **분리** 후 flange만으로 첫 모션
  - 첫 명령은 매우 작은 joint 변위(±5°)부터
  - READY_POSE는 UR10 값을 절대 그대로 보내지 말 것 (joint limit + 자세 다름)
  - E-stop 손 닿는 곳

---

## Appendix — 명령 레퍼런스

### Driver launch (fake)
```bash
source /opt/ros/jazzy/setup.bash
source ~/rb10_ws/install/setup.bash
ros2 launch rbpodo_moveit_config moveit.launch.py \
  model_id:=rb10_1300e_u \
  use_fake_hardware:=true \
  cb_simulation:=false
```

### Driver launch (실로봇)
```bash
ros2 launch rbpodo_moveit_config moveit.launch.py \
  model_id:=rb10_1300e_u \
  use_fake_hardware:=false \
  cb_simulation:=false \
  robot_ip:="<RB10_IP>"
```

### 정보 추출 (driver 띄운 채로 별도 터미널)
```bash
# joint state 한 번
ros2 topic echo /joint_states --once

# TF tree
ros2 run tf2_tools view_frames

# Controllers
ros2 control list_controllers

# URDF에서 link/joint 목록
ros2 param get /robot_state_publisher robot_description 2>/dev/null \
  | grep -oP '<link name="\K[^"]+' | sort -u
ros2 param get /robot_state_publisher robot_description 2>/dev/null \
  | grep -oP '<joint name="\K[^"]+'

# SRDF 보기
find ~/rb10_ws/install/rbpodo_moveit_config -name "*.srdf*" -exec cat {} \;
```
