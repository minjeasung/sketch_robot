# Sketch Robot — 프로젝트 개요

새 Claude 세션 시작 시 이 문서를 첨부하면 컨텍스트 즉시 회복.
`journal.txt`, `prompts/` 폴더와 함께 사용.

---

## 1. 프로젝트 정체성

**한 줄**: 스케치 인터페이스로 그린 2D 경로를 카메라 기반 3D 표면 경로로 변환하고, 먼저 페인트 롤러 EOAT 로 표면 추종을 검증한 뒤 용접 토치로 확장하는 시스템.

**현재 구현 전략 (중요):**
- 1차 구현 대상은 **용접 토치가 아니라 페인트 롤러 EOAT**.
- 롤러로 먼저 검증하는 범위:
  - ZED 카메라 기반 작업면 인식
  - sketch pixel → 3D waypoint 변환
  - RB10 의 표면 접근 / 추종 / 후퇴 / READY 복귀
  - EOAT offset, collision, hand-eye calibration, 실로봇 안전 절차
- 용접 구현은 롤러로 위 pipeline 이 안정화된 뒤, EOAT 를 토치로 바꾸고 공정 파라미터를 추가하는 **후속 확장 단계**.
- 따라서 현재 Isaac Sim 의 RB10 환경은 용접 물리 시뮬레이션이 아니라 **롤러 기반 sketch-to-surface path following 검증용 디지털 테스트베드**.

**최종 흐름**:
1. 사용자가 카메라 화면을 본다 (Unity 또는 Python GUI)
2. 화면에서 클릭/드래그로 직선/도형을 그린다
3. 시스템이 그 픽셀 경로를 작업물 표면 위 3D 경로로 변환한다
4. 현재 단계: 로봇이 페인트 롤러를 들고 그 경로를 따라 표면을 추종한다
5. 후속 단계: 동일한 perception/control 구조를 용접 토치 EOAT 로 확장한다

---

## 2. Phase 진행 현황

| Phase | 내용 | 상태 |
|---|---|---|
| 1 | Python sketch_ui (tkinter) + Isaac Sim 검증, MoveIt 2-stage approach | ✓ 완료 |
| 2 | Unity SketchManager.cs 인터페이스로 교체, ROS-TCP-Connector | ✓ 완료 |
| 3 | MoveIt 충돌 회피 안정화 (4-stage + Stage 5 복귀) | ◐ 부분 완료, wall 등록 이슈 보류 |
| 4 | RB10 + ZED + 페인트 롤러 EOAT 통합 (시뮬 검증) | ✓ Session 5 완료 (시뮬 검증) |
| 5 | 실로봇 (snucem) 연결 + 롤러 기반 검증 | ◐ 일감 1+2.3 완료, 2.4 진행, 2.5 script 작성 완료 |

### Phase 4 진행 현황 (2026-05-12 기준)

**완료:**
- 4.1 RB10 driver 셋업 (snucem)
  - minjeasung/rbpodo_ros2 fork (commit 2971eac): base limit ±2π, joint accel 5.0
- 4.2 moveit_executor.py UR10→RB10 포팅 (sketch_robot commit 90e9653)
- 4.3 Stage 5 단독 실로봇 검증 통과 (snucem, commit 7a6f87c)
- 4.0 Isaac Sim에 RB10 URDF import (시뮬 컴 minjea@minjea)
  - scripts/setup_isaac_assets.sh 로 xacro → urdf → mesh 복사 자동화
  - Isaac Sim 5.1 File > Import 로 .usd 생성 (~/sketch_robot_ws/isaac_assets/)

**미완성 (보관):**
- 4.X Stage 1 단독 trigger + jog (commit d7c4d73, 다른 채팅 시도 실패)
  - prompts/2026-05-05_phase4_*.md 참고

**다음:**
- isaac_sim_rb10.py 작성 (UR10 → RB10 포팅)
  - link/joint 이름 매핑: shoulder_pan→base, shoulder_lift→shoulder,
    elbow→elbow, wrist_1→wrist1, wrist_2→wrist2, wrist_3→wrist3, tool0→tcp
  - READY_POSE 갱신 (UR10 값 → RB10 값)
- 시뮬에서 Stage 1~5 검증 (Phase 3 보류 이슈인 wall 등록도 같이 풀기)
- 롤러 EOAT 기준으로 실로봇 Stage 1~4 검증 (ROD_AXIS / ROLLER_LONG_AXIS 결정)
- 용접 토치는 롤러 pipeline 안정화 후 후속 확장으로 전환
- ZED 통합 (외부 고정, eye-to-hand, static TF, RANSAC plane)

**자산:**
- snucem: 실로봇 + driver 검증 완료
- 시뮬 컴 (minjea@minjea): Isaac Sim 5.1 + RB10 USD
  - 위치: ~/sketch_robot_ws/isaac_assets/rb10_1300e_u.usd
  - 재생성: bash ~/sketch_robot_ws/scripts/setup_isaac_assets.sh
- perception/ransac_multiplane.py: ZED 통합 출발점 (synthetic 점군까지 완성)
- minjeasung/rbpodo_ros2 fork: RB10 환경 패치

**환경:**
- 두 컴이 sketch_robot repo 통해 동기화 (origin/main)
- isaac_assets/ 는 .gitignore (재생성 가능)

**다음 세션 시작 시 (시뮬 컴):**
- Isaac Sim 띄우기: ~/sketch_robot_ws/run_isaac_sim.sh (현재는 ur10용 스크립트)
- 또는 빈 sim: `source ~/isaac_env/bin/activate && isaacsim`
- USD reference: ~/sketch_robot_ws/isaac_assets/rb10_1300e_u.usd

**다음 세션 시작 시 (snucem):**
- 로봇 자세 = base +179° 근처가 READY (펜던트로 잡음)
- driver launch: `ros2 launch rbpodo_moveit_config moveit.launch.py model_id:=rb10_1300e_u robot_ip:=10.0.2.7 use_fake_hardware:=false`
- moveit_executor: `ros2 run sketch_control moveit_executor`
- Stage 5 sanity: `ros2 topic pub --once /debug_trigger_stage5 std_msgs/Bool "data: true"`

### Phase 3 보류 이슈
ApplyPlanningScene service 가 success=True 응답하지만 GetPlanningScene 으로 확인 시 `world.collision_objects=[]` (wall 미등록). frame_id 'base_link', 'world', 'World' 모두 동일 증상. ROS2 Jazzy + UR + Isaac Sim + ur_moveit_config 의 누적 셋업 이슈로 추정. 시뮬에서 wall 등록 디버깅 보류, 실로봇 셋업 시 재검토 예정.

→ 현재 단일 점 / 단순 시나리오는 작동. 두 번째 이상 Submit 의 Stage 1 에서 토치가 wall 살짝 통과하는 한계 있음.

---

## 3. 환경

- **Linux**: Ubuntu 24.04, ROS2 Jazzy, Isaac Sim 5.1, RTX 5060
- **Windows**: Unity 6.4 (sketchrobotunity)
- **통신**: ROS-TCP-Connector (147.46.95.52:10000)

---

## 4. 핵심 상수

```
READY_POSE        = [-0.08, -1.6, 1.76, -1.76, -1.9, 3.14]
FIXED_TOOL0_QUAT  = (0.726, 0.668, -0.111, 0.122)  # 벽 작업용 EE
TORCH_LENGTH      = 0.25 m  (tool0 의 +Y 방향)
WALL              = position (1.05, 0, 0.2), size (0.1, 1.5, 1.2)
CAMERA_EYE        = (-1.1, -1.5, 0.8)
CAMERA_TARGET     = (1.0, 0, 0.2)
CAM_W, CAM_H      = 320, 240
```

---

## 5. 코드 지도

### 워크스페이스 루트: `~/sketch_robot_ws/`

```
sketch_robot_ws/
├─ src/sketch_control/
│   ├─ sketch_control/                     ← Python 노드들
│   │   ├─ moveit_executor.py              ★ MoveIt planning + 궤적 실행
│   │   │                                    Phase 3 의 4-stage 파이프라인 + Stage 5
│   │   │                                    OMPL Stage 1, cartesian Stage 2/3/4, MoveGroup Stage 5
│   │   │                                    44 KB, 가장 큰 파일. 핵심.
│   │   ├─ moveit_executor.py.bak_phase2   Phase 2 시점 백업
│   │   ├─ isaac_sim_ur10.py               Isaac Sim 안에서 도는 씬 셋업
│   │   │                                    UR10 + 벽 + 토치 + 카메라 + OmniGraph
│   │   │                                    ROS2 토픽: /joint_states /joint_command /tf /camera/*
│   │   ├─ isaac_sim_ur10.py.bak_pre_depth (depth 카메라 추가 전 백업)
│   │   ├─ sketch_ui.py                    Phase 1 의 tkinter GUI (Phase 2 부터 미사용)
│   │   ├─ weld_visualizer.py              torch_tip TF 구독 → RViz Marker (비드 시각화)
│   │   │                                    signed_dist 로그도 여기서 (충돌 진단용)
│   │   ├─ targets.py                      objects.yaml 로드 + 표면 평면/EE quat 계산
│   │   │                                    moveit_executor + sketch_ui 공유
│   │   └─ joint_calibrator.py             6-슬라이더 tkinter GUI (수동 캘리브레이션 도구)
│   ├─ launch/
│   │   ├─ core.launch.py                  공통: rsp + static_tf + ur_moveit + executor + visualizer
│   │   ├─ phase1_python.launch.py         core + sketch_ui
│   │   ├─ phase2_unity.launch.py          core + ros_tcp_endpoint    ← 현재 사용
│   │   ├─ moveit_with_rsp.launch.py       (MoveIt + rsp 만, 옵션)
│   │   └─ sketch_control.launch.py        (legacy 통합 launch, 신규 사용 X)
│   └─ config/
│       └─ objects.yaml                    작업 대상 정의 (wall 활성, demo_box 비활성)
├─ run_isaac_sim.sh                        Isaac Sim 시작 스크립트
├─ sketch_ur10.py                          (legacy 별도 시작 스크립트 — isaac_sim_ur10.py 와
│                                            구조 다름. 사용 안 하지만 보존)
├─ sketch_ui_standalone.py                 (legacy standalone GUI — 사용 안 함)
├─ journal.txt                             백업/이력 누적 로그
├─ prompts/                                Claude Code 에 던지는 .md 패치 보관소
│   ├─ README.md                           사용법 + 진행 이력
│   ├─ 2026-04-27_phase3_approach_4stage.md
│   ├─ 2026-04-27_phase3_safety_followup.md
│   ├─ 2026-04-27_phase3_ompl_revival.md
│   ├─ 2026-04-27_phase3_stage1_2_tighten.md
│   ├─ 2026-04-27_phase3_torch_collision_axis_fix.md
│   ├─ 2026-04-27_phase3_apply_planning_scene.md
│   └─ 2026-04-27_phase3_ready_pose_routing.md
├─ build/, install/, log/                  colcon 빌드 산물
└─ logs/                                   진단 로그 (tee 출력 보관)
```

### Unity 측: `C:\Users\min67\sketchrobotunity\`
- `Assets/SketchManager.cs` — 클릭 → 픽셀 → 3D world 변환 (광선-평면 교차) + ROS 발행

### 시스템 변경 (root 권한)
- `/opt/ros/jazzy/share/ur_moveit_config/config/ompl_planning.yaml` 수정됨 (OMPL planner 등록)
- `.bak_phase3` 백업 존재

---

## 6. 작동 원리 (큰 그림)

### 발행 흐름
```
[Unity SketchManager.cs] 또는 [Python sketch_ui]
  ↓ 카메라 픽셀 → 광선 → 표면(wall plane) 교차
  ↓ /sketch_waypoints (PoseArray)
  ↓ /sketch_execute (Bool)
[moveit_executor]
  ↓ 4-stage + Stage 5
  ↓ Stage 1: 현재 → 안전점 (OMPL/PTP 자유 경로)
  ↓ Stage 2: 안전점 → 표면 첫 점 (cartesian 5cm 직선)
  ↓ Stage 3: 표면 위 비드 (cartesian path)
  ↓ Stage 4: 마지막 점 → 후퇴점 (cartesian)
  ↓ Stage 5: 후퇴점 → READY_POSE (MoveGroup action)
  ↓ /joint_command (JointState, 직접 publish)
[Isaac Sim (isaac_sim_ur10.py)]
  ↓ joint_command 받아 articulation drive
  ↓ /tf, /joint_states 발행
[weld_visualizer]
  ↓ torch_tip TF 구독
  ↓ wall 표면과의 거리(signed_dist) 계산
  ↓ 접촉 시 비드 marker 누적
  ↓ /weld_beads (MarkerArray)
[RViz]
```

### 좌표계 주의사항
- Isaac Sim 의 root frame: **`World`** (대문자)
- URDF 의 root frame: **`world`** (소문자)
- `core.launch.py` 의 `static_transform_publisher` 가 둘을 잇는다
- moveit_executor 의 `BASE_FRAME = "base_link"` (URDF 의 로봇 base)

### EE orientation (벽 작업)
- objects.yaml 의 `wall.ee_orientation = [0.726, 0.668, -0.111, 0.122]`
- 이 quaternion 으로 tool0 의 local +Y 가 world +X (벽 향함) 가 됨
- 토치는 tool0 의 **+Y 방향으로 25cm** 뻗는다 (수동 캘리브레이션)
- attached collision shape (cylinder) 도 +Y 방향으로 회전해서 등록 (Phase 3 에서 수정됨)

---

## 7. 작업 컨벤션

### 패치 작성 절차
1. claude.ai 세션에서 설계 + 합의
2. claude.ai 가 `.md` 로 패키징 (이 문서와 같은 형식)
3. `~/sketch_robot_ws/prompts/YYYY-MM-DD_phaseN_<task>.md` 로 저장
4. Claude Code (VS Code Linux) 에 다음과 같이 던짐:
   ```
   @prompts/YYYY-MM-DD_phaseN_<task>.md 적용해줘.
   끝나면 변경 요약 + colcon build 결과 알려줘.
   ```
5. 결과 받으면 claude.ai 로 가져가 다음 단계 결정

### Build
```bash
cd ~/sketch_robot_ws
colcon build --packages-select sketch_control --symlink-install
source install/setup.bash
```

### 백업 컨벤션
- 큰 마일스톤마다: `~/sketch_robot_ws_BACKUP_<phase>-<status>_<YYYYMMDD>`
- 파일 단위 작은 백업: `<file>.bak_<tag>` (같은 폴더)

### 실행 절차 (Phase 2 — 현재)
```bash
# 정리
pkill -9 -f isaac_sim
sleep 5
pkill -9 -f "ros2|moveit|rviz|tcp_endpoint|robot_state|world_to|move_group|weld"
sleep 5

# 터미널 1: Isaac Sim
~/sketch_robot_ws/run_isaac_sim.sh

# 터미널 2: launch (로그 저장)
source /opt/ros/jazzy/setup.bash
source ~/sketch_robot_ws/install/setup.bash
ros2 launch sketch_control phase2_unity.launch.py 2>&1 | tee /tmp/<task>.log

# Windows: Unity 실행
```

---

## 8. 새 세션 시작 시 절차

1. 이 PROJECT.md + 최신 `journal.txt` 첨부
2. 필요 시 `sketch_robot_ws.zip` 첨부 (코드 직접 봐야 하는 작업일 때만)
3. 첫 메시지에 다음 명시:
   - 어느 Phase 에서 이어가는지
   - 직전 세션에서 어디까지 됐는지
   - 이번 세션의 목표
4. claude.ai 가 `prompts/` 폴더 + `journal.txt` 확인하면 직전 작업 이력 파악 가능

### 백업 목록 (현재 시점)
- `~/sketch_robot_ws_BACKUP_phase1-stable_20260420`
- `~/sketch_robot_ws_BACKUP_phase2-complete_20260427`
- `~/sketch_robot_ws_BACKUP_phase3-stuck_20260427`
- Unity: `C:\Users\min67\sketchrobotunity_BACKUP_phase2-complete_20260427`

### Phase 4 Session 4 추가 완료 (2026-05-13)

**시뮬 환경 (minjea@minjea):**
- Isaac Sim 5.1 + RB10 시각 환경 (테이블/철판/벽/ㄴ자 막대/카메라/롤러)
- READY_POSE 자세 유지 + ROS2 OmniGraph
- 실로봇과 동일 stack: rbpodo_ros2 fork + JointStateTopicSystem + jtc + MoveIt
- moveit_executor 로 Stage 5 시뮬 검증 통과

**rbpodo_ros2 fork 추가 패치 (commit 1adac25):**
- rb_6dof.ros2_control.xacro: use_isaac_sim 분기
- rb10_1300e_u.urdf.xacro: use_isaac_sim arg
- moveit.launch.py: use_isaac_sim arg + joint_states remap

**시뮬 띄우는 법 (시뮬 컴):**

```bash
# 1. Isaac Sim
source ~/isaac_env/bin/activate
isaacsim --exec ~/sketch_robot_ws/src/sketch_control/sketch_control/isaac_sim_rb10.py
# 그 다음 Play 버튼

# 2. controller_manager + moveit + rviz
ros2 launch rbpodo_moveit_config moveit.launch.py \
    model_id:=rb10_1300e_u use_isaac_sim:=true use_fake_hardware:=false

# 3. moveit_executor
ros2 run sketch_control moveit_executor

# 4. trigger
ros2 topic pub --once /debug_trigger_stage5 std_msgs/Bool "data: true"
```

**환경:**
- ROS_DOMAIN_ID=11 (시뮬 컴, snucem 과 분리)
- ros-jazzy-joint-state-topic-hardware-interface (apt 설치 필요)
- 모든 ROS 패키지 최신 (apt upgrade 후 ABI mismatch 해결)

**다음 (Session 5+):**
- 롤러 EOAT 축 결정 (ROD_AXIS / ROLLER_LONG_AXIS, 시뮬 시각으로)
- Stage 1~4 시뮬 검증
- objects.yaml: wall/table/철판/막대/카메라/롤러 collision 등록
- ZED 통합 시작 (perception/ransac_multiplane.py 활용)
- 실로봇 롤러 마운팅 우선 검증, 이후 용접 토치 마운팅으로 확장

### Phase 4 Session 5 추가 완료 (2026-05-15 ~ 16)

**ZED stereo + perception pipeline 완성 (시뮬 컴):**
- Isaac Sim 의 ZED stereo (left + right) 카메라, 6 ROS topic 발행
  (`/zed/zed_node/left/image_rect_color`, `right/image_rect_color`, 각 `camera_info`,
   `depth/depth_registered`, `point_cloud/cloud_registered`)
- `wall_detector_node`: ZED pointcloud → RANSAC (`segment_plane`, distance crop 2m) →
  `/perception/wall_plane` (PoseStamped, normal=+Z)
- `wall_projector_node`: ZED RGB + wall_plane → 정면 가상 view (homography) →
  `/perception/wall_front_view` (800×800 rgb8)
- `sketch_to_waypoints_node`: 브라우저 sketch (정면 view 픽셀) → world 3D waypoints
  (`/sketch_waypoints` PoseArray)

**브라우저 sketch UI (`web/`) 신설:**
- rosbridge_websocket (`:9090`) 통한 ROS 통신 (roslibjs 1.4.1)
- ZED RGB canvas 위에 sketch overlay (두 canvas 겹침)
- Freehand + Line 그리기, Clear/Undo, ESC 취소
- Wall Front 모드 (벽 정면 가상 view 위에 그림)
- Execute (`/sketch_pixels`) + Run Robot (`/sketch_execute`) 버튼

**moveit_executor 새 perception 흐름 통합:**
- yaml 기반 표면 스냅 비활성 (`_compute_snapped_tcp_waypoints`, `plan_cartesian`) →
  `sketch_to_waypoints` 좌표 그대로 사용
- 작업 영역 0.6m × 0.6m
- EOAT 벽 표면 offset 2cm (안전)
- Stage 2 cartesian threshold 0.85 (95% → 85% 완화)

**검증:** 브라우저 sketch → Stage 1~5 동작, EOAT 벽 뚫지 않음.
자세 상세는 `docs/phase4_session5_perception_sketch_ui.md`.

**주요 commit:** fdafea2 (Session 5 핵심), 6a87315, bd9a3ff.

---

## Phase 5: 실로봇 (snucem) 연결 + 롤러 기반 검증 — 진행 중

**Phase 5 의 기준 EOAT:** 페인트 롤러.
이 Phase 의 목적은 용접 자체 구현이 아니라, 롤러로 카메라-스케치-3D 변환-로봇 표면 추종 pipeline 을 실로봇에서 안전하게 검증하는 것. 용접 토치 전환은 이 흐름이 안정화된 뒤 진행한다.

**일감 1 — 시뮬 ZED 통합 (옵션 C, 2026-05-21):** ✅
- 시도 A (zed-isaac-sim sl.sensor.camera streamer + zed-ros2-wrapper sim_mode):
  - SlCameraStreamer (UDP/IPC) 가 wrapper 의 stream subscribe 와 silent fail.
  - 5+ 라운드 디버깅 (cameraPrim relationship, transportLayerMode, IMU schema,
    Helper init 순서) 후에도 wrapper / ZED Depth Viewer 둘 다 stream 수신 X.
  - IMU 등록 (kit command `IsaacSensorCreateImuSensor`), FixedJoint anchoring,
    ball head + bolt mount (sim-to-real visual fidelity) 모두 완성.
- 시도 C (옵션 C): Isaac Sim **native ROS2 Camera Helper** 직접 사용 (zed-isaac-sim
  extension 우회).
  - `isaacsim.ros2.bridge.ROS2CameraHelper` + `IsaacCreateRenderProduct` 로
    ZED_X.usdc 의 CameraLeft / CameraRight prim 으로부터 직접 발행.
  - `IsaacReadIMU` → `ROS2PublishImu` 로 IMU 도 wrapper 동일 topic 발행.
  - ZED_X.usdc reference 와 mount geometry (Seg1/Seg2/ball/bolt), FixedJoint
    anchoring 은 시각 fidelity 위해 그대로 유지.
- 발행 topic (실 ZED ROS2 wrapper 와 1:1):
  - `/zed/zed_node/rgb/color/rect/image` (+ `_right`)
  - `/zed/zed_node/rgb/color/rect/camera_info` (+ `_right`)
  - `/zed/zed_node/depth/depth_registered` (+ `camera_info`, ground-truth 32FC1)
  - `/zed/zed_node/imu/data`
- 해상도: 1280×720 @ 30 Hz (ZED X HD720 와 일치).
- Frame ID: `zed_left_camera_frame_optical`, `zed_right_camera_frame_optical`,
  `zed_imu_link`. (sim USD prim 명 ↔ ZED frame 변환은 launch-side static TF
  로 보강 예정 — 일감 2 에 묶음.)

**학회 contribution 노트:**
zed-isaac-sim 의 IPC channel silent fail 확인 (UDP listen 까지는 동작, 그러나
wrapper 의 stream subscribe 가 timeout). wrapper 호환성 위해 Isaac Sim native
ROS2 Camera Helper 로 우회 — sim/real 의 interface (topic, frame, intrinsic)
**동일 유지로 perception 코드 100% 재사용 보장**.

**일감 2 — Hand-eye calibration (2026-05-21 진행 중):**
- 2.1 자산 준비: AprilTag PNG (`isaac_assets/apriltag/tag36_11_00000_large.png`,
  500×500 1-bit grayscale, tag36h11 ID 0). URDF (rb10_1300e_u) 는 sim/real 동일
  `~/sketch_robot_ws/isaac_assets/rb10_1300e_u.urdf` ↔
  `~/rb10_ws/src/rbpodo_ros2/rbpodo_description/robots/rb10_1300e_u.urdf`.
- 2.2 *(plan — claude.ai)*
- **2.3 ✅ Sim 환경 수정** (isaac_sim_rb10.py):
  - 기존 1×0.05×1m 흰 박스 wall 을 2.0×0.02×1.5m 로 확장 (front_y=-0.80).
  - 벽 위 0.5×0.4m 노란 마스킹 테이프 outline (4 strip, tape_w=0.02m, RGB=(1.0,0.85,0)).
  - AprilTag plane (0.08m 정사각형) 을 TCP local +Z 6cm 에 부착.
    UsdGeom.Mesh quad + UsdShade UsdPreviewSurface + UsdUVTexture (sourceColorSpace=raw)
    로 PNG 직접 매핑 → apriltag_ros 가 검정/흰 marker 인식.
  - Ground truth dump: `~/sketch_robot_ws/ground_truth.json`
    (wall pose + size, work_area 4 corner world coord, camera optical world pose,
    robot base world pose, apriltag tcp local pose, topic/frame names).
  - 학회 contribution baseline — calibration script 가 5 method (Tsai/Park/Horaud/
    Andreff/Daniilidis) 별 sim ground truth vs estimate 오차 정량 비교.
- **2.4 진행** apriltag_ros detector launch + detection 확인.
  - AprilTag prim: TCP local +Z 6cm, identity 회전 (link6 박힘 회피).
  - Robot pose 분리: `WORK_POSE` (roller -Y, wall 향함, 롤러 작업 시) vs `CALIB_POSE`
    (TCP world-X +90° rotation → roller world+Z, AprilTag world+Y camera face-on).
  - `READY_POSE_DICT = CALIB_POSE` 로 calibration phase 시작. 일감 2.5 완료 후
    한 줄 `READY_POSE_DICT = WORK_POSE` 변경으로 롤러 작업 pose 복귀.
  - `CALIB_DELTA = (wrist_name, ±1)` constant — 4 candidate (wrist1±π/2,
    wrist2±π/2) 시각 검증으로 정답 결정. URDF wrist 축 wrist1=(0,1,0),
    wrist2=(0,0,1), wrist3=(0,1,0) — 직접 IK 없이 empirical fix.
- **2.5 진행** Calibration script — eye-to-hand.
  - 신규: `src/sketch_control/sketch_control/calibration_handeye.py`
    (`ros2 run sketch_control calibration_handeye` 또는 `python3 ...py` 둘 다 OK).
  - 흐름: `/joint_states` 의 현재 pose 를 base → wrist1/2/3 axis-aligned ±0.10/±0.20
    (12 pose) + random multi-wrist variation (~8 pose) = 20 pose 생성 →
    각 pose 에서 `/joint_command` (JointState) 발행 → settling 1.5s + 10-sample
    corner average → `cv2.solvePnP(IPPE_SQUARE)` 로 cam→tag pose 산출 →
    `/tf` link0↔tcp lookup 으로 base→tcp FK 수집 → `cv2.calibrateHandEye(DANIILIDIS)`
    로 `T_cam→base` 단일 추정 (5 method 비교 X).
  - Sanity check: ground_truth.json 의 `camera_optical_world_pose` ×
    `inv(robot_base_world_pose)` 와 비교 → translation_error_mm + rotation_error_deg.
    `< 1mm` 면 ✅, `1~5mm` borderline, `>5mm` setup fail.
  - 출력: `~/sketch_robot_ws/calibration_results.json` + console 표 +
    `static_transform_publisher` 명령 한 줄 출력 (TF tree 등록용).
  - ground_truth.json 의 `robot_base_world_pose` 가 null 일 수 있는 케이스
    (ARTICULATION_PATH 가 비-Xformable) — isaac_sim_rb10.py 수정으로 RB10_PRIM_PATH
    fallback 추가 (실제 -90° Z 회전 포함).

**남은 일감:**
- 일감 3: 시뮬 RANSAC 평면 인식 → `/perception/wall_plane` 재검증 (native depth 기반)
- 일감 4: snucem 에 sketch_robot_ws 복제 + 실 ZED 2i 연결 + 실 환경 측정
- 일감 5: 시뮬에서 검증된 롤러 흐름 (wall_detector → wall_projector → sketch UI →
  sketch_to_waypoints → moveit_executor) 실로봇 적용
- 일감 6: 안전 검증 — 속도 제한 (0.1×), 단계적 검증 (Stage 1 단독 → 1~2 → 전체)

**ROS_DOMAIN_ID 분리:** 시뮬 컴 = 11, snucem = 다른 값 (충돌 방지).

**핵심 파일 (옵션 C):**
- `src/sketch_control/sketch_control/isaac_sim_rb10.py` (line ~580 부터
  `ZedROS2Graph` OG 그래프)
- `zed-isaac-sim/` 은 ZED_X.usdc 시각 자산용으로만 유지 (streamer 의존성 끊김)

---

## 시뮬 검증 실행 절차 (터미널 4개)

새 터미널마다 ROS 환경 source 필요. ROS_DOMAIN_ID=11 이 `~/.bashrc` 에 박혀있음 (시뮬 컴).

### 터미널 1 — Isaac Sim

```bash
source ~/isaac_env/bin/activate
isaacsim --exec ~/sketch_robot_ws/src/sketch_control/sketch_control/isaac_sim_rb10.py
```

Isaac Sim 창 뜨면 **▶ Play** 버튼 누름.

### 터미널 2 — controller_manager + MoveIt + RViz

```bash
source /opt/ros/jazzy/setup.bash
source ~/rb10_ws/install/setup.bash

ros2 launch rbpodo_moveit_config moveit.launch.py \
    model_id:=rb10_1300e_u use_isaac_sim:=true use_fake_hardware:=false
```

기대: segfault 없이 controller_manager activate, jsb + jtc active, RViz 창 뜸.

### 터미널 3 — moveit_executor

```bash
source /opt/ros/jazzy/setup.bash
source ~/rb10_ws/install/setup.bash
source ~/sketch_robot_ws/install/setup.bash

ros2 run sketch_control moveit_executor
```

기대: "MoveIt Executor 노드 시작" + PlanningScene apply + [TCP_NOW] 주기 발행.

### 터미널 4 — trigger (검증마다 새로)

```bash
source /opt/ros/jazzy/setup.bash

# 다른 자세로 옮기기 (Stage 5 가 의미 있게 동작하도록)
ros2 topic pub --once /joint_trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{
  joint_names: ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3'],
  points: [{positions: [0.5, -0.9343, 2.4247, -1.6293, 1.5676, 0.0], time_from_start: {sec: 3, nanosec: 0}}]
}"

# Stage 5 trigger (READY_POSE 로 복귀)
ros2 topic pub --once /debug_trigger_stage5 std_msgs/Bool "data: true"
```

### 종료 순서

터미널 3 → 2 → 1 (Ctrl+C). Isaac Sim 마지막. 처음부터 다시 띄울 때도 같은 순서로.

### 흔한 트러블슈팅

- **/joint_command Subscription count: 0** → Isaac Sim 안 떠있음 (또는 ▶ Play 안 누름)
- **/joint_states Publisher count: 2** → moveit.launch.py 가 옛 버전 (remap 누락). rbpodo_ros2 rebuild 필요
- **Trajectory goal rejected** → start state 와 trajectory 첫 point 가 너무 다름. 또는 jtc 아직 inactive
- **segfault on write** → ros 패키지 버전 mismatch. `sudo apt upgrade` 후 재시도
- **ros2 control list 에서 다른 컴 노드 보임** → ROS_DOMAIN_ID 미확인. `echo $ROS_DOMAIN_ID` 가 11 인지
