# Robot Computer Handoff

이 문서는 시뮬레이션 PC에서 하던 대화를 로봇 PC에서 새 Codex 세션으로 이어가기 위한 인수인계 메모다.

## 현재 원격 상태

- Repository: `https://github.com/minjeasung/sketch_robot`
- Branch: `main`
- Latest pushed commit: `1629636 Add D405 EOAT refinement and real robot backend`
- `zed-isaac-sim` 내부 캐시/생성물 변경은 커밋하지 않았다.

## 현재 구현 요약

- RB10 MoveIt robot description에 EOAT를 고정 링크로 포함했다.
- TCP에는 `AFT200 -> RR-00A_B no-camera EOAT -> D405` 순서로 붙는다.
- AprilTag 기반 흐름은 현재 사용하지 않는다.
- ZED는 전체 scene/target/work-area 후보 인식에 사용한다.
- 사용자가 웹 UI에서 `Set Target`, `Set Work Area`, `Send Path` 순서로 의도를 준다.
- `Set Work Area` 이후 D405가 작업영역 근처를 prescan/refinement 해서 작업영역 평면/거리를 보정한다.
- MoveIt collision에는 robot/EOAT, table/camera mount/static objects, dynamic obstacles, active work-area surface가 들어간다.
- 시뮬레이션 제어 토픽은 `/isaac_joint_command`로 분리했다.
- 실제 로봇 제어는 `/joint_trajectory_controller/follow_joint_trajectory` action을 사용해야 한다.

## 로봇 PC에서 업데이트

```bash
cd ~/sketch_robot_ws
git pull origin main

source /opt/ros/jazzy/setup.bash
source ~/rb10_ws/install/setup.bash

colcon build --packages-select eoat_description sketch_control
source ~/sketch_robot_ws/install/setup.bash
```

## 실로봇 실행 시 필수 차이

시뮬레이션과 다르게 아래 값이 중요하다.

```bash
use_isaac_sim:=false
use_sim_time:=false
execution_backend:=follow_joint_trajectory
```

`moveit_executor` 로그 첫 부분에 반드시 아래처럼 떠야 한다.

```text
execution_backend=follow_joint_trajectory
```

그렇지 않으면 실로봇이 아니라 시뮬레이션용 joint command backend로 실행된 것이다.

## 터미널별 실행

### Terminal 1: MoveIt full + RViz + 실제 RB10

```bash
source /opt/ros/jazzy/setup.bash
source ~/rb10_ws/install/setup.bash
source ~/sketch_robot_ws/install/setup.bash

ros2 launch sketch_control rb10_moveit_full.launch.py \
  robot_ip:=10.0.2.7 \
  use_isaac_sim:=false \
  use_sim_time:=false
```

### Terminal 2: perception/sketch

실제 ZED/D405 driver가 point cloud를 발행한다고 가정한다. 그래서 simulation depth 변환 노드는 끈다.

```bash
source /opt/ros/jazzy/setup.bash
source ~/rb10_ws/install/setup.bash
source ~/sketch_robot_ws/install/setup.bash

ros2 launch sketch_control rb10_perception_sketch.launch.py \
  use_sim_depth_pointcloud:=false \
  use_sim_d405_depth_pointcloud:=false \
  d405_cloud_topic:=/d405/d405/depth/color/points
```

### Terminal 3: moveit executor

```bash
source /opt/ros/jazzy/setup.bash
source ~/rb10_ws/install/setup.bash
source ~/sketch_robot_ws/install/setup.bash

ros2 run sketch_control moveit_executor --ros-args \
  -p use_sim_time:=false \
  -p execution_backend:=follow_joint_trajectory
```

### Terminal 4: rosbridge

```bash
source /opt/ros/jazzy/setup.bash
source ~/rb10_ws/install/setup.bash
source ~/sketch_robot_ws/install/setup.bash

ros2 launch rosbridge_server rosbridge_websocket_launch.xml port:=9090
```

### Terminal 5: web UI

```bash
cd ~/sketch_robot_ws/web
python3 -m http.server 8000
```

Open:

```text
http://localhost:8000
```

## 실행 전 확인

```bash
ros2 action list | grep follow_joint_trajectory
ros2 control list_controllers
ros2 topic echo /joint_states --once
ros2 topic info /d405/d405/depth/color/points -v
ros2 topic info /zed/zed_node/point_cloud/cloud_registered -v
```

D405 point cloud topic 이름이 다르면 Terminal 2의 `d405_cloud_topic:=...` 값을 실제 topic으로 바꾼다.

## 안전 체크

- 첫 실로봇 테스트는 경로를 짧게 그린다.
- E-stop과 teach pendant 정지 버튼을 손 닿는 곳에 둔다.
- `Set Work Area` 뒤 D405 prescan이 이상한 큰 관절 우회 동작을 만들면 즉시 중지한다.
- FT sensor 영점은 접촉 전에 잡혀야 한다. 현재 코드는 sketch 시작 전 자동 zero 흐름을 포함한다.
- 작업면 접촉은 Stage 2/3에서만 허용한다. Stage 1 접근과 Stage 4 이탈에서는 작업면과 안전거리를 유지해야 한다.

## 새 Codex 세션에 붙여넣을 요약 프롬프트

```text
나는 sketch_robot_ws를 실제 RB10 로봇컴에서 실행하려고 한다.
repo는 https://github.com/minjeasung/sketch_robot, main 최신 커밋은
1629636 Add D405 EOAT refinement and real robot backend 이다.

현재 시스템은 RB10 + AFT200 force sensor + RR-00A_B roller EOAT + D405 + ZED이다.
ZED로 scene/target/work-area를 보고, 사용자가 web UI에서 Set Target, Set Work Area,
Send Path를 누른다. Set Work Area 후 D405 prescan/refinement로 작업영역 평면/거리를 보정한다.
MoveIt은 robot/EOAT 및 장애물 collision을 관리한다.

시뮬레이션에서는 /isaac_joint_command를 쓰지만, 실제 로봇에서는 반드시
moveit_executor를 execution_backend:=follow_joint_trajectory 로 실행해야 한다.
MoveIt launch도 use_isaac_sim:=false, use_sim_time:=false 로 실행해야 한다.

docs/ROBOT_COM_HANDOFF.md를 읽고, 실제 로봇컴에서 안전하게 실행/디버깅을 이어가자.
```
