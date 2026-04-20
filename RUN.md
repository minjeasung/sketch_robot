# 실행 방법

## 터미널 1 — Isaac Sim

ROS2 환경 변수를 자동으로 정리하는 스크립트를 사용하세요.

```bash
~/sketch_robot_ws/run_isaac_sim.sh
```

Isaac Sim 창이 뜨면 스페이스바(또는 ▶) 눌러서 play.

## 터미널 2 — ROS2 노드들 (robot_state_publisher + MoveIt + sketch_ui + moveit_executor)

```bash
source /opt/ros/jazzy/setup.bash
source ~/sketch_robot_ws/install/setup.bash
ros2 launch sketch_control sketch_control.launch.py
```

## 검증 (터미널 3)

```bash
source /opt/ros/jazzy/setup.bash

# URDF 확인
ros2 topic echo /robot_description --once | head -5

# 토픽 확인
ros2 topic list | grep -E "camera|joint|sketch|tf"
ros2 topic hz /camera/image_raw         # ~30Hz

# TF 확인
ros2 run tf2_tools view_frames
# → frames.pdf 에 world, base_link, tool0, SketchCamera 전부 있어야 함

# 웨이포인트 모니터링
ros2 topic echo /sketch_waypoints
```

## 사용 절차

1. sketch_ui 창에 로봇+벽 카메라 뷰가 실시간으로 뜸.
2. 벽 위에서 마우스로 선 드래그.
3. 작업 평면 x 슬라이더: 0.55~0.75, 기본 0.60 (벽 앞면).
4. "실행" 버튼 → 로봇이 벽 쪽으로 approach 한 뒤 스케치대로 움직임.

## 트러블슈팅

| 증상 | 원인 후보 |
|---|---|
| Isaac Sim 렌더 검은 화면 | RTX 5060 드라이버 < 570, 또는 play 상태 아님 |
| Isaac Sim xformOp 에러 | USD 내부 transform 충돌. 최신 코드인지 확인 |
| `Could not import rclpy` 경고 | run_isaac_sim.sh 대신 직접 실행함. 스크립트 사용 |
| sketch_ui 이미지 안 뜸 | `ros2 topic hz /camera/image_raw` 확인. 0 이면 Isaac Sim play 상태 체크 |
| TF lookup 실패 | sketch_ui 로그에서 사용 가능한 프레임 목록 확인 |
| MoveIt fraction 낮음 | 스케치가 로봇 리치 밖, x_plane 을 0.55~0.65 로 |
| MoveIt robot_description 없음 | launch 파일에 robot_state_publisher 포함 확인 |
| 로봇이 베이스 위에 안 올라감 | isaac_sim_ur10.py 의 xform translate 코드 확인 |
| `/joint_command` 무시됨 | ROS2SubscribeJointState 의 joint 이름 순서 확인 |

## Isaac Sim 버전별 차이

이 프로젝트는 Isaac Sim **5.0+** 기준으로 작성됨. 4.5 에서도 백엔드 호환 시도가 코드에
들어 있지만, 5.0 이 공식 권장.

업그레이드:
```bash
source ~/isaac_env/bin/activate
pip install --upgrade "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
```
