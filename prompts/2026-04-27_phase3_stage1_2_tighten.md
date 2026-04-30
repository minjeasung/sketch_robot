# Phase 3 — Stage 1 + 2 의 tight tolerance + short normal approach

## 배경

OMPL Stage 1 + cartesian Stage 2 의 4-stage 파이프라인이 작동하지만,
두 번째 Submit (retreat 위치 → 새 안전점 → 새 표면 첫 점) 에서 토치가 wall 을 21mm 침투.

진단 (signed_dist 로그):
- Stage 1 의 OMPL planning OK, 자유 경로로 안전점 근처 도달
- Stage 2 의 cartesian path: fraction = 100% 라고 보고됨
- 하지만 실제 stage 2 trajectory 가 wall 을 가로지름 (signed_dist 가 +119 → -21mm 로)

추정 원인:
1. OMPL position constraint 의 sphere tolerance 가 5mm 로 큼.
   OMPL 이 안전점에서 옆으로 5mm 빗나간 위치에서 끝남.
2. 빗나간 위치에서 표면 첫 점까지 cartesian 직선 (15cm) 그으면 wall 가로지를 수 있음.
3. Cartesian path 의 collision check 가 attached cylinder torch 를 제대로 검사하지 못해
   wall 침투를 못 잡음 (fraction = 100% 잘못 보고).

해결 전략:
- OMPL 의 position tolerance 를 1mm 로 tight 하게 → OMPL 끝점이 안전점에 거의 정확히.
- SAFETY_OFFSET 을 5cm 로 줄임 → Stage 2 의 cartesian 직선이 짧음.
- 두 변경의 결합으로 Stage 2 직선이 거의 정확한 normal 방향 (5cm) 이 되어
  geometrically wall 안으로 못 들어감.

## 변경 사항

대상 파일: `~/sketch_robot_ws/src/sketch_control/sketch_control/moveit_executor.py`

### 1. SAFETY_OFFSET 변경

기존 (0.15 또는 0.25):
```python
SAFETY_OFFSET = 0.15   # 또는 0.25 — 이전 실험에서 변경됐을 수 있음
```

변경:
```python
SAFETY_OFFSET = 0.05   # 5cm 짧은 normal-direction approach
```

`grep -n "SAFETY_OFFSET" ~/sketch_robot_ws/src/sketch_control/sketch_control/moveit_executor.py`
로 현재 값 확인 후 sed:
```bash
sed -i 's/SAFETY_OFFSET = 0.15/SAFETY_OFFSET = 0.05/' \
   ~/sketch_robot_ws/src/sketch_control/sketch_control/moveit_executor.py
sed -i 's/SAFETY_OFFSET = 0.25/SAFETY_OFFSET = 0.05/' \
   ~/sketch_robot_ws/src/sketch_control/sketch_control/moveit_executor.py
```

(둘 중 하나만 적용됨, 둘 다 OK.)

### 2. `_make_pose_constraints` 의 sphere tolerance tighten

기존 함수에서 다음 줄:
```python
sp.dimensions = [0.005]  # 5mm tolerance
```

변경:
```python
sp.dimensions = [0.001]  # 1mm tolerance (tight)
```

또한 orientation constraint 의 tolerance 도 약간 tight:

기존:
```python
oc.absolute_x_axis_tolerance = 0.05
oc.absolute_y_axis_tolerance = 0.05
oc.absolute_z_axis_tolerance = 0.05
```

변경:
```python
oc.absolute_x_axis_tolerance = 0.02
oc.absolute_y_axis_tolerance = 0.02
oc.absolute_z_axis_tolerance = 0.02
```

### 3. Stage 2 의 cartesian 검증 디버그 로그 (선택, 권장)

`stage2_approach_linear` 함수의 끝 부분 (cartesian_client.call_async 호출 직전) 에 다음 디버그 로그 추가:

```python
# 디버그: stage 2 의 시작 → 끝 거리와 방향 검증
import numpy as np
end_pose = self._stage3_tool0_wps[0]
# 현재 tool0 위치는 정확히 모르지만, 직전 stage 1 의 의도된 도착점 (safety_tool0_pose) 로 근사
start_pose = self._safety_tool0_pose
delta = np.array([
    end_pose.position.x - start_pose.position.x,
    end_pose.position.y - start_pose.position.y,
    end_pose.position.z - start_pose.position.z,
])
dist = float(np.linalg.norm(delta))
direction = delta / (dist + 1e-9)
target = get_target(self.cfg, self.active_target_name)
_, n_target = get_surface_plane(target)
align = float(np.dot(direction, -np.asarray(n_target)))  # +1 이 완벽한 normal 진입
self.get_logger().info(
    f"[STAGE 2 DEBUG] dist={dist*100:.1f}cm "
    f"direction=({direction[0]:+.2f},{direction[1]:+.2f},{direction[2]:+.2f}) "
    f"normal_align={align:+.3f} (1.0=perfect)")
```

`stage2_approach_linear` 함수 안에서 `future = self.cartesian_client.call_async(req)` 줄 바로 앞에 삽입.

이 로그로 stage 2 가 진짜 normal 방향 짧은 직선인지 확인 가능. `dist` 는 5cm 근처여야 하고 `normal_align` 은 +0.95 이상이어야.

## Build 및 검증

```bash
cd ~/sketch_robot_ws
colcon build --packages-select sketch_control --symlink-install
source install/setup.bash
python3 -c "from sketch_control.moveit_executor import MoveItExecutor; print('OK')"
```

## 작업 가이드

1. 변경 1, 2, 3 순서대로 적용.
2. Build 확인.
3. 변경 요약 출력 (어떤 라인이 바뀌었는지).

## 참고 — 변경 후 기대 동작

각 Submit 시:
- Stage 1: OMPL 이 현재 → 안전점 (표면에서 normal 5cm 후퇴) 자유 경로
- Stage 2: 안전점 → 표면 첫 점 (정확히 normal 방향 5cm 직선)
- Stage 3: 표면 위 비드 (기존)
- Stage 4: 마지막 점 → 후퇴점 (normal 15cm 후퇴)

Stage 2 의 5cm 직선은 정확히 표면 normal 방향이라
geometrically wall 안으로 들어갈 수 없음 (wall 두께 10cm 의 -X 면 위에서 +X 방향으로 진입).
