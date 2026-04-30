# Phase 3 — Attached torch collision shape 의 axis 수정

## 배경 — 진짜 원인 발견

수많은 테스트 끝에 OMPL Stage 1 의 trajectory 자체가 wall 을 가로지른 게 확인됨
(signed_dist 가 -77mm 까지). 이전에 cartesian collision check 문제로 추정했지만
실제로는 OMPL 까지 포함한 **MoveIt 전체의 collision check** 가 잘못된 것이었음.

근본 원인: **attached torch 의 collision shape (cylinder) 의 axis 가 잘못 등록됨**.

## 현재 코드 (잘못됨)

`moveit_executor.py` 의 `publish_scene_periodic` 에서:

```python
torch_prim = SolidPrimitive()
torch_prim.type = SolidPrimitive.CYLINDER
torch_prim.dimensions = [TORCH_LENGTH, 0.025]  # [height=0.25, radius=0.025]
torch_pose = Pose()
torch_pose.position.z = TORCH_LENGTH / 2.0  # 중심 +Z 12.5cm  ← 잘못!
torch_pose.orientation.w = 1.0  # identity 회전
```

이는 cylinder 가 **tool0 의 +Z 축 방향으로** 0~25cm 에 위치한다고 등록함.
하지만 실제 torch 는 `_brush_tip_to_tool0` 함수에서 보듯 **tool0 의 +Y 축 방향**으로 25cm 뻗음.

```python
# _brush_tip_to_tool0 의 핵심:
ly = np.array([2*(x*y - z*w), 1 - 2*(x*x + z*z), 2*(y*z + x*w)])  # tool0 local +Y in world
new_pos = pos - TORCH_LENGTH * ly  # tool0 = torch_tip - TORCH_LENGTH * (tool0 +Y)
```

즉 **실제 torch 는 +Y 방향**, **collision shape 은 +Z 방향** — 90도 어긋남.

결과: OMPL/cartesian 의 collision check 는 잘못된 위치 (+Z 방향) 의 가짜 cylinder 를
wall 과 충돌 안 시키게 경로 생성. 실제 torch (+Y 방향) 는 wall 통과해도 무관.

이게 Phase 1 부터의 잠재 버그였음. Cartesian path (stage 3) 는 표면을 따라가니
torch 가 wall 안 통과 → 우연히 드러나지 않음. OMPL 의 자유 경로 (stage 1) 가
도입되면서 비로소 드러남.

## 해결

SolidPrimitive.CYLINDER 의 기본 axis 는 +Z 이므로, 90도 X-축 회전 quaternion 을 적용해
cylinder 의 axis 가 +Y 가 되도록 함. 그리고 중심 위치도 +Y 12.5cm 로 조정.

## 변경 사항

대상 파일: `~/sketch_robot_ws/src/sketch_control/sketch_control/moveit_executor.py`

`publish_scene_periodic` 함수 안의 attached torch 등록 부분을 다음과 같이 수정.

기존 코드 (블록 단위로 식별):

```python
        # --- 토치 (tool0 에 attached, 단순 cylinder 25cm, radius 2.5cm) ---
        torch_aco = AttachedCollisionObject()
        torch_aco.link_name = EE_LINK  # "tool0"
        torch_aco.object.id = "brush"  # id 는 기존 유지 (scene_confirmed 로직 호환)
        torch_aco.object.header.frame_id = EE_LINK
        torch_prim = SolidPrimitive()
        torch_prim.type = SolidPrimitive.CYLINDER
        torch_prim.dimensions = [TORCH_LENGTH, 0.025]  # [height, radius]
        torch_pose = Pose()
        torch_pose.position.z = TORCH_LENGTH / 2.0  # 중심 +Z 12.5cm
        torch_pose.orientation.w = 1.0
        torch_aco.object.primitives.append(torch_prim)
        torch_aco.object.primitive_poses.append(torch_pose)
        torch_aco.object.operation = CollisionObject.ADD
        torch_aco.touch_links = ["tool0", "wrist_3_link", "wrist_2_link", "flange"]
        ps.robot_state.attached_collision_objects.append(torch_aco)
        ps.robot_state.is_diff = True
```

변경 후:

```python
        # --- 토치 (tool0 에 attached, cylinder 25cm, radius 2.5cm) ---
        # 실제 torch 는 tool0 의 +Y 방향으로 뻗음 (수동 캘리브레이션으로 확정).
        # SolidPrimitive.CYLINDER 의 기본 axis 는 +Z 이므로 X-축 90도 회전 적용.
        import math
        torch_aco = AttachedCollisionObject()
        torch_aco.link_name = EE_LINK  # "tool0"
        torch_aco.object.id = "brush"  # id 는 기존 유지 (scene_confirmed 로직 호환)
        torch_aco.object.header.frame_id = EE_LINK
        torch_prim = SolidPrimitive()
        torch_prim.type = SolidPrimitive.CYLINDER
        torch_prim.dimensions = [TORCH_LENGTH, 0.025]  # [height=0.25, radius=0.025]
        torch_pose = Pose()
        # 중심 위치를 +Y 12.5cm 로 (cylinder 가 +Y 0 ~ +Y 25cm 차지)
        torch_pose.position.x = 0.0
        torch_pose.position.y = TORCH_LENGTH / 2.0  # 중심 +Y 12.5cm
        torch_pose.position.z = 0.0
        # X-축 90도 회전: cylinder 의 default +Z axis 를 +Y 로 회전
        # quaternion = (sin(45°), 0, 0, cos(45°)) ≈ (0.7071, 0, 0, 0.7071)
        half_angle = math.pi / 4.0
        torch_pose.orientation.x = math.sin(half_angle)
        torch_pose.orientation.y = 0.0
        torch_pose.orientation.z = 0.0
        torch_pose.orientation.w = math.cos(half_angle)
        torch_aco.object.primitives.append(torch_prim)
        torch_aco.object.primitive_poses.append(torch_pose)
        torch_aco.object.operation = CollisionObject.ADD
        torch_aco.touch_links = ["tool0", "wrist_3_link", "wrist_2_link", "flange"]
        ps.robot_state.attached_collision_objects.append(torch_aco)
        ps.robot_state.is_diff = True
```

핵심 변경:
1. `torch_pose.position.z = TORCH_LENGTH / 2.0` → `torch_pose.position.y = TORCH_LENGTH / 2.0`
2. `torch_pose.orientation.w = 1.0` (identity) → X-축 90도 회전 quaternion
3. import math 추가 (이미 있는 import 라면 무시)

## 추가 — `import math` 위치

`import math` 가 파일 상단의 import 영역에 이미 있는지 확인. 없으면 다음과 같이 추가:

```python
import copy
import math   # ← 추가
import threading
import numpy as np
```

함수 내부에 `import math` 하는 것보다 상단에 두는 게 깔끔.

## Build 및 검증

```bash
cd ~/sketch_robot_ws
colcon build --packages-select sketch_control --symlink-install
source install/setup.bash
python3 -c "from sketch_control.moveit_executor import MoveItExecutor; print('OK')"
```

## 작업 가이드

1. import math 추가 (상단).
2. `publish_scene_periodic` 의 torch 등록 블록 변경.
3. 함수 내부의 `import math` 는 제거 (상단으로 이동).
4. Build + import 확인.
5. 변경 요약 출력 (어느 라인 / 어떻게).
