# Phase 3 — PlanningScene 을 ApplyPlanningScene service 로 적용

## 배경 — 진짜 진짜 원인 발견

진단 명령 (`/get_planning_scene` service call) 결과:
```
world=PlanningSceneWorld(collision_objects=[], ...)
```

`world.collision_objects` 가 **비어있음**. 즉 `wall` 이 MoveIt 의 PlanningScene 에
등록되지 않은 상태. OMPL Stage 1 의 collision check 시 충돌할 객체가 없으니
직선 trajectory 생성 → wall 통과.

이전에 `[OK] PlanningScene 검증: ['wall'] + brush 등록` 로그가 거짓 안심 메시지였음.
그 검증은 `/monitored_planning_scene` topic 으로 돌아오는 메시지 확인일 뿐,
실제 MoveIt collision_detection 에 등록됐는지는 보장 못 함.

원인: `scene_pub.publish(ps)` 는 단순 broadcast. MoveIt 의 PlanningSceneMonitor 가
받아서 처리하는 건 timing/race 의존. `/apply_planning_scene` service 호출이
공식 등록 방법.

## 변경 사항

대상 파일: `~/sketch_robot_ws/src/sketch_control/sketch_control/moveit_executor.py`

### 1. import 추가

기존:
```python
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
```

변경:
```python
from moveit_msgs.srv import GetCartesianPath, GetPositionIK, ApplyPlanningScene
```

### 2. `__init__` 안에 service client 추가

기존 `self.ik_client = self.create_client(...)` 근처에 추가:

```python
self.apply_scene_client = self.create_client(
    ApplyPlanningScene, "/apply_planning_scene")
```

`self.move_action_client` 추가한 곳 옆이 적당.

### 3. `publish_scene_periodic` 함수 변경

기존 함수의 끝 부분:

```python
        self.scene_pub.publish(ps)
        if not self.scene_initialized:
            self.get_logger().info(
                "PlanningScene: 물체 + 토치(attached to tool0, 25cm) 퍼블리시")
            self.scene_initialized = True
```

변경:

```python
        # publish 도 유지 (RViz 시각화 용)
        self.scene_pub.publish(ps)

        # ApplyPlanningScene service 로 진짜 등록 (MoveIt 의 collision detection 에 반영)
        if self.apply_scene_client.wait_for_service(timeout_sec=1.0):
            req = ApplyPlanningScene.Request()
            req.scene = ps
            future = self.apply_scene_client.call_async(req)
            future.add_done_callback(self._apply_scene_done)
        else:
            self.get_logger().warn("/apply_planning_scene service 없음")

        if not self.scene_initialized:
            self.get_logger().info(
                "PlanningScene: 물체 + 토치 publish + apply 시도")
            self.scene_initialized = True

def _apply_scene_done(self, future):
    """ApplyPlanningScene service 응답 처리. 성공 시 scene_confirmed."""
    try:
        resp = future.result()
    except Exception as e:
        self.get_logger().warn(f"ApplyPlanningScene 실패: {e}")
        return
    if resp.success and not self.scene_confirmed:
        self.scene_confirmed = True
        self.get_logger().info(
            "[OK] PlanningScene apply 성공 (MoveIt 에 wall+brush 등록됨)")
    elif not resp.success:
        self.get_logger().warn("ApplyPlanningScene 실패 (success=False)")
```

`_apply_scene_done` 함수는 `publish_scene_periodic` 함수 바로 다음에 추가.

### 4. 기존 `on_scene_update` callback 변경

기존 callback 은 `/monitored_planning_scene` topic 의 메시지로 `scene_confirmed` 를
true 로 만들었지만, 이제 ApplyPlanningScene 결과 기반이므로 이 callback 의
`scene_confirmed` 설정은 제거 (남기면 두 경로에서 동시에 설정해 혼란).

기존:
```python
def on_scene_update(self, msg):
    scene_ids = set(obj.id for obj in msg.world.collision_objects)
    objects_ok = self._enabled_ids.issubset(scene_ids)
    brush_ok = any(
        ao.object.id == "brush"
        for ao in msg.robot_state.attached_collision_objects)
    if objects_ok and brush_ok and not self.scene_confirmed:
        self.scene_confirmed = True
        self.get_logger().info(
            f"[OK] PlanningScene 검증: {sorted(self._enabled_ids)} + brush 등록")
```

변경 (확인용 로그만 남기고 scene_confirmed 설정은 제거):
```python
def on_scene_update(self, msg):
    """모니터링 용. scene_confirmed 는 ApplyPlanningScene 결과로 설정."""
    scene_ids = set(obj.id for obj in msg.world.collision_objects)
    objects_ok = self._enabled_ids.issubset(scene_ids)
    brush_ok = any(
        ao.object.id == "brush"
        for ao in msg.robot_state.attached_collision_objects)
    if objects_ok and brush_ok and not getattr(self, "_monitor_logged", False):
        self._monitor_logged = True
        self.get_logger().info(
            f"[INFO] /monitored_planning_scene 에 {sorted(self._enabled_ids)} + brush 보임 "
            "(apply 결과로 confirmed 됨)")
```

## Build 및 검증

```bash
cd ~/sketch_robot_ws
colcon build --packages-select sketch_control --symlink-install
source install/setup.bash
python3 -c "from sketch_control.moveit_executor import MoveItExecutor; print('OK')"
```

## 검증 시 기대 동작

launch 재기동 후 약 1-3 초 안에:
```
[INFO] [moveit_executor]: PlanningScene: 물체 + 토치 publish + apply 시도
[INFO] [moveit_executor]: [OK] PlanningScene apply 성공 (MoveIt 에 wall+brush 등록됨)
```

이 로그 보이면 등록 성공.

그 후 진단 명령:
```bash
ros2 service call /get_planning_scene moveit_msgs/srv/GetPlanningScene \
  "{components: {components: 1024}}" 2>&1 | head -50
```

기대: `world.collision_objects` 안에 `id='wall'` 보임.

## 작업 가이드

1. import 추가.
2. service client 추가.
3. `publish_scene_periodic` 변경.
4. `_apply_scene_done` 함수 추가.
5. `on_scene_update` 변경.
6. Build + import 확인.
7. 변경 요약 출력.
