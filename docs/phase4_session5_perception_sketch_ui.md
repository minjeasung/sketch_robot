# Phase 4 Session 5 — ZED Perception + 브라우저 Sketch UI

## 세션 기간: 2026-05-15 ~ 16

## 목표 + 동기
Session 4 까지 시뮬에서 RB10 자세/EOAT/벽/카메라 마운트가 시각적으로 정리되었고,
Stage 5 의 motion pipeline 도 검증됨. 다음 단계는:

이 세션의 기준 EOAT 는 **페인트 롤러**다. 목표는 용접 공정 자체가 아니라, 롤러로
카메라 기반 sketch-to-surface path following 을 먼저 안정화하는 것. 이후 같은
perception / waypoint / MoveIt pipeline 위에서 용접 토치 EOAT 로 확장한다.

1. ZED 의 raw 데이터 (image, depth, pointcloud) 를 ROS topic 으로 흘리기 (Isaac Sim 안)
2. 그 데이터로 perception (벽 평면 검출, 정면 가상 view) 수행
3. 브라우저 UI 에서 정면 view 위에 사용자가 그림 → 그 픽셀이 실제 3D world 좌표로 변환
4. moveit_executor 가 그 world 좌표를 받아 롤러 EOAT 로 Stage 1~5 실행

→ 이게 Phase 4 의 마지막 시뮬 검증. Phase 5 에서 실로봇으로 옮김.

---

## 작업 분해

### Phase A — Isaac Sim ZED 통합 (시뮬 데이터 흘리기)

`isaac_sim_rb10.py` 의 카메라 부분 stereo 확장 + ROS topic 발행:

- **A1.1 Stereo Camera prim** — `/World/SketchCamera/zed_left_camera_frame`,
  `/zed_right_camera_frame`. baseline 120mm (ZED 2i 스펙). parent (SketchCamera)
  의 look-at transform 상속, 각 child 는 local `±X 0.060m` translate.
- **A1.2 ZedCameraGraph (OmniGraph)** — 2 render product +
  `ROS2CameraHelper` × 6 (left/right RGB+info, left depth + depth_pcl).
  ZED ros2 wrapper 와 동일 topic 이름.
- **A1.3 TF** — `TFGraph` 의 `targetPrims` 에 left/right prim 추가 →
  `/tf` 에 `zed_left_camera_frame`, `zed_right_camera_frame` 자동 발행.

### Phase B — Perception 파이프라인 (2 새 노드)

**B1: `wall_detector_node.py`** — pointcloud → RANSAC 평면 검출
- 입력: `/zed/zed_node/point_cloud/cloud_registered` (`sensor_msgs/PointCloud2`)
- 처리:
  1. `pc2.read_points(structured array)` → numpy (xyz)
  2. `voxel_down_sample(0.01)` — RANSAC 속도용
  3. **distance crop (`< 2.0m`)** — 작업공간 외 ground plane / 배경 제거.
     초기엔 RANSAC 이 ground plane 을 잡았는데 (벽보다 면적 큼), crop 으로 해결
  4. `o3d.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)`
  5. inlier centroid 계산, normal 정규화, +Z → normal 회전 quaternion
- 출력:
  - `/perception/wall_plane` (`geometry_msgs/PoseStamped`, frame = `zed_left_camera_frame`,
    `orientation` 의 `+Z` 가 wall normal)
  - `/perception/wall_inliers` (검증 시각용 PointCloud2)

**B2: `wall_projector_node.py`** — RGB + wall_plane → 정면 가상 view
- 입력:
  - `/zed/zed_node/left/image_rect_color` (Image, rgb8 | bgr8)
  - `/zed/zed_node/left/camera_info` (CameraInfo, K 매트릭스)
  - `/perception/wall_plane` (PoseStamped)
- 처리:
  1. quaternion → normal (local `+Z` 의 world 방향)
  2. wall 평면의 right/up axes 정의 (camera `+Y` down 기준)
  3. 4 꼭짓점 계산 (centroid ± W/2 right ± H/2 up). 작업 영역 `WALL_RECT_W × WALL_RECT_H = 0.6 × 0.6 m`
  4. K 로 카메라 픽셀 projection (`u = fx·X/Z + cx`)
  5. `cv2.getPerspectiveTransform` + `warpPerspective` → 정면 view (800×800)
- 출력: `/perception/wall_front_view` (Image, rgb8, frame = `wall_front_view`)

### Phase C — 브라우저 UI + moveit 통합

**C1: `web/` 폴더 신설** — roslibjs 기반 단일 페이지
- `index.html` — 상태 pill, ZED 카드 (canvas 두 개 겹침), wall_plane 카드, controls
- `style.css` — 다크 테마, status pill 4 상태, overlay layout (zed-wrap relative,
  sketch-canvas absolute)
- `js/app.js`:
  - `ROSLIB.Ros({url:"ws://localhost:9090"})` 연결, 상태 핸들러
  - `/perception/wall_plane` subscribe → frame_id/stamp/position/orientation 표시
  - `/zed/zed_node/left/image_rect_color` subscribe → `putImageData` 로 canvas 갱신
    (rgb8 / bgr8 / rgba8 encoding 분기)
  - sketch overlay: pointer events (mouse + touch 통합), Freehand + Line 모드,
    ESC 취소, Clear/Undo. native canvas 좌표 보관 (CSS 크기 무관)

**C2: `sketch_to_waypoints_node.py`** — 브라우저 픽셀 → 3D world waypoints
- 입력: `/sketch_pixels` (PoseArray, 브라우저가 픽셀 좌표를 pose.position.x/y 로 보냄)
- 처리: 정면 view 픽셀 → wall plane 의 world 좌표 (역 homography +
  `tf2` 로 `zed_left_camera_frame` → `world` 변환). EOAT 안전 offset (2cm) 적용.
- 출력: `/sketch_waypoints` (PoseArray, world frame, orientation = wall normal 향함)

**C3: `moveit_executor.py` 패치** — 새 perception 흐름 통합
- `_compute_snapped_tcp_waypoints` + `plan_cartesian` 의 yaml 기반 표면 스냅 제거.
- waypoint 의 position/orientation 그대로 사용 (sketch_to_waypoints 가 이미 처리).
- target/n 은 caller 호환 위해 yaml 에서 계속 받음 (offset_along_normal 용).
- Stage 2 cartesian threshold 0.85 (95% → 85% 완화, 실제 trajectory 의 미세 편차 허용).
- 현재 EOAT 는 AFT200+roller attached collision 이며, 용접 토치용 공정 제어는 아직
  적용하지 않는다.

---

## 새 ROS nodes 요약

| 노드 | 입력 | 출력 | 역할 |
|---|---|---|---|
| `wall_detector` | `/zed/...point_cloud/cloud_registered` | `/perception/wall_plane`, `/perception/wall_inliers` | RANSAC 벽 평면 |
| `wall_projector` | `/zed/...left/image_rect_color`, `.../camera_info`, `/perception/wall_plane` | `/perception/wall_front_view` | 정면 가상 view (homography) |
| `sketch_to_waypoints` | `/sketch_pixels` (브라우저), `/perception/wall_plane`, `/tf` | `/sketch_waypoints` | 픽셀 → world 3D |

---

## 브라우저 UI 구조 (`web/`)

```
web/
├─ index.html         (78 lines)
├─ style.css          (~210 lines)
└─ js/app.js          (~340 lines)
```

- **2 canvas overlay**: `#zed-canvas` (RGB putImageData, 1280×720) +
  `#sketch-canvas` (absolute top:0, transparent, pointer-events on)
- **데이터**: `strokes: [{type:"freehand"|"line", points:[{u,v}]}]`, canvas native 좌표
- **모드**: 라디오 (Freehand/Line). Line 모드 preview (점선 + 첫 점 marker), ESC 취소
- **버튼**: Undo (마지막 stroke), Clear, Execute (`/sketch_pixels` publish),
  Run Robot (`/sketch_execute` publish)

---

## 검증 결과

- **RANSAC inlier ratio:** ~99% (벽 정면, 거리 1.3m, distance_threshold=0.02m).
  초기엔 ground plane 잡힘 → distance crop (<2m) 으로 해결.
- **/perception/wall_plane rate:** ~8 Hz (RANSAC 무거움. PointCloud rate 보다 낮음).
- **/perception/wall_front_view rate:** ~30 Hz (warpPerspective 가벼움, RGB rate 따라감).
- **브라우저 sketch → moveit_executor Stage 1~5:** SUCCESS. 롤러 EOAT 가 벽을 뚫지 않고
  2cm offset 으로 표면 추종. Stage 2 의 cartesian 이 0.85 threshold 로 통과.

---

## 남은 이슈

1. **Stage 1 OMPL 우회** — 일부 자세에서 RRTConnect 가 plan 실패. RRTstar 시도했으나
   유의미한 개선 없음. RRTConnect 유지. 자세 분포가 좁아 large-scale planning 부담.
2. **EOAT 와 로봇팔 자기 충돌 시각 부근 의심** — RViz 시각에서 AFT200/roller 가 wrist 와
   가까워 보일 때 있음. 실제 MoveIt 의 collision check 는 OK (touch_links 설정 덕).
   외형만 의심스러움.
3. **작업 가능 영역 시각 표시 (옵션 보류)** — 벽 위에서 로봇이 도달 가능한 영역을
   브라우저 UI 에 overlay 하는 기능. 현재는 사용자가 시도 후 plan 실패 시 알게 됨.

---

## 다음 단계 — Phase 5 (실로봇, 롤러 우선)

- snucem 에 sketch_robot_ws 동기화 (origin/main pull)
- rbpodo_ros2 fork 도 동일 commit (use_isaac_sim 분기는 false 로 두면 실로봇 path)
- 실 ZED 2i 연결 + zed_ros2_wrapper 띄움 (Isaac Sim 의 토픽 이름 그대로라 무변경)
- 실 환경 측정 (벽/카메라 위치) 으로 `docs/phase4_session4_environment.md` 갱신
- 페인트 롤러 EOAT 기준 offset / collision / reachable workspace 를 먼저 확정
- 단계적 검증:
  1. wall_detector 가 실 ZED pointcloud 로 평면 검출하는지
  2. wall_projector 결과가 시각적으로 맞는지
  3. 브라우저 sketch → sketch_to_waypoints → moveit_executor Stage 1 단독
  4. Stage 1~5 전체, 속도 0.1× 부터 시작
- 위 검증이 끝난 뒤 용접 토치 EOAT 와 용접 공정 파라미터로 확장

---

## 주요 commit

- `fdafea2` — Session 5 핵심 (ZED stereo + perception 노드들 + 브라우저 UI)
- `6a87315`
- `bd9a3ff`

(자세한 변경 내역은 `git log --oneline phase4-session5..main` 또는 위 commit 의 message 참조)
