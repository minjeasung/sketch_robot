# Phase 4 Session 4 — 실험실 환경 측정값

## 측정일: 2026-05-12

## 좌표계
- world / World = 실제 RB10 pendant/base = (0, 0, 0)
- 단위: 미터 (Isaac Sim / ROS 표준)
- **현재 실행 기준은 RB10 base 좌표계**
- 벽/작업면은 base +X 방향에 있음
- link0 의 z=0 평면 = 철판 윗면 (로봇이 실제 마운트된 면)
- 주의: rbpodo URDF 의 `link0` 좌표계는 실제 RB10 pendant/base 좌표계와
  Z축 기준 90도 차이가 있다. 실행 시 `world`/`World` 는 실제 base 로 두고,
  URDF `link0` 는 static TF / Isaac root 에서 +90도 보정한다.

## 좌표 변환 메모
초기 환경 측정 일부는 READY_POSE 의 TCP 축 기준으로 기록되어 있었다.
실행 코드와 `objects.yaml` 에서는 아래 변환을 적용해 base 좌표로 사용한다:
- base_x = -tcp_y
- base_y = +tcp_x
- base_z = tcp_z

주의:
- 로봇 joint pose 값은 이미 실제 RB10/base 기준으로 측정된 값이므로 변환하지 않는다.
- 변환 대상은 벽, 테이블, 카메라, 마운트 등 환경/충돌 물체 좌표이다.

## 로봇
- RB10-1300E_U, 철판 위 고정
- 현재 READY/VIEW_POSE 로 사용하는 실제 RB10 joint pose:
  - J0 (base):     +0.03°  ≈  +0.0005 rad
  - J1 (shoulder): -53.53° ≈  -0.9343 rad
  - J2 (elbow):   +138.92° ≈  +2.4246 rad
  - J3 (wrist1):   -93.35° ≈  -1.6293 rad
  - J4 (wrist2):   +89.81° ≈  +1.5675 rad
  - J5 (wrist3):    0.000 rad
- 이 pose 값은 변환하지 않고 그대로 사용한다.

## 마운팅 적층 (아래에서 위로)
- 바닥 ~ 테이블 윗면: 테이블 (높이 0.813 m)
- 테이블 윗면 ~ 철판 윗면: 10mm 철판 (로봇 base 바로 아래의 작은 마운팅 플레이트 — 테이블 ≠ 철판)
- 철판 윗면 = link0 z=0 (로봇 base 가 이 면에 마운트)

## 테이블 (로봇 받침대)
- 중심: [-0.10, +0.05, -0.4165] m
- 크기: [0.60, 0.80, 0.813] m (base X × base Y × Z)
- link0 기준 영역:
  - x ∈ [-0.40, +0.20]
  - y ∈ [-0.35, +0.45]
  - z ∈ [-0.823, -0.010]

## 철판 (로봇 마운팅 플레이트)
- 의미: 로봇 base 가 마운트되는 작은 강판. 테이블 ≠ 철판 (테이블 위에 철판이 얹힘).
- 크기: 0.20 × 0.20 × 0.010 m (200mm 정사각형, 두께 10mm)
- link0 기준 영역:
  - x ∈ [-0.10, +0.10]
  - y ∈ [-0.10, +0.10]
  - z ∈ [-0.010, 0]
- 윗면이 link0 의 z=0 평면.

## 벽 (작업면)
- 중심: [+0.81, 0.0, +0.5] m
- 위치 평면: x = +0.80 m (link0 기준 +X 방향)
- 평면 normal: (-1, 0, 0) — 벽 표면이 base -X, 즉 로봇/자유공간 방향을 향함
- 크기: [0.02, 2.0, 1.5] m (두께 X × 폭 Y × 높이 Z)
- Y 범위: [-1.0, +1.0]
- Z 범위: [-0.25, +1.25]
- 작업영역: 중심 [0.80, 0.0, 0.5] m, 폭 0.5 m(base Y), 높이 0.4 m(Z)
- 표면: 흰색 벽, 노란/주황 마스킹 테이프 격자

## EOAT (페인트 롤러)
- 현재 단계의 기준 EOAT 는 용접 토치가 아니라 페인트 롤러.
- 목적: 용접 전 단계로, 롤러를 이용해 카메라 기반 sketch-to-surface 경로 추종과
  실로봇 안전 절차를 먼저 검증한다.
- 용접 토치 EOAT 는 롤러 pipeline 이 안정화된 뒤 후속 단계에서 같은 perception /
  motion 구조 위에 얹는다.
- 도구 체인: TCP → AFT200 F/T sensor → 페인트 롤러 (도장 작업)
- AFT200 URDF / RR-00A_B__EOAT STEP 은 CAD local +Z 방향으로 뻗지만, 실제 장착은
  TCP local -Y 방향으로 해석한다.
- ROS/MoveIt TCP frame 기준 → 롤러 중심까지 TCP local -Y 방향 거리:
  0.261675 m (= AFT200 0.0522 m + roller STEP forward 0.209475 m)
- 롤러 직경: 0.05 m (Φ50mm)
- 롤러 길이 (축 방향): 0.18 m
- Mount axis: 코드 변수 TOOL_AXIS="-Y"/"-y" 로 관리
- 실제 base 보정 후 스케치 목표 자세에서는 TCP local -Y 가 base +X,
  즉 벽 접근 방향을 향한다.
- 작업 시 EE 와 벽 거리: 0.261675 + 0.025 = 0.286675 m (벽 표면 ~ TCP)

## ZED 카메라 (외부 고정, eye-to-hand)
- ZED collision/root 중심 (link0 기준): [-0.5, 0.2, 1.0] m
- ZED optical static TF:
  - translation: [-0.471575, 0.256286, 1.008593] m
  - rotation xyzw: [-0.536213, 0.626502, -0.429749, 0.367815]
- 마운팅: ㄴ자 알루미늄 프레임 (50×50mm 단면, 총 높이 0.9m)
  - 세그먼트1 (수평, 책상 위): 중심 [-0.35, 0.2, 0.015], 크기 [0.3, 0.05, 0.05], z ∈ [-0.01, 0.04]
  - 세그먼트2 (수직, 세그먼트1 위): 중심 [-0.5, 0.2, 0.465], 크기 [0.05, 0.05, 0.85], z ∈ [0.04, 0.89]
  - 볼헤드: 중심 [-0.5, 0.2, 0.92]
  - 볼트: 중심 [-0.5, 0.2, 0.95]
- 방향: 벽 향함 (base +X 방향)

## 안전
- EE z 좌표 ≥ 0 유지 (base 아래로 가지 않음)
- 작업면도 z ∈ [0, 1.0] 만 사용
- 롤러 attached collision 로 등록
- 테이블/철판도 collision object 로 등록

## 다음 단계
- isaac_sim_rb10.py 와 objects.yaml 은 위 base 기준 좌표 사용
- wall + table + 철판 + mount + zed_camera collision 등록
- 벽/작업영역은 ZED perception 으로 실제 평면을 다시 잡되, set target 이후 작업대상은 고정
- 페인트 롤러 attached collision (cylinder: r=0.025, L=0.18)
- 롤러 검증 완료 후 용접 토치 EOAT 로 확장할 때 offset / collision / process
  parameter 를 별도 갱신
