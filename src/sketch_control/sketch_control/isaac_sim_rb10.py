"""
Isaac Sim 5.1 — RB10-1300E_U 시뮬레이션 환경 (Phase 4 Session 4 셋업)

좌표계: world = link0 (RB10 base) = (0, 0, 0), 단위 미터, TCP frame 정렬
환경 측정값 출처: docs/phase4_session4_environment.md (2026-05-12)

실행: (ROS2 bridge 환경 필요 — run_isaac_sim.sh 와 동일 LD_LIBRARY_PATH 셋업)
    source ~/isaac_env/bin/activate
    isaacsim --exec ~/sketch_robot_ws/src/sketch_control/sketch_control/isaac_sim_rb10.py

씬 구성:
  - RB10 USD (link0 = world 원점)
  - 테이블 (0.6×0.8×0.813 m, z ∈ [-0.823, -0.010])
  - 철판 (0.6×0.8×0.010 m, z ∈ [-0.010, 0])
  - 벽 (y=-0.78 평면, 1.0×1.0 m, 두께 0.05m)
  - ZED 카메라 prim (-0.25, 0.7, 0.9) → 벽 향함
  - ground plane (z=-0.823, 실제 바닥 — 테이블 다리 끝)

퍼블리시 (OmniGraph): /joint_states, /tf
구독 (OmniGraph): /joint_command
"""
import numpy as np
import time as _time
import omni
import omni.kit.app
import omni.graph.core as og
from pxr import UsdGeom, UsdPhysics, Gf, Sdf
import usdrt.Sdf

# ---- ROS2 bridge enable -------------------------------------------------------
try:
    from isaacsim.core.utils.extensions import enable_extension
except ImportError:
    from omni.isaac.core.utils.extensions import enable_extension

try:
    enable_extension("isaacsim.ros2.bridge")
    _bridge_name = "isaacsim.ros2.bridge"
except Exception:
    enable_extension("omni.isaac.ros2_bridge")
    _bridge_name = "omni.isaac.ros2_bridge"

_manager = omni.kit.app.get_app().get_extension_manager()
for _i in range(50):
    _ext_id = _manager.get_enabled_extension_id(_bridge_name)
    if _ext_id:
        print(f"[OK] ROS2 bridge enabled: {_ext_id}")
        break
    _time.sleep(0.1)
else:
    print(f"[FATAL] ROS2 bridge '{_bridge_name}' enable 실패")
    import sys; sys.exit(1)

# ---- Isaac Sim core import (5.1 / 5.0 / 4.5 호환) ------------------------------
try:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation as Robot
    from isaacsim.core.api.objects import VisualCuboid
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
except ImportError:
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.objects import VisualCuboid
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.types import ArticulationAction

# ---- RB10 USD 경로 ------------------------------------------------------------
RB10_USD_PATH = "/home/minjea/sketch_robot_ws/isaac_assets/rb10_1300e_u.usd"
RB10_PRIM_PATH = "/World/rb10"

# ---- 월드 + ground plane ------------------------------------------------------
stage = omni.usd.get_context().get_stage()
world = World(stage_units_in_meters=1.0)
# Ground plane = 실제 바닥 (테이블 다리 바닥). docs 의 적층:
#   link0=z=0 (철판 윗면) → 철판 z∈[-0.01,0] → 테이블 z∈[-0.823,-0.01] → 바닥 z=-0.823
world.scene.add_default_ground_plane(z_position=-0.823)
print(f"[OK] World + ground plane (z=-0.823, 실제 바닥)")

# ---- RB10 USD 로딩 ------------------------------------------------------------
add_reference_to_stage(usd_path=RB10_USD_PATH, prim_path=RB10_PRIM_PATH)
print(f"[OK] RB10 USD reference 추가: {RB10_USD_PATH} → {RB10_PRIM_PATH}")

# RB10 USD 의 base frame 이 펜던트 좌표축과 90° 어긋남 → 시계방향 -90° (Z축) 회전.
# 회전 후: USD local +X → world -Y, USD local +Y → world +X (펜던트 X-Y 와 일치)
_rb10_root_prim = stage.GetPrimAtPath(RB10_PRIM_PATH)
_rb10_xform = UsdGeom.Xformable(_rb10_root_prim)
for _op in _rb10_xform.GetOrderedXformOps():
    _rb10_xform.GetPrim().RemoveProperty(_op.GetOpName())
_rb10_xform.AddRotateZOp().Set(-90.0)  # degrees, 시계방향 90°
print(f"[OK] RB10 base Z축 -90° (시계방향) 회전 적용")

# ---- Joint drive 강제 적용 (URDF→USD 변환 시 drive 누락 가능) ------------------
# 모든 revolute joint 에 UsdPhysics.DriveAPI 추가. stiffness/damping 미설정 시
# articulation 의 position drive 가 0 토크로 시뮬되어 중력에 굴복.
JOINT_STIFFNESS = 10000.0
JOINT_DAMPING = 100.0
_n_drives = 0
for _p in stage.Traverse():
    if not _p.GetPath().pathString.startswith(RB10_PRIM_PATH):
        continue
    # RevoluteJoint 만 (PrismaticJoint 도 있으면 별도 처리)
    if _p.GetTypeName() != "PhysicsRevoluteJoint":
        continue
    _drive = UsdPhysics.DriveAPI.Apply(_p, "angular")
    _drive.CreateTypeAttr("force")
    _drive.CreateStiffnessAttr(JOINT_STIFFNESS)
    _drive.CreateDampingAttr(JOINT_DAMPING)
    _drive.CreateMaxForceAttr(1e6)
    _n_drives += 1
print(f"[OK] revolute joint {_n_drives}개에 angular drive 적용 "
      f"(stiffness={JOINT_STIFFNESS}, damping={JOINT_DAMPING})")

# ArticulationRoot 탐색 (USD 내부 어딘가에 있을 수 있음)
ARTICULATION_PATH = None
for _p in stage.Traverse():
    _path_str = _p.GetPath().pathString
    if _path_str.startswith(RB10_PRIM_PATH) and _p.HasAPI(UsdPhysics.ArticulationRootAPI):
        ARTICULATION_PATH = _path_str
        print(f"[OK] ArticulationRoot 발견: {ARTICULATION_PATH}")
        break

if ARTICULATION_PATH is None:
    rb10_prim = stage.GetPrimAtPath(RB10_PRIM_PATH)
    UsdPhysics.ArticulationRootAPI.Apply(rb10_prim)
    ARTICULATION_PATH = RB10_PRIM_PATH
    print(f"[FIX] ArticulationRoot 없음 → {ARTICULATION_PATH} 에 Apply")

robot = world.scene.add(Robot(prim_path=ARTICULATION_PATH, name="rb10"))

# ---- 테이블 + 철판 (docs/phase4_session4_environment.md) -----------------------
# 좌표 범위로부터 size/center 계산.
# 테이블: x∈[-0.35,+0.45] y∈[-0.20,+0.40] z∈[-0.823,-0.010] (x 0.8m, y 0.6m)
TABLE_SIZE = np.array([0.80, 0.60, 0.813])
TABLE_CENTER = np.array([0.05, 0.10, -0.4165])
# 철판: x∈[-0.1,+0.1] y∈[-0.1,+0.1] z∈[-0.010, 0]  (200mm 정사각형, 사용자 갱신)
PLATE_SIZE = np.array([0.20, 0.20, 0.010])
PLATE_CENTER = np.array([0.0, 0.0, -0.005])

world.scene.add(VisualCuboid(
    prim_path="/World/table",
    name="table",
    position=TABLE_CENTER,
    scale=TABLE_SIZE,
    color=np.array([0.25, 0.18, 0.12]),  # 어두운 갈색 (목재)
))
print(f"[OK] 테이블 추가: center={TABLE_CENTER.tolist()} size={TABLE_SIZE.tolist()}")

world.scene.add(VisualCuboid(
    prim_path="/World/steel_plate",
    name="steel_plate",
    position=PLATE_CENTER,
    scale=PLATE_SIZE,
    color=np.array([0.75, 0.75, 0.78]),  # 은색 (스틸)
))
print(f"[OK] 철판 추가: center={PLATE_CENTER.tolist()} size={PLATE_SIZE.tolist()}")

# ---- 벽 (Step 3) --------------------------------------------------------------
# 벽 표면(작업면)이 y=-0.78 (+Y normal, 로봇 향함). 두께 0.05m → 중심 y=-0.805.
# 표면 Z 범위 [0, 1.0] 이라 두께 cube 중심 z=0.5 (cube Z 폭 1.0m).
WALL_SIZE = np.array([1.0, 0.05, 1.0])
WALL_CENTER = np.array([0.0, -0.805, 0.5])

world.scene.add(VisualCuboid(
    prim_path="/World/wall",
    name="wall",
    position=WALL_CENTER,
    scale=WALL_SIZE,
    color=np.array([0.95, 0.95, 0.95]),  # 흰색
))
print(f"[OK] 벽 추가: center={WALL_CENTER.tolist()} size={WALL_SIZE.tolist()}")

# ---- 카메라 마운트 (ㄴ자 알루미늄 프레임, 50×50mm 단면) ------------------------
# 세그먼트1: 수평 (책상 위, +Y 방향으로 누움), 책상 위에 얹힘
# 세그먼트2: 수직 (세그먼트1 위에서 솟음)
# 둘이 합쳐 ㄴ자 — z 적층 합 0.9m
MOUNT_SEG1_CENTER = np.array([0.2, 0.35, 0.015])
MOUNT_SEG1_SIZE = np.array([0.05, 0.3, 0.05])    # z ∈ [-0.01, 0.04]
MOUNT_SEG2_CENTER = np.array([0.2, 0.5, 0.465])
MOUNT_SEG2_SIZE = np.array([0.05, 0.05, 0.85])   # z ∈ [0.04, 0.89]
_MOUNT_COLOR = np.array([0.6, 0.6, 0.65])        # 알루미늄 회색

world.scene.add(VisualCuboid(
    prim_path="/World/CameraMount_Seg1",
    name="camera_mount_seg1",
    position=MOUNT_SEG1_CENTER,
    scale=MOUNT_SEG1_SIZE,
    color=_MOUNT_COLOR,
))
print(f"[OK] 카메라 마운트 세그먼트1 (수평): "
      f"center={MOUNT_SEG1_CENTER.tolist()} size={MOUNT_SEG1_SIZE.tolist()}")

world.scene.add(VisualCuboid(
    prim_path="/World/CameraMount_Seg2",
    name="camera_mount_seg2",
    position=MOUNT_SEG2_CENTER,
    scale=MOUNT_SEG2_SIZE,
    color=_MOUNT_COLOR,
))
print(f"[OK] 카메라 마운트 세그먼트2 (수직): "
      f"center={MOUNT_SEG2_CENTER.tolist()} size={MOUNT_SEG2_SIZE.tolist()}")

# ---- ZED 카메라 prim (Step 4) -------------------------------------------------
# 위치: (0.2, 0.5, 0.915) — 세그먼트2 윗면 (z=0.89) + 카메라 cube 절반 (0.025).
# 방향: 벽 향함. target = 벽 표면 중앙 (0, -0.78, 0.5).
CAMERA_PATH = "/World/SketchCamera"
CAMERA_EYE = Gf.Vec3d(0.2, 0.5, 0.915)
CAMERA_TARGET = Gf.Vec3d(0.0, -0.78, 0.5)

cam_prim = UsdGeom.Camera.Define(stage, CAMERA_PATH)
cam_prim.GetFocalLengthAttr().Set(8.4)
cam_prim.GetHorizontalApertureAttr().Set(24.0)
cam_xform = UsdGeom.Xformable(cam_prim.GetPrim())


# USD Camera convention: local +X=right, +Y=up, -Z=forward (OpenGL).
# Gf.Matrix4d 의 row/col convention 의 모호성으로 직접 행렬 구성 시 여러 시도 모두 실패.
# scipy 로 회전 행렬 → quaternion 명시 계산 후 Translate + Orient Op 으로 적용.
from scipy.spatial.transform import Rotation as _R


def _look_at_quaternion(eye, target, up_axis=(0.0, 0.0, 1.0)):
    """eye→target 을 바라보는 USD Camera 의 world orientation.
    반환: Gf.Quatf(w, x, y, z) — AddOrientOp 와 호환."""
    eye_np = np.array([eye[0], eye[1], eye[2]], dtype=float)
    target_np = np.array([target[0], target[1], target[2]], dtype=float)
    fwd = target_np - eye_np
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up_axis, dtype=float))
    right /= np.linalg.norm(right)
    new_up = np.cross(right, fwd)
    # 회전 행렬 (columns = local basis vectors in world frame):
    #   local +X = right, local +Y = up, local +Z = -forward (Camera 의 -Z 가 forward)
    R_mat = np.column_stack([right, new_up, -fwd])
    qx, qy, qz, qw = _R.from_matrix(R_mat).as_quat()
    return Gf.Quatf(float(qw), float(qx), float(qy), float(qz))


for _op in cam_xform.GetOrderedXformOps():
    cam_xform.GetPrim().RemoveProperty(_op.GetOpName())
# 통합된 transform: 회전 + 평행이동을 단일 4x4 행렬로.
# 두 op (Orient + Translate) 분리 시 frame 간섭으로 tf 위치 어긋나는 문제 → 한 op 로
# 합쳐서 모호성 제거. SetRotateOnly + SetTranslateOnly 는 행렬의 독립 부분이라 순서 무관.
_q = _look_at_quaternion(CAMERA_EYE, CAMERA_TARGET)
# Gf.Rotation 은 Gf.Quatd 를 받음 → Quatf → Quatd 명시 변환.
_qd = Gf.Quatd(float(_q.real),
               float(_q.imaginary[0]),
               float(_q.imaginary[1]),
               float(_q.imaginary[2]))
_M = Gf.Matrix4d(1.0)
_M.SetRotateOnly(Gf.Rotation(_qd))
_M.SetTranslateOnly(Gf.Vec3d(CAMERA_EYE[0], CAMERA_EYE[1], CAMERA_EYE[2]))
cam_xform.AddTransformOp().Set(_M)
print(f"[OK] ZED 카메라 prim: {CAMERA_PATH} eye=(0.2,0.5,0.915) → target=(0,-0.78,0.5)")

# ---- ZED stereo (left/right child Camera prim) -------------------------------
# baseline 120mm (ZED 2i 스펙). USD Camera default frame: +X right, +Y up, -Z forward.
# child path 의 prim 이름 = TF link 이름 (zed_left_camera_frame / zed_right_camera_frame)
ZED_BASELINE = 0.120
ZED_IMAGE_W = 1280
ZED_IMAGE_H = 720
LEFT_CAMERA_PATH = CAMERA_PATH + "/zed_left_camera_frame"
RIGHT_CAMERA_PATH = CAMERA_PATH + "/zed_right_camera_frame"

for _path, _local_x in [(LEFT_CAMERA_PATH, -ZED_BASELINE / 2.0),
                         (RIGHT_CAMERA_PATH, +ZED_BASELINE / 2.0)]:
    _c = UsdGeom.Camera.Define(stage, _path)
    _c.GetFocalLengthAttr().Set(8.4)
    _c.GetHorizontalApertureAttr().Set(24.0)
    _xf = UsdGeom.Xformable(_c.GetPrim())
    for _op in _xf.GetOrderedXformOps():
        _xf.GetPrim().RemoveProperty(_op.GetOpName())
    _xf.AddTranslateOp().Set(Gf.Vec3d(_local_x, 0.0, 0.0))
print(f"[OK] ZED stereo: {LEFT_CAMERA_PATH} ({-ZED_BASELINE/2:+.3f}m X) | "
      f"{RIGHT_CAMERA_PATH} ({+ZED_BASELINE/2:+.3f}m X)")

# 시각 marker — Camera prim 은 viewport 에서 아이콘으로만 보임.
# ZED2 실제 dimension (~175×30×33mm) 박스로 위치 확인 가능하게.
ZED_CASE_SIZE = np.array([0.1753, 0.0303, 0.0431])
world.scene.add(VisualCuboid(
    prim_path="/World/SketchCameraCase",
    name="sketch_camera_case",
    position=np.array([CAMERA_EYE[0], CAMERA_EYE[1], CAMERA_EYE[2]]),
    scale=ZED_CASE_SIZE,
    color=np.array([0.08, 0.08, 0.08]),  # 검은색 (ZED 케이스)
))
print(f"[OK] ZED 케이스 marker: pos=(0.2,0.5,0.915) size={ZED_CASE_SIZE.tolist()}")

# ---- 초기화 -------------------------------------------------------------------
world.reset()

try:
    _dofs = robot.num_dof
    _joint_names = robot.dof_names
    print(f"[OK] RB10 articulation 유효: {_dofs} DOF, joints={_joint_names}")
except Exception as _e:
    print(f"[ERROR] RB10 articulation 실패: {_e}")
    print("      → RB10 USD 자체 문제 또는 ArticulationRoot 깨짐")

# ---- READY_POSE 자세 적용 (moveit_executor.py 의 READY_POSE_JOINTS 와 동일) -----
# 측정일 2026-05-12, TCP 위치 (0.173, -0.153, 0.739), 롤러가 -Y (벽) 향함.
# 진단용 flow (사용자 가이드):
#   1) set_joints_default_state — articulation 의 default 자세 = READY_POSE
#   2) world.reset() — default state 가 시뮬에 적용 (이미 위에서 호출됨)
#   3) set_joint_positions — 현재 자세 텔레포트
#   4) set_joint_position_targets — drive target 설정 (drive 가 그쪽으로 유지)
#   5) ArticulationController.apply_action(positions=...) — 동일 목적, 다른 API
#   6) set_gains — drive PD gain 조정 (USD drive 가 약할 경우)
# Drive 가 정상이면 위만으로 자세 유지. physics callback 으로 강제 holding 안 함.
READY_POSE_DICT = {
    "base":     0.0005,   # J0 +0.03°
    "shoulder": -0.9343,  # J1 -53.53°
    "elbow":    2.4247,   # J2 +138.92°
    "wrist1":  -1.6293,   # J3 -93.35°
    "wrist2":   1.5676,   # J4 +89.81°
    "wrist3":   0.0000,   # J5 0°
}

try:
    if all(n in READY_POSE_DICT for n in _joint_names):
        _pose_array = np.array([READY_POSE_DICT[n] for n in _joint_names])
    else:
        # dof_names 가 다른 이름이면 URDF kinematic 순서 (base→wrist3) 가정
        _pose_array = np.array([
            READY_POSE_DICT["base"], READY_POSE_DICT["shoulder"], READY_POSE_DICT["elbow"],
            READY_POSE_DICT["wrist1"], READY_POSE_DICT["wrist2"], READY_POSE_DICT["wrist3"],
        ])
        print(f"[WARN] dof_names={_joint_names} 가 READY_POSE_DICT 와 다름, URDF 순서 가정")

    # 1) default state — 이후 reset 호출 시 이 자세로 돌아감
    try:
        robot.set_joints_default_state(positions=_pose_array)
        print(f"[INIT] (1) set_joints_default_state OK")
    except Exception as _de:
        print(f"[WARN] (1) set_joints_default_state 실패 ({_de})")

    # 2) reset — default state 적용 (articulation 핸들도 재초기화)
    world.reset()
    print(f"[INIT] (2) world.reset() OK (default state 시뮬에 반영)")

    # 3) 명시적 현재 position 텔레포트
    robot.set_joint_positions(_pose_array)
    print(f"[INIT] (3) set_joint_positions OK (텔레포트)")

    # 4) drive target (drive 가 이 값으로 유지하도록)
    try:
        robot.set_joint_position_targets(_pose_array)
        print(f"[INIT] (4) set_joint_position_targets OK (drive target)")
    except Exception as _te:
        print(f"[WARN] (4) set_joint_position_targets 실패 ({_te})")

    # 5) ArticulationController 도 같은 target — (4) 와 중복이지만 안전.
    _controller = robot.get_articulation_controller()
    _controller.apply_action(ArticulationAction(joint_positions=_pose_array))
    print(f"[INIT] (5) ArticulationController.apply_action OK")

    # 6) PD gains — USD drive 가 약할 경우 보강. 적당히 큰 값.
    try:
        _controller.set_gains(
            kps=np.array([JOINT_STIFFNESS] * _dofs),
            kds=np.array([JOINT_DAMPING] * _dofs),
        )
        print(f"[INIT] (6) articulation gains: kps={JOINT_STIFFNESS} kds={JOINT_DAMPING}")
    except Exception as _ge:
        print(f"[WARN] (6) set_gains 실패 ({_ge})")

    print(f"[INIT] READY_POSE 설정 (rad): {_pose_array.tolist()}")
    print(f"[INIT] drive 기반 holding 활성 (physics callback 없음)")
    print(f"      자세 유지 실패 시 진단: drive 설정 vs gain 강도 vs dof 매핑")
except Exception as _e:
    print(f"[ERROR] READY_POSE 설정 실패: {_e}")

# ---- 페인트 롤러 EOAT (tcp 자식 prim 으로 부착) --------------------------------
# 가정 (시각 확인 후 수정):
#   - TCP local -Y 가 EOAT 가 뻗어나가는 방향 → 롤러 중심까지 (0, -0.260, 0)
#   - 롤러 cylinder long axis 는 손잡이와 직각 (가로 굴림용)
# READY_POSE 시 롤러가 world -Y (벽 쪽) 으로 뻗어야 함.

ROLLER_LENGTH = 0.18    # 축 방향
ROLLER_RADIUS = 0.025   # Φ50mm
ROLLER_OFFSET = 0.260   # TCP → roller center (axis 방향)
ROLLER_AXIS = "-Y"      # TCP local 어느 축이 손잡이 방향. 시각 검증 후 수정.

# axis 문자열 → translate 벡터 + cylinder long axis 결정
_axis_letter = ROLLER_AXIS.lstrip("+-").upper()
_axis_sign = -1.0 if ROLLER_AXIS.startswith("-") else 1.0
_axis_index = {"X": 0, "Y": 1, "Z": 2}[_axis_letter]
_offset_v = [0.0, 0.0, 0.0]
_offset_v[_axis_index] = _axis_sign * ROLLER_OFFSET
_mid_v = [0.0, 0.0, 0.0]
_mid_v[_axis_index] = _axis_sign * ROLLER_OFFSET / 2.0
# 롤러 long axis 는 손잡이와 직각. 시각 검증 결과 사용자 갱신:
#   손잡이=Y → cylinder long axis = X (Y축 둘레 90° 회전 결과)
_long_axis_map = {"X": "Y", "Y": "X", "Z": "X"}
_roller_long_axis = _long_axis_map[_axis_letter]

# tcp prim 자동 탐색
TCP_PRIM_PATH = None
for _p in stage.Traverse():
    _name = _p.GetName().lower()
    if _name == "tcp" and _p.GetPath().pathString.startswith(RB10_PRIM_PATH):
        TCP_PRIM_PATH = _p.GetPath().pathString
        break

if TCP_PRIM_PATH is None:
    print(f"[ERROR] tcp prim 못 찾음 ({RB10_PRIM_PATH} 서브트리). 롤러 부착 skip.")
else:
    print(f"[OK] tcp prim 발견: {TCP_PRIM_PATH}")

    # 기존 prim 있으면 제거 (반복 실행 안전)
    for _old in ["paint_roller", "roller_rod"]:
        _pp = TCP_PRIM_PATH + "/" + _old
        if stage.GetPrimAtPath(_pp).IsValid():
            stage.RemovePrim(_pp)

    # 롤러 cylinder
    _roller_path = TCP_PRIM_PATH + "/paint_roller"
    _roller = UsdGeom.Cylinder.Define(stage, _roller_path)
    _roller.CreateHeightAttr(ROLLER_LENGTH)
    _roller.CreateRadiusAttr(ROLLER_RADIUS)
    _roller.CreateAxisAttr(_roller_long_axis)
    _roller.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.95, 0.92)])  # 흰색
    UsdGeom.Xformable(_roller.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*_offset_v))

    # TCP ↔ 롤러 연결 막대 (시각용, 회색)
    _rod_path = TCP_PRIM_PATH + "/roller_rod"
    _rod = UsdGeom.Cylinder.Define(stage, _rod_path)
    _rod.CreateHeightAttr(ROLLER_OFFSET)
    _rod.CreateRadiusAttr(0.010)
    _rod.CreateAxisAttr(_axis_letter)  # long axis = TCP local axis 방향 (부호 무관)
    _rod.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.5, 0.55)])  # 회색 (금속)
    UsdGeom.Xformable(_rod.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*_mid_v))

    print(f"[OK] 페인트 롤러 부착: {_roller_path}")
    print(f"     손잡이 방향 (TCP local): {ROLLER_AXIS} offset={ROLLER_OFFSET}m")
    print(f"     롤러: Φ{ROLLER_RADIUS*2}m × {ROLLER_LENGTH}m, long axis=TCP local +{_roller_long_axis}")
    print(f"     연결 막대: 길이 {ROLLER_OFFSET}m, Φ0.02m, long axis=TCP local {_axis_letter}")

# ==== ROS2 OmniGraph (Step 5) ==================================================

def _node_type(name_50, name_45):
    try:
        if og.get_node_type(name_50) is not None:
            return name_50
    except Exception:
        pass
    return name_45

NT_TICK = "omni.graph.action.OnPlaybackTick"
NT_SIM_TIME = _node_type("isaacsim.core.nodes.IsaacReadSimulationTime",
                          "omni.isaac.core_nodes.IsaacReadSimulationTime")
NT_PUB_TF = _node_type("isaacsim.ros2.bridge.ROS2PublishTransformTree",
                         "omni.isaac.ros2_bridge.ROS2PublishTransformTree")
NT_PUB_JS = _node_type("isaacsim.ros2.bridge.ROS2PublishJointState",
                         "omni.isaac.ros2_bridge.ROS2PublishJointState")
NT_SUB_JS = _node_type("isaacsim.ros2.bridge.ROS2SubscribeJointState",
                         "omni.isaac.ros2_bridge.ROS2SubscribeJointState")
NT_ARTIC_CTRL = _node_type("isaacsim.core.nodes.IsaacArticulationController",
                            "omni.isaac.core_nodes.IsaacArticulationController")
NT_PUB_CLOCK = _node_type("isaacsim.ros2.bridge.ROS2PublishClock",
                            "omni.isaac.ros2_bridge.ROS2PublishClock")
NT_CREATE_RP = _node_type("isaacsim.core.nodes.IsaacCreateRenderProduct",
                          "omni.isaac.core_nodes.IsaacCreateRenderProduct")
NT_CAM = _node_type("isaacsim.ros2.bridge.ROS2CameraHelper",
                     "omni.isaac.ros2_bridge.ROS2CameraHelper")
NT_CAMINFO = _node_type("isaacsim.ros2.bridge.ROS2CameraInfoHelper",
                         "omni.isaac.ros2_bridge.ROS2CameraInfoHelper")

keys = og.Controller.Keys

# ---- TFGraph (RB10 articulation tree + ZED 카메라 TF publish) -----------------
# parentPrim=/World 기준, target 으로 articulation root 와 카메라.
# PublishTransformTree 는 articulation root 를 받으면 그 하위 link 들을 자동 추적.
_tf_targets = [
    Sdf.Path(ARTICULATION_PATH),
    Sdf.Path(CAMERA_PATH),
    Sdf.Path(LEFT_CAMERA_PATH),
    Sdf.Path(RIGHT_CAMERA_PATH),
]
if TCP_PRIM_PATH is not None:
    _tf_targets.append(Sdf.Path(TCP_PRIM_PATH))

og.Controller.edit(
    {"graph_path": "/World/TFGraph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("Tick", NT_TICK),
            ("SimTime", NT_SIM_TIME),
            ("PubTF", NT_PUB_TF),
        ],
        keys.CONNECT: [
            ("Tick.outputs:tick", "PubTF.inputs:execIn"),
            ("SimTime.outputs:simulationTime", "PubTF.inputs:timeStamp"),
        ],
        keys.SET_VALUES: [
            ("PubTF.inputs:parentPrim", [Sdf.Path("/World")]),
            ("PubTF.inputs:targetPrims", _tf_targets),
            ("PubTF.inputs:topicName", "/tf"),
        ],
    },
)

# ---- JointGraph (joint_states pub + joint_command sub) -------------------------
og.Controller.edit(
    {"graph_path": "/World/JointGraph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("Tick", NT_TICK),
            ("SimTime", NT_SIM_TIME),
            ("PubJS", NT_PUB_JS),
            ("SubJC", NT_SUB_JS),
            ("WriteJC", NT_ARTIC_CTRL),
        ],
        keys.CONNECT: [
            ("Tick.outputs:tick", "PubJS.inputs:execIn"),
            ("Tick.outputs:tick", "SubJC.inputs:execIn"),
            ("Tick.outputs:tick", "WriteJC.inputs:execIn"),
            ("SimTime.outputs:simulationTime", "PubJS.inputs:timeStamp"),
            ("SubJC.outputs:jointNames", "WriteJC.inputs:jointNames"),
            ("SubJC.outputs:positionCommand", "WriteJC.inputs:positionCommand"),
            ("SubJC.outputs:velocityCommand", "WriteJC.inputs:velocityCommand"),
            ("SubJC.outputs:effortCommand", "WriteJC.inputs:effortCommand"),
        ],
        keys.SET_VALUES: [
            ("WriteJC.inputs:robotPath", ARTICULATION_PATH),
            ("PubJS.inputs:topicName", "/joint_states"),
            ("SubJC.inputs:topicName", "/joint_command"),
            ("PubJS.inputs:targetPrim", [usdrt.Sdf.Path(ARTICULATION_PATH)]),
        ],
    },
)
# ---- ClockGraph (/clock publisher) --------------------------------------------
og.Controller.edit(
    {"graph_path": "/World/ClockGraph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("Tick", NT_TICK),
            ("SimTime", NT_SIM_TIME),
            ("PubClock", NT_PUB_CLOCK),
        ],
        keys.CONNECT: [
            ("Tick.outputs:tick", "PubClock.inputs:execIn"),
            ("SimTime.outputs:simulationTime", "PubClock.inputs:timeStamp"),
        ],
        keys.SET_VALUES: [
            ("PubClock.inputs:topicName", "/clock"),
        ],
    },
)
# ---- ZED CameraGraph (stereo RGB + camera_info + left depth + left pointcloud) -
# topic 이름은 zed-ros-wrapper 의 표준과 일치 (실로봇 ↔ 시뮬 코드 공유).
og.Controller.edit(
    {"graph_path": "/World/ZedCameraGraph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("Tick", NT_TICK),
            ("RpLeft", NT_CREATE_RP),
            ("RpRight", NT_CREATE_RP),
            ("LeftRgb", NT_CAM),
            ("LeftInfo", NT_CAMINFO),
            ("RightRgb", NT_CAM),
            ("RightInfo", NT_CAMINFO),
            ("LeftDepth", NT_CAM),
            ("LeftPcl", NT_CAM),
        ],
        keys.CONNECT: [
            ("Tick.outputs:tick", "RpLeft.inputs:execIn"),
            ("Tick.outputs:tick", "RpRight.inputs:execIn"),
            # render product → 각 helper 의 execIn + renderProductPath
            ("RpLeft.outputs:execOut", "LeftRgb.inputs:execIn"),
            ("RpLeft.outputs:execOut", "LeftInfo.inputs:execIn"),
            ("RpLeft.outputs:execOut", "LeftDepth.inputs:execIn"),
            ("RpLeft.outputs:execOut", "LeftPcl.inputs:execIn"),
            ("RpRight.outputs:execOut", "RightRgb.inputs:execIn"),
            ("RpRight.outputs:execOut", "RightInfo.inputs:execIn"),
            ("RpLeft.outputs:renderProductPath", "LeftRgb.inputs:renderProductPath"),
            ("RpLeft.outputs:renderProductPath", "LeftInfo.inputs:renderProductPath"),
            ("RpLeft.outputs:renderProductPath", "LeftDepth.inputs:renderProductPath"),
            ("RpLeft.outputs:renderProductPath", "LeftPcl.inputs:renderProductPath"),
            ("RpRight.outputs:renderProductPath", "RightRgb.inputs:renderProductPath"),
            ("RpRight.outputs:renderProductPath", "RightInfo.inputs:renderProductPath"),
        ],
        keys.SET_VALUES: [
            # render product (left + right)
            ("RpLeft.inputs:cameraPrim", [Sdf.Path(LEFT_CAMERA_PATH)]),
            ("RpLeft.inputs:width", ZED_IMAGE_W),
            ("RpLeft.inputs:height", ZED_IMAGE_H),
            ("RpRight.inputs:cameraPrim", [Sdf.Path(RIGHT_CAMERA_PATH)]),
            ("RpRight.inputs:width", ZED_IMAGE_W),
            ("RpRight.inputs:height", ZED_IMAGE_H),
            # left RGB + camera_info
            ("LeftRgb.inputs:type", "rgb"),
            ("LeftRgb.inputs:topicName", "/zed/zed_node/left/image_rect_color"),
            ("LeftRgb.inputs:frameId", "zed_left_camera_frame"),
            ("LeftInfo.inputs:topicName", "/zed/zed_node/left/camera_info"),
            ("LeftInfo.inputs:frameId", "zed_left_camera_frame"),
            # right RGB + camera_info
            ("RightRgb.inputs:type", "rgb"),
            ("RightRgb.inputs:topicName", "/zed/zed_node/right/image_rect_color"),
            ("RightRgb.inputs:frameId", "zed_right_camera_frame"),
            ("RightInfo.inputs:topicName", "/zed/zed_node/right/camera_info"),
            ("RightInfo.inputs:frameId", "zed_right_camera_frame"),
            # left depth (32FC1) + pointcloud (PointCloud2, depth_pcl helper)
            ("LeftDepth.inputs:type", "depth"),
            ("LeftDepth.inputs:topicName", "/zed/zed_node/depth/depth_registered"),
            ("LeftDepth.inputs:frameId", "zed_left_camera_frame"),
            ("LeftPcl.inputs:type", "depth_pcl"),
            ("LeftPcl.inputs:topicName", "/zed/zed_node/point_cloud/cloud_registered"),
            ("LeftPcl.inputs:frameId", "zed_left_camera_frame"),
        ],
    },
)
print("[OK] ROS2 OmniGraph: TFGraph + JointGraph + ClockGraph + ZedCameraGraph 생성")

print("=" * 60)
print("Isaac Sim RB10 씬 준비 완료 (Step 1~5 + EOAT)")
print(f"  RB10:        link0=(0,0,0), articulation={ARTICULATION_PATH}")
print(f"  Table:       center={TABLE_CENTER.tolist()} size={TABLE_SIZE.tolist()}")
print(f"  Steel plate: center={PLATE_CENTER.tolist()} size={PLATE_SIZE.tolist()}")
print(f"  Wall:        center={WALL_CENTER.tolist()} size={WALL_SIZE.tolist()}")
print(f"  Camera:      {CAMERA_PATH} eye=(0.2,0.5,0.915) → target=(0,-0.78,0.5)")
print(f"  Mount seg1:  center={MOUNT_SEG1_CENTER.tolist()} size={MOUNT_SEG1_SIZE.tolist()}")
print(f"  Mount seg2:  center={MOUNT_SEG2_CENTER.tolist()} size={MOUNT_SEG2_SIZE.tolist()}")
print(f"  Roller:      tcp local +{ROLLER_AXIS} offset={ROLLER_OFFSET}m, Φ{ROLLER_RADIUS*2}m × {ROLLER_LENGTH}m")
print(f"  ROS topics:  /joint_states, /joint_command, /tf, /clock,")
print(f"               /zed/zed_node/{{left,right}}/{{image_rect_color,camera_info}},")
print(f"               /zed/zed_node/depth/depth_registered,")
print(f"               /zed/zed_node/point_cloud/cloud_registered")
print(f"  READY_POSE 적용: 롤러가 -Y (벽) 향해야 정상")
print("=" * 60)
