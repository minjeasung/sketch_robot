"""
Isaac Sim 5.1 — RB10-1300E_U 시뮬레이션 환경 (Phase 4 Session 4 셋업)

좌표계: world / World = 실제 RB10 pendant/base = (0, 0, 0), 단위 미터.
rbpodo URDF 의 link0 는 실제 base 와 Z축 기준 90도 차이가 있어 RB10 USD root 에
+90도 보정을 적용한다.
기존 환경 측정값은 READY_POSE 의 TCP 축으로 기록되어 있었으므로, 본 파일에서
base 좌표로 변환해 배치한다:
  tcp x = base y, tcp y = base -x, tcp z = base z
환경 측정값 출처: docs/phase4_session4_environment.md (2026-05-12)

실행: (ROS2 bridge 환경 필요 — run_isaac_sim.sh 와 동일 LD_LIBRARY_PATH 셋업)
    source ~/isaac_env/bin/activate
    isaacsim --exec ~/sketch_robot_ws/src/sketch_control/sketch_control/isaac_sim_rb10.py

씬 구성:
  - RB10 USD (URDF link0 를 World/real base 에 대해 +90도 보정)
  - 테이블 (base 0.6×0.8×0.813 m, z ∈ [-0.823, -0.010])
  - 철판 (0.2×0.2×0.010 m, z ∈ [-0.010, 0])
  - 벽 (base x=+0.80 평면, 2.0×1.5 m, 두께 0.02m)
  - ZED 카메라 prim (base -0.5, 0.2, 1.0) → 벽 향함
  - ground plane (z=-0.823, 실제 바닥 — 테이블 다리 끝)

퍼블리시 (OmniGraph): /joint_states, /tf
구독 (OmniGraph): /joint_command
"""
import os
import numpy as np
import time as _time
import struct
import omni
import omni.kit.app
import omni.graph.core as og
from pxr import UsdGeom, UsdPhysics, Gf, Sdf, Usd, PhysxSchema
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


def _tcp_authored_xyz_to_base(v):
    """READY_POSE TCP 축으로 기록한 좌표를 RB10 base(link0) 좌표로 변환."""
    x_tcp, y_tcp, z_tcp = np.asarray(v, dtype=float)
    return np.array([-y_tcp, x_tcp, z_tcp], dtype=float)


def _tcp_authored_box_size_to_base(v):
    """축 정렬 box size: tcp X/Y 축이 base Y/-X 로 바뀌므로 X/Y 크기를 교환."""
    sx_tcp, sy_tcp, sz_tcp = np.asarray(v, dtype=float)
    return np.array([sy_tcp, sx_tcp, sz_tcp], dtype=float)

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

# 실제 RB10 pendant/base 좌표와 rbpodo URDF link0 는 Z축 기준 90도 차이가 있다.
# 같은 조인트값에서 URDF FK 결과를 +90도 회전하면 pendant TCP 위치와 일치한다.
# 따라서 Isaac World 는 실제 base 로 두고, URDF/USD 로봇 root 만 +90도 보정한다.
_rb10_root_prim = stage.GetPrimAtPath(RB10_PRIM_PATH)
_rb10_xform = UsdGeom.Xformable(_rb10_root_prim)
for _op in _rb10_xform.GetOrderedXformOps():
    _rb10_xform.GetPrim().RemoveProperty(_op.GetOpName())
_rb10_xform.AddRotateZOp().Set(90.0)
print(f"[OK] RB10 USD root Z축 +90° 회전 적용 (World=real base, link0=URDF base)")

# ---- Joint drive 강제 적용 (URDF→USD 변환 시 drive 누락 가능) ------------------
# 모든 revolute joint 에 UsdPhysics.DriveAPI 추가. stiffness/damping 미설정 시
# articulation 의 position drive 가 0 토크로 시뮬되어 중력에 굴복.
JOINT_STIFFNESS = float(os.environ.get("RB10_SIM_JOINT_STIFFNESS", "6000.0"))
JOINT_DAMPING = float(os.environ.get("RB10_SIM_JOINT_DAMPING", "600.0"))
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
# 기존 측정값은 TCP축 기준이므로 base 좌표로 변환해 배치한다.
# TCP 기준 테이블: x∈[-0.35,+0.45] y∈[-0.20,+0.40] z∈[-0.823,-0.010]
TABLE_SIZE = _tcp_authored_box_size_to_base([0.80, 0.60, 0.813])
TABLE_CENTER = _tcp_authored_xyz_to_base([0.05, 0.10, -0.4165])
# 철판: TCP 기준 x/y 200mm 정사각형이라 base 변환 후에도 size 동일.
PLATE_SIZE = _tcp_authored_box_size_to_base([0.20, 0.20, 0.010])
PLATE_CENTER = _tcp_authored_xyz_to_base([0.0, 0.0, -0.005])

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

# ---- 벽 + 작업 영역 (Phase 5 일감 2.3) -----------------------------------------
# 큰 벽 plane (2.0 × 1.5m, 흰색) + 노란 마스킹 테이프 4 strip 으로 0.5 × 0.4m
# 작업 영역 outline. RANSAC plane 인식과 sketch projection 의 시각 baseline.
# TCP 기준 벽 표면 y=-0.8 은 base 기준 x=+0.8 이다.
# 벽 front surface normal 은 base -X 방향(RB10/자유공간 방향).
WALL_SIZE = _tcp_authored_box_size_to_base([2.0, 0.02, 1.5])
WALL_CENTER = _tcp_authored_xyz_to_base([0.0, -0.81, 0.5])
_WALL_FRONT_X = float(WALL_CENTER[0] - WALL_SIZE[0] / 2.0)   # +0.80

world.scene.add(VisualCuboid(
    prim_path="/World/wall",
    name="wall",
    position=WALL_CENTER,
    scale=WALL_SIZE,
    color=np.array([0.92, 0.92, 0.92]),                      # 밝은 회색
))
print(f"[OK] 벽 추가: center={WALL_CENTER.tolist()} size={WALL_SIZE.tolist()} "
      f"front_x={_WALL_FRONT_X}")

# 노란 마스킹 테이프 (4 strip, 작업 영역 0.5×0.4 outline). 벽 표면 -X 1mm 앞에
# 띄워 z-fighting 회피. tape 폭 0.02m.
WORK_AREA_W = 0.5                                            # base y 방향 (가로)
WORK_AREA_H = 0.4                                            # z 방향 (세로)
WORK_AREA_CENTER = np.array([_WALL_FRONT_X - 0.001, 0.0, 0.5])
TAPE_W = 0.02
_TAPE_COLOR = np.array([1.0, 0.85, 0.0])                     # 선명한 노랑
_TAPE_X = float(WORK_AREA_CENTER[0])                         # 벽 surface - 1mm

# Outline outer corners: y = ±(WORK_AREA_W/2), z = WORK_AREA_CENTER[2] ± (WORK_AREA_H/2).
_y_outer = float(WORK_AREA_W / 2.0)                          # 0.25
_z_top   = float(WORK_AREA_CENTER[2] + WORK_AREA_H / 2.0)    # 0.70
_z_bot   = float(WORK_AREA_CENTER[2] - WORK_AREA_H / 2.0)    # 0.30

_tape_strips = [
    # name, center, scale  (tape thickness in X = 0.001 — paper-thin)
    ("Top",    np.array([_TAPE_X, 0.0, _z_top - TAPE_W/2]),
                np.array([0.001, WORK_AREA_W, TAPE_W])),
    ("Bottom", np.array([_TAPE_X, 0.0, _z_bot + TAPE_W/2]),
                np.array([0.001, WORK_AREA_W, TAPE_W])),
    ("Left",   np.array([_TAPE_X, -_y_outer + TAPE_W/2, WORK_AREA_CENTER[2]]),
                np.array([0.001, TAPE_W, WORK_AREA_H])),
    ("Right",  np.array([_TAPE_X,  _y_outer - TAPE_W/2, WORK_AREA_CENTER[2]]),
                np.array([0.001, TAPE_W, WORK_AREA_H])),
]
for _name, _center, _scale in _tape_strips:
    world.scene.add(VisualCuboid(
        prim_path=f"/World/MaskingTape_{_name}",
        name=f"masking_tape_{_name.lower()}",
        position=_center,
        scale=_scale,
        color=_TAPE_COLOR,
    ))
print(f"[OK] 마스킹 테이프 4 strip: outline {WORK_AREA_W}m × {WORK_AREA_H}m "
      f"@ z=[{_z_bot}, {_z_top}], tape_w={TAPE_W}m")

# Ground truth: 작업 영역 4 outer corners (벽 surface x=_WALL_FRONT_X).
WORK_AREA_CORNERS = [
    ("tl", [_WALL_FRONT_X, -_y_outer, _z_top]),
    ("tr", [_WALL_FRONT_X,  _y_outer, _z_top]),
    ("bl", [_WALL_FRONT_X, -_y_outer, _z_bot]),
    ("br", [_WALL_FRONT_X,  _y_outer, _z_bot]),
]
for _cid, _w in WORK_AREA_CORNERS:
    print(f"[DIAG] work_area corner {_cid} = ({_w[0]:.4f}, {_w[1]:.4f}, {_w[2]:.4f})")

# ---- 카메라 마운트 (ㄴ자 알루미늄 프레임, 50×50mm 단면) ------------------------
# 세그먼트1: 수평 (책상 위, base X 방향으로 누움), 책상 위에 얹힘
# 세그먼트2: 수직 (세그먼트1 위에서 솟음)
# 둘이 합쳐 ㄴ자 — z 적층 합 0.9m
MOUNT_SEG1_CENTER = _tcp_authored_xyz_to_base([0.2, 0.35, 0.015])
MOUNT_SEG1_SIZE = _tcp_authored_box_size_to_base([0.05, 0.3, 0.05])    # z ∈ [-0.01, 0.04]
MOUNT_SEG2_CENTER = _tcp_authored_xyz_to_base([0.2, 0.5, 0.465])
MOUNT_SEG2_SIZE = _tcp_authored_box_size_to_base([0.05, 0.05, 0.85])   # z ∈ [0.04, 0.89]
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

# 카메라 mount: ball head + 1점 볼트 (실로봇 reference 사진과 1:1 일치).
# 체인 — MountSeg2 top → ball head (sphere) → 볼트 (cylinder) → ZED 바닥.
# 실 ZED X 는 보통 swivel head 로 부착, 1/4" 볼트 1점 → 시뮬도 동일 chain 으로
# hand-eye calib 시 sim/real transform parameterization 일치.
MOUNT_BALL_CENTER = _tcp_authored_xyz_to_base([0.2, 0.5, 0.92])      # Seg2 top (0.89) 위 +0.03m
MOUNT_BALL_RADIUS = 0.025                            # Φ50mm
_MOUNT_BALL_COLOR = (0.08, 0.08, 0.08)               # 검은 ball head
_ball_sphere = UsdGeom.Sphere.Define(stage, "/World/CameraMount_Ballhead")
_ball_sphere.CreateRadiusAttr(float(MOUNT_BALL_RADIUS))
_ball_sphere.CreateDisplayColorAttr([Gf.Vec3f(*_MOUNT_BALL_COLOR)])
UsdGeom.Xformable(_ball_sphere.GetPrim()).AddTranslateOp().Set(
    Gf.Vec3d(*MOUNT_BALL_CENTER)
)
print(f"[OK] 카메라 마운트 ball head: "
      f"center={MOUNT_BALL_CENTER.tolist()} radius={MOUNT_BALL_RADIUS}")

# 1/4" 볼트 (시각용, ball top 과 카메라 바닥 사이 연결)
MOUNT_BOLT_CENTER = _tcp_authored_xyz_to_base([0.2, 0.5, 0.95])      # ball top (0.945) 위 +0.005m
MOUNT_BOLT_RADIUS = 0.004                            # Φ8mm
MOUNT_BOLT_HEIGHT = 0.01
_MOUNT_BOLT_COLOR = (0.55, 0.55, 0.60)              # 금속 회색
_bolt_cyl = UsdGeom.Cylinder.Define(stage, "/World/CameraMount_Bolt")
_bolt_cyl.CreateRadiusAttr(float(MOUNT_BOLT_RADIUS))
_bolt_cyl.CreateHeightAttr(float(MOUNT_BOLT_HEIGHT))
_bolt_cyl.CreateAxisAttr("Z")
_bolt_cyl.CreateDisplayColorAttr([Gf.Vec3f(*_MOUNT_BOLT_COLOR)])
UsdGeom.Xformable(_bolt_cyl.GetPrim()).AddTranslateOp().Set(
    Gf.Vec3d(*MOUNT_BOLT_CENTER)
)
print(f"[OK] 카메라 마운트 볼트: center={MOUNT_BOLT_CENTER.tolist()} "
      f"radius={MOUNT_BOLT_RADIUS} h={MOUNT_BOLT_HEIGHT}")

# ---- 초기화 -------------------------------------------------------------------
world.reset()

# ---- ZED 카메라 (zed-isaac-sim 의 ZED_X.usdc reference) -----------------------
# ZED_X.usdc 를 reference 하는 이유: 시각적 sim-to-real fidelity (실 ZED X mesh +
# CameraLeft/CameraRight intrinsic + IMU prim 모두 검증된 ZED official asset).
# Phase 5 옵션 C 부터는 sl.sensor.camera ZED_Camera Helper (IPC streamer) 사용 X —
# 아래 ZedROS2Graph 가 Isaac Sim native ROS2CameraHelper 로 wrapper 와 동일한
# topic/frame/intrinsic 발행. 시각 자산만 활용.
#
# 위치: world.reset() 다음. 이유 — reset 전에 USD reference 로 새 rigid body 가 추가되면
# Robot articulation 의 simulation view 가 invalidate 되어 무한 에러 → Isaac Sim crash.
CAMERA_PATH = "/World/SketchCamera"
# ZED X USD 의 origin 은 base_link 와 일치 — [DIAG] 가 (0.97 의도 → 0.97 실제)
# 확인 (이전 run). 그러나 본체 mesh 들이 origin 아래로 ~0.04m 뻗음 (이전 viewport
# 에서 어댑터 plate top (z=0.93) 까지 관통). 따라서 mesh 아래 extent 보다 더
# 큰 buffer 필요 — bolt top 위로 0.045m 잡음.
_BALL_TOP_Z = float(MOUNT_BALL_CENTER[2] + MOUNT_BALL_RADIUS)             # 0.945
_BOLT_TOP_Z = float(MOUNT_BOLT_CENTER[2] + MOUNT_BOLT_HEIGHT / 2.0)       # 0.955
CAMERA_EYE = Gf.Vec3d(
    float(MOUNT_BALL_CENTER[0]),            # -0.5
    float(MOUNT_BALL_CENTER[1]),            # 0.2
    _BOLT_TOP_Z + 0.045,                    # 1.0 (bolt top + 본체 mesh extent buffer)
)
CAMERA_TARGET = Gf.Vec3d(_WALL_FRONT_X, 0.0, 0.5)
ZED_X_USD_PATH = (
    "/home/minjea/sketch_robot_ws/zed-isaac-sim/"
    "exts/sl.sensor.camera/data/usd/ZED_X.usdc"
)

# CAMERA_PATH 를 Xform 으로 만들고 ZED_X.usdc 를 reference. defaultPrim 이 child 가 됨.
zed_carrier = UsdGeom.Xform.Define(stage, CAMERA_PATH)
zed_carrier.GetPrim().GetReferences().AddReference(ZED_X_USD_PATH)

# ---- ZED X 의 world pose (look-at) 를 USD reference 직후 먼저 적용 ----------------
# 이유: FixedJoint 가 ZED root 의 world pose 를 기준으로 anchor 계산. transform 이
# 나중에 적용되면 joint 가 origin 으로 끌어당겨 받침대 한가운데에 박힘.
from scipy.spatial.transform import Rotation as _R


def _body_look_at_quaternion(eye, target, up_axis=(0.0, 0.0, 1.0)):
    """REP 103 body convention (+X=fwd, +Y=left, +Z=up) 의 +X 가 target 향하도록.
    반환: Gf.Quatf(w, x, y, z)."""
    eye_np = np.array([eye[0], eye[1], eye[2]], dtype=float)
    target_np = np.array([target[0], target[1], target[2]], dtype=float)
    fwd = target_np - eye_np
    fwd /= np.linalg.norm(fwd)
    up = np.asarray(up_axis, dtype=float)
    left = np.cross(up, fwd)
    left /= np.linalg.norm(left)
    new_up = np.cross(fwd, left)
    R_mat = np.column_stack([fwd, left, new_up])
    qx, qy, qz, qw = _R.from_matrix(R_mat).as_quat()
    return Gf.Quatf(float(qw), float(qx), float(qy), float(qz))


_q_lookat = _body_look_at_quaternion(CAMERA_EYE, CAMERA_TARGET)
_qd_lookat = Gf.Quatd(float(_q_lookat.real),
                      float(_q_lookat.imaginary[0]),
                      float(_q_lookat.imaginary[1]),
                      float(_q_lookat.imaginary[2]))

zed_xf = UsdGeom.Xformable(zed_carrier.GetPrim())
for _op in zed_xf.GetOrderedXformOps():
    zed_xf.GetPrim().RemoveProperty(_op.GetOpName())
_M = Gf.Matrix4d(1.0)
_M.SetRotateOnly(Gf.Rotation(_qd_lookat))
_M.SetTranslateOnly(Gf.Vec3d(CAMERA_EYE[0], CAMERA_EYE[1], CAMERA_EYE[2]))
zed_xf.AddTransformOp().Set(_M)
print(f"[OK] ZED X USD reference + transform: {CAMERA_PATH}")
print(f"     eye={tuple(CAMERA_EYE)} → target={tuple(CAMERA_TARGET)}")

# ---- 진단: ZED 내부 prim 들의 실제 world 위치 ---------------------------------
# ZED_X.usdc 의 base_link 가 자체 xformOp 을 갖고 있을 수 있음. 의도한 CAMERA_EYE
# 와 실제 base_link world 위치 차이 = ZED USD 내부 origin offset → 시각적 박힘
# 원인 추적용. (의도값 ≠ 실제값이면 ZED 내부 transform 추가 보정 필요.)
for _diag_path in [
    CAMERA_PATH,
    CAMERA_PATH + "/base_link",
    CAMERA_PATH + "/base_link/ZED_X",
]:
    _diag_prim = stage.GetPrimAtPath(_diag_path)
    if _diag_prim.IsValid() and _diag_prim.IsA(UsdGeom.Xformable):
        _world_xf = UsdGeom.Xformable(_diag_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        _world_t = _world_xf.ExtractTranslation()
        print(f"[DIAG] {_diag_path} world pos = "
              f"({_world_t[0]:.4f}, {_world_t[1]:.4f}, {_world_t[2]:.4f})")

# ---- ZED X rigid body: dynamic + disableGravity --------------------------------
# IMU 는 dynamic body 에서만 sensor reading 유효. 중력 차단은 아래 fixed joint 와
# 함께 작용 (joint 가 위치 고정 + gravity off 가 외력 차단).
_zed_rb_paths = []
for _descendant in Usd.PrimRange(zed_carrier.GetPrim()):
    if _descendant.HasAPI(UsdPhysics.RigidBodyAPI):
        _rb_api = UsdPhysics.RigidBodyAPI(_descendant)
        _rb_api.GetKinematicEnabledAttr().Set(False)         # dynamic
        _physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(_descendant)
        _physx_rb.CreateDisableGravityAttr().Set(True)       # gravity off
        _zed_rb_paths.append(_descendant.GetPath().pathString)
print(f"[OK] ZED X rigid body: {len(_zed_rb_paths)} prim (dynamic + disableGravity)")
print(f"     paths={_zed_rb_paths}")

# ---- ZED X IMU prim 재등록 (kit command — sensor backend 등록 필수) -------------
# ZED_X.usdc 의 Imu_Sensor 는 IsaacImuSensor typed prim 으로 USD 안에 정의됨.
# 그러나 C++ sensor backend (acquire_imu_sensor_interface) 는 USD load 만으론
# 등록 안 함 — IsaacSensorCreateImuSensor kit command 가 호출돼야 internal
# registration 됨 → 이전 run 의 "no valid sensor reading" 원인.
# 해결: 기존 prim 의 pose 보존하면서 kit command 로 재생성.
import omni.kit.commands  # noqa: E402
IMU_PARENT_PATH = CAMERA_PATH + "/base_link/ZED_X"
IMU_PRIM_PATH = IMU_PARENT_PATH + "/Imu_Sensor"
try:
    _old_imu_prim = stage.GetPrimAtPath(IMU_PRIM_PATH)
    _imu_t = Gf.Vec3d(0.0, 0.0, 0.0)
    _imu_q = Gf.Quatd(1.0, 0.0, 0.0, 0.0)
    if _old_imu_prim.IsValid():
        # 기존 prim 의 translate/orient 보존 (ZED_X.usdc 의 IMU 위치/자세)
        _t_attr = _old_imu_prim.GetAttribute("xformOp:translate")
        _q_attr = _old_imu_prim.GetAttribute("xformOp:orient")
        if _t_attr and _t_attr.HasAuthoredValue():
            _tv = _t_attr.Get()
            _imu_t = Gf.Vec3d(float(_tv[0]), float(_tv[1]), float(_tv[2]))
        if _q_attr and _q_attr.HasAuthoredValue():
            _qv = _q_attr.Get()
            _imu_q = Gf.Quatd(float(_qv.GetReal()),
                              float(_qv.GetImaginary()[0]),
                              float(_qv.GetImaginary()[1]),
                              float(_qv.GetImaginary()[2]))
        stage.RemovePrim(IMU_PRIM_PATH)                       # 기존 USD prim 제거
        print(f"[OK] 기존 IMU prim 제거: {IMU_PRIM_PATH}")
        print(f"     pose 보존: t={tuple(_imu_t)} q={_imu_q}")
    # kit command 로 재생성 → C++ sensor backend 에 internal 등록 트리거
    _ok, _new_imu_prim = omni.kit.commands.execute(
        "IsaacSensorCreateImuSensor",
        path="/Imu_Sensor",
        parent=IMU_PARENT_PATH,
        sensor_period=1.0 / 60.0,                             # physics dt (60 Hz)
        translation=_imu_t,
        orientation=_imu_q,
        linear_acceleration_filter_size=1,
        angular_velocity_filter_size=1,
        orientation_filter_size=1,
    )
    if _ok and _new_imu_prim:
        print(f"[OK] IMU prim 재생성 (kit command): {IMU_PRIM_PATH}")
        print(f"     sensorPeriod={1.0/60.0:.5f}s (60 Hz), filterWidth=1")
    else:
        print(f"[ERROR] IsaacSensorCreateImuSensor 실패: ok={_ok}")
except Exception as _e:
    print(f"[ERROR] IMU 재생성 실패: {_e}")

# ---- ZED X 를 MountBallhead (ball head sphere) 에 fixed joint 로 anchoring --------
# 체인: World ← (kinematic) MountBallhead ← (FixedJoint) → ZED /Root.
# Sim-to-real: 실 ZED 의 swivel head ball + 1/4" 볼트 1점 부착과 1:1 모사.
# localPos0 = CAMERA_EYE - MOUNT_BALL_CENTER (Ball local frame, identity rot).
_MOUNT_BALL_PATH = "/World/CameraMount_Ballhead"
_ball_prim = stage.GetPrimAtPath(_MOUNT_BALL_PATH)
if _ball_prim.IsValid() and _zed_rb_paths:
    # (a) Ball = kinematic rigid body (위치 고정, FixedJoint anchor 역할).
    if not _ball_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(_ball_prim)
    _ball_rb = UsdPhysics.RigidBodyAPI(_ball_prim)
    _ball_rb.CreateRigidBodyEnabledAttr(True)
    _ball_rb.CreateKinematicEnabledAttr(True)
    # (b) FixedJoint with explicit localPos0/localRot0.
    _zed_root_rb_path = _zed_rb_paths[0]                     # 첫 rigid body (= /World/SketchCamera)
    _joint_path = "/World/SketchCamera_Ballhead_FixedJoint"
    if stage.GetPrimAtPath(_joint_path).IsValid():
        stage.RemovePrim(_joint_path)                        # 반복 실행 안전
    _fj = UsdPhysics.FixedJoint.Define(stage, _joint_path)
    _fj.CreateBody0Rel().SetTargets([_MOUNT_BALL_PATH])
    _fj.CreateBody1Rel().SetTargets([_zed_root_rb_path])
    # body0 (Ballhead, rot=identity) local 에서 anchor = (CAMERA_EYE - Ball_center).
    _local_pos0 = Gf.Vec3f(
        float(CAMERA_EYE[0] - MOUNT_BALL_CENTER[0]),
        float(CAMERA_EYE[1] - MOUNT_BALL_CENTER[1]),
        float(CAMERA_EYE[2] - MOUNT_BALL_CENTER[2]),
    )
    _fj.CreateLocalPos0Attr(_local_pos0)
    _fj.CreateLocalRot0Attr(_q_lookat)                       # Ball identity → ZED lookat
    # body1 (ZED root, world pose=CAMERA_EYE+lookat) local 에서 anchor = origin+identity.
    _fj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    _fj.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    print(f"[OK] FixedJoint: {_MOUNT_BALL_PATH} ↔ {_zed_root_rb_path}")
    print(f"     localPos0={tuple(_local_pos0)} (Ball center → ZED 위치)")
else:
    print(f"[WARN] FixedJoint anchoring skip "
          f"(ball valid={_ball_prim.IsValid()}, zed_rb_count={len(_zed_rb_paths)})")

# CameraHelper 가 참조할 left/right camera prim path (ZED_X USD 내부 구조)
LEFT_CAMERA_PATH = CAMERA_PATH + "/base_link/ZED_X/CameraLeft"
RIGHT_CAMERA_PATH = CAMERA_PATH + "/base_link/ZED_X/CameraRight"

try:
    _dofs = robot.num_dof
    _joint_names = robot.dof_names
    print(f"[OK] RB10 articulation 유효: {_dofs} DOF, joints={_joint_names}")
except Exception as _e:
    print(f"[ERROR] RB10 articulation 실패: {_e}")
    print("      → RB10 USD 자체 문제 또는 ArticulationRoot 깨짐")

# ---- WORK_POSE / CALIB_POSE 분리 (Phase 5 일감 2.4) -----------------------------
# 시작 pose 적용 flow (사용자 가이드):
#   1) set_joints_default_state — articulation 의 default 자세
#   2) world.reset() — default state 가 시뮬에 적용 (이미 위에서 호출됨)
#   3) set_joint_positions — 현재 자세 텔레포트
#   4) set_joint_position_targets — drive target 설정 (drive 가 그쪽으로 유지)
#   5) ArticulationController.apply_action(positions=...) — 동일 목적, 다른 API
#   6) set_gains — drive PD gain 조정 (USD drive 가 약할 경우)
# Drive 가 정상이면 위만으로 자세 유지. physics callback 으로 강제 holding 안 함.
#
# WORK_POSE — 실제 RB10 pendant/driver 에서 읽은 작업 시작 자세.
#   moveit_executor.py 의 READY_POSE_JOINTS 와 동일하게 유지한다.
#   weld/sketch 작업 시 사용 pose.
WORK_POSE = {
    "base":     0.0005,   # pendant: +0.03 deg
    "shoulder": -0.9343,  # pendant: -53.53 deg
    "elbow":    2.4246,   # pendant: +138.92 deg
    "wrist1":  -1.6293,   # pendant: -93.35 deg
    "wrist2":   1.5675,   # pendant: +89.81 deg
    "wrist3":   0.0000,
}

# CALIB_POSE — wrist 가동 범위 확인용. 기본 실행은 WORK_POSE 로 시작하고,
# 필요할 때만 preset/service 로 사용한다.
import math as _math_calib                                    # noqa: E402
CALIB_DELTA = ("wrist1", -1)                                  # 첫 시도 — 시각 검증 후 조정
CALIB_POSE = dict(WORK_POSE)
CALIB_POSE[CALIB_DELTA[0]] += CALIB_DELTA[1] * (_math_calib.pi / 2.0)

# 시작 시 적용할 pose — 실제 로봇 작업 시작 자세.
READY_POSE_DICT = WORK_POSE
print(f"[OK] 시작 pose = WORK_POSE (real RB10 coordinates)")
print(f"     WORK_POSE:  base={WORK_POSE['base']:.4f}, "
      f"shoulder={WORK_POSE['shoulder']:.4f}, elbow={WORK_POSE['elbow']:.4f}, "
      f"wrist1={WORK_POSE['wrist1']:.4f}, wrist2={WORK_POSE['wrist2']:.4f}, "
      f"wrist3={WORK_POSE['wrist3']:.4f}")
print(f"     CALIB_POSE: wrist1={CALIB_POSE['wrist1']:.4f}, "
      f"wrist2={CALIB_POSE['wrist2']:.4f}, wrist3={CALIB_POSE['wrist3']:.4f}")

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

# ---- AFT200 + 페인트 롤러 EOAT (tcp 자식 prim 으로 부착) -----------------------
# 체인: tcp -> AFT200 -> roller. 제공 CAD 는 +Z 방향으로 뻗지만 실제 장착은
# TCP local -Y 방향이다. URDF link0 를 실제 base 로 +90도 보정하면 TCP local -Y
# 가 base +X, 즉 벽/작업면 방향을 향한다.

TOOL_AXIS = "-Y"
AFT200_LENGTH = 0.0522
AFT200_STL_PATH = "/home/minjea/Downloads/aft200_description/meshes/visual/aft200.stl"
AFT200_SIZE = (0.104, AFT200_LENGTH, 0.082)  # TCP frame box size: x, y, z
AFT200_CENTER = (-0.0116, -AFT200_LENGTH / 2.0, 0.0)

EOAT_STL_PATH = "/home/minjea/sketch_robot_ws/src/eoat_description/meshes/rr_00a_b_eoat_no_camera_collision.stl"
EOAT_MESH_FORWARD_LENGTH = 0.2305
EOAT_MESH_CENTER_OFFSET = (
    0.0,
    -(AFT200_LENGTH + EOAT_MESH_FORWARD_LENGTH / 2.0),
    0.0,
)

ROLLER_FORWARD_REACH = 0.209475
ROLLER_SUPPORT_RADIUS = 0.012
ROLLER_LENGTH = 0.18    # 축 방향
ROLLER_RADIUS = 0.025   # Φ50mm
ROLLER_LONG_AXIS = "+X"
EOAT_TIP_OFFSET = AFT200_LENGTH + ROLLER_FORWARD_REACH

# Intel RealSense D405 official description (realsense2_description r/4.58.2).
# Camera frame +X is its front direction; install it as TCP local -Y.
D405_STL_PATH = "/home/minjea/sketch_robot_ws/src/vendor/realsense-ros/realsense2_description/meshes/d405.stl"
D405_SIZE = (0.042, 0.023, 0.042)  # TCP frame bbox: x(width), y(depth), z(height)
# Position from the removed D4xx camera volume in rr_00a_b_eoat_no_camera metadata.
# Keep D405 optical/front direction aligned with EOAT forward, but sit it on the bracket.
D405_LINK_POSITION = (0.00905, -0.07640, 0.04375)
D405_COLLISION_CENTER = (0.0, -0.06870, 0.04375)
D405_MESH_SCALE = 0.001
D405_VISUAL_ORIGIN = np.array([0.0038, -0.009, 0.0], dtype=float)
D405_VISUAL_RPY = (np.pi / 2.0, 0.0, np.pi / 2.0)
D405_CAMERA_RPY_IN_TCP = (0.0, 0.0, -np.pi / 2.0)


def _axis_offset_local(axis, distance):
    sign = -1.0 if axis.startswith("-") else 1.0
    letter = axis.lstrip("+-").upper()
    out = [0.0, 0.0, 0.0]
    out[{"X": 0, "Y": 1, "Z": 2}[letter]] = sign * distance
    return out


_axis_letter = TOOL_AXIS.lstrip("+-").upper()
_roller_long_axis = ROLLER_LONG_AXIS.lstrip("+-").upper()
_support_mid_v = _axis_offset_local(
    TOOL_AXIS, AFT200_LENGTH + ROLLER_FORWARD_REACH / 2.0)
_roller_center_v = _axis_offset_local(TOOL_AXIS, EOAT_TIP_OFFSET)


def _cad_z_to_tcp_minus_y(x, y, z):
    """CAD +Z 방향을 TCP local -Y 로 회전한다. Rx(+90deg): (x, y, z)->(x, -z, y)."""
    return (float(x), float(-z), float(y))


def _rpy_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


_D405_VISUAL_R = _rpy_matrix(*D405_VISUAL_RPY)
_D405_CAMERA_R = _rpy_matrix(*D405_CAMERA_RPY_IN_TCP)
_D405_LINK_P = np.array(D405_LINK_POSITION, dtype=float)


def _d405_stl_to_tcp_frame(x, y, z):
    """Official D405 STL(mm) -> RealSense camera_link -> TCP frame."""
    p_camera_link = _D405_VISUAL_R @ (np.array([x, y, z], dtype=float) * D405_MESH_SCALE)
    p_camera_link += D405_VISUAL_ORIGIN
    p_tcp = _D405_CAMERA_R @ p_camera_link + _D405_LINK_P
    return (float(p_tcp[0]), float(p_tcp[1]), float(p_tcp[2]))


def _read_stl_points_transformed(stl_path, transform_vertex):
    """STL vertex 를 읽고 transform_vertex(x,y,z) 결과를 USD point 로 만든다."""
    with open(stl_path, "rb") as f:
        data = f.read()

    points = []
    face_counts = []
    face_indices = []

    def _append_vertex(x, y, z):
        points.append(Gf.Vec3f(*transform_vertex(x, y, z)))
        return len(points) - 1

    if len(data) >= 84:
        tri_count = struct.unpack("<I", data[80:84])[0]
        expected_len = 84 + tri_count * 50
        if expected_len == len(data):
            off = 84
            for _ in range(tri_count):
                off += 12  # normal
                ids = []
                for _v in range(3):
                    x, y, z = struct.unpack("<fff", data[off:off + 12])
                    off += 12
                    ids.append(_append_vertex(x, y, z))
                off += 2
                face_counts.append(3)
                face_indices.extend(ids)
            return points, face_counts, face_indices

    text = data.decode(errors="ignore")
    tri = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("vertex "):
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        tri.append(_append_vertex(float(parts[1]), float(parts[2]), float(parts[3])))
        if len(tri) == 3:
            face_counts.append(3)
            face_indices.extend(tri)
            tri = []
    return points, face_counts, face_indices


def _define_stl_mesh(
    stage,
    prim_path,
    stl_path,
    color,
    translate=(0.0, 0.0, 0.0),
    transform_vertex=_cad_z_to_tcp_minus_y,
):
    points, face_counts, face_indices = _read_stl_points_transformed(
        stl_path, transform_vertex)
    if not points:
        raise RuntimeError(f"STL vertex 없음: {stl_path}")
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    zs = [float(p[2]) for p in points]
    mesh.CreateExtentAttr([
        Gf.Vec3f(min(xs), min(ys), min(zs)),
        Gf.Vec3f(max(xs), max(ys), max(zs)),
    ])
    if any(abs(float(v)) > 1e-12 for v in translate):
        UsdGeom.Xformable(mesh.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*translate))
    return mesh

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

    # 기존 prim 있으면 제거 (반복 실행 안전). apriltag 는 이전 버전 잔여물 정리용.
    for _old in [
        "paint_roller", "roller_rod", "aft200", "roller_support",
        "rr_00a_b_eoat", "d405", "realsense_d405", "apriltag",
    ]:
        _pp = TCP_PRIM_PATH + "/" + _old
        if stage.GetPrimAtPath(_pp).IsValid():
            stage.RemovePrim(_pp)

    # AFT200 sensor body. 실제 STL visual 을 TCP 기준에 붙이고, CAD +Z 를 TCP -Y 로 회전.
    _aft_path = TCP_PRIM_PATH + "/aft200"
    try:
        _define_stl_mesh(stage, _aft_path, AFT200_STL_PATH, (0.32, 0.34, 0.38))
    except Exception as _aft_e:
        print(f"[WARN] AFT200 STL 로드 실패 → bbox proxy 사용: {_aft_e}")
        _aft = UsdGeom.Cube.Define(stage, _aft_path)
        _aft.CreateSizeAttr(1.0)
        _aft.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.20, 0.24)])
        _aft_xf = UsdGeom.Xformable(_aft.GetPrim())
        _aft_xf.AddTranslateOp().Set(Gf.Vec3d(*AFT200_CENTER))
        _aft_xf.AddScaleOp().Set(Gf.Vec3f(*AFT200_SIZE))

    # RR-00A_B EOAT no-camera. mesh origin 은 bbox center 이므로, 가까운 면이
    # AFT200 출력면(TCP local y=-0.0522)에 오도록 center 를 -Y 방향으로 이동한다.
    _eoat_path = TCP_PRIM_PATH + "/rr_00a_b_eoat"
    try:
        _define_stl_mesh(
            stage, _eoat_path, EOAT_STL_PATH, (0.72, 0.70, 0.66),
            translate=EOAT_MESH_CENTER_OFFSET,
        )
    except Exception as _eoat_e:
        print(f"[WARN] EOAT STL 로드 실패 → support/roller proxy 사용: {_eoat_e}")
        _support_path = TCP_PRIM_PATH + "/roller_support"
        _support = UsdGeom.Cylinder.Define(stage, _support_path)
        _support.CreateHeightAttr(ROLLER_FORWARD_REACH)
        _support.CreateRadiusAttr(ROLLER_SUPPORT_RADIUS)
        _support.CreateAxisAttr(_axis_letter)
        _support.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.5, 0.55)])
        UsdGeom.Xformable(_support.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(*_support_mid_v))

        _roller_path = TCP_PRIM_PATH + "/paint_roller"
        _roller = UsdGeom.Cylinder.Define(stage, _roller_path)
        _roller.CreateHeightAttr(ROLLER_LENGTH)
        _roller.CreateRadiusAttr(ROLLER_RADIUS)
        _roller.CreateAxisAttr(_roller_long_axis)
        _roller.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.95, 0.92)])
        UsdGeom.Xformable(_roller.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(*_roller_center_v))
    else:
        _roller_path = _eoat_path

    _d405_path = TCP_PRIM_PATH + "/d405"
    try:
        _define_stl_mesh(
            stage, _d405_path, D405_STL_PATH, (0.94, 0.94, 0.90),
            transform_vertex=_d405_stl_to_tcp_frame,
        )
    except Exception as _d405_e:
        print(f"[WARN] D405 STL 로드 실패 → bbox proxy 사용: {_d405_e}")
        _d405 = UsdGeom.Cube.Define(stage, _d405_path)
        _d405.CreateSizeAttr(1.0)
        _d405.CreateDisplayColorAttr([Gf.Vec3f(0.94, 0.94, 0.90)])
        _d405_xf = UsdGeom.Xformable(_d405.GetPrim())
        _d405_xf.AddTranslateOp().Set(Gf.Vec3d(*D405_COLLISION_CENTER))
        _d405_xf.AddScaleOp().Set(Gf.Vec3f(*D405_SIZE))

    print(f"[OK] AFT200 + RR-00A_B EOAT(no-camera) + D405 부착: {_aft_path}, {_roller_path}, {_d405_path}")
    print(f"     장착 방향 (TCP local): {TOOL_AXIS}, tip offset={EOAT_TIP_OFFSET:.6f}m")
    print(f"     AFT200: STL visual={AFT200_STL_PATH}, collision proxy size={AFT200_SIZE}")
    print(f"     EOAT(no-camera): STL visual={EOAT_STL_PATH}, center offset={EOAT_MESH_CENTER_OFFSET}")
    print(f"     D405: official STL visual={D405_STL_PATH}, link pos={D405_LINK_POSITION}, collision center={D405_COLLISION_CENTER}")

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
# Isaac Sim native ROS2 / sensor 노드 (Phase 5 옵션 C — zed-isaac-sim IPC 우회).
NT_CREATE_RP = _node_type("isaacsim.core.nodes.IsaacCreateRenderProduct",
                            "omni.isaac.core_nodes.IsaacCreateRenderProduct")
NT_CAMERA_HELPER = _node_type("isaacsim.ros2.bridge.ROS2CameraHelper",
                                "omni.isaac.ros2_bridge.ROS2CameraHelper")
NT_CAMERA_INFO_HELPER = _node_type("isaacsim.ros2.bridge.ROS2CameraInfoHelper",
                                     "omni.isaac.ros2_bridge.ROS2CameraInfoHelper")
NT_READ_IMU = _node_type("isaacsim.sensors.physics.IsaacReadIMU",
                           "omni.isaac.isaac_sensor.IsaacReadIMU")
NT_PUB_IMU = _node_type("isaacsim.ros2.bridge.ROS2PublishImu",
                          "omni.isaac.ros2_bridge.ROS2PublishImu")

keys = og.Controller.Keys

# ---- TFGraph (ZED 카메라 TF publish) -----------------------------------------
# 로봇 TF 는 robot_state_publisher 가 /joint_states + 같은 URDF 로 만든다.
# Isaac 은 camera TF 만 publish 하여 RViz/MoveIt 과 TF authority 를 분리한다.
_tf_targets = [
    Sdf.Path(CAMERA_PATH),         # ZED X USD reference root — base_link/ZED_X/Camera* 자동
]

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
# ---- Phase 5 옵션 C: Isaac Sim native ROS2 Camera Helper (ZED extension 우회) ----
# zed-isaac-sim 의 SlCameraStreamer + zed-ros2-wrapper IPC 가 silent fail (RGB/Depth
# Viewer 모두 stream 수신 X). 우회: Isaac Sim native ROS2CameraHelper 로 wrapper 와
# 동일한 topic / frame / intrinsic 발행 → perception 코드 100% 재사용.
#
# 발행 토픽 (실 ZED ROS2 wrapper 와 1:1):
#   /zed/zed_node/rgb/color/rect/image           (sensor_msgs/Image)
#   /zed/zed_node/rgb/color/rect/camera_info     (sensor_msgs/CameraInfo)
#   /zed/zed_node/depth/depth_registered         (sensor_msgs/Image, 32FC1, ground truth)
#   /zed/zed_node/depth/camera_info              (sensor_msgs/CameraInfo)
#   /zed/zed_node/imu/data                       (sensor_msgs/Imu)
#
# 해상도: ZED X HD720 (1280x720) — sim 성능 vs 실 ZED X 출력 절충.
# Frame ID: zed_left_camera_frame_optical / zed_imu_link (실 wrapper 와 동일).
# 발행 빈도: 시뮬 physics 60 Hz, frameSkipCount=1 → 30 Hz (실 ZED X HD720 와 일치).
ZED_RES_W = 1280
ZED_RES_H = 720
ZED_FRAME_SKIP = 1                          # 60 Hz / (1+1) = 30 Hz
ZED_LEFT_FRAME_ID = "zed_left_camera_frame_optical"
ZED_RIGHT_FRAME_ID = "zed_right_camera_frame_optical"
ZED_IMU_FRAME_ID = "zed_imu_link"
ZED_RGB_TOPIC = "/zed/zed_node/rgb/color/rect/image"
ZED_RGB_INFO_TOPIC = "/zed/zed_node/rgb/color/rect/camera_info"
ZED_DEPTH_TOPIC = "/zed/zed_node/depth/depth_registered"
ZED_DEPTH_INFO_TOPIC = "/zed/zed_node/depth/camera_info"
ZED_IMU_TOPIC = "/zed/zed_node/imu/data"

og.Controller.edit(
    {"graph_path": "/World/ZedROS2Graph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnTick", NT_TICK),
            ("LeftRP", NT_CREATE_RP),       # IsaacCreateRenderProduct (좌)
            ("RightRP", NT_CREATE_RP),      # IsaacCreateRenderProduct (우, camera_info 의 stereo intrinsic 용)
            ("RGBHelper", NT_CAMERA_HELPER),
            ("DepthHelper", NT_CAMERA_HELPER),
            ("CamInfoRGB", NT_CAMERA_INFO_HELPER),
            ("CamInfoDepth", NT_CAMERA_INFO_HELPER),
            ("ReadIMU", NT_READ_IMU),
            ("PubIMU", NT_PUB_IMU),
            ("SimTime", NT_SIM_TIME),
        ],
        keys.CONNECT: [
            # Tick → render product 생성 → camera helpers
            ("OnTick.outputs:tick", "LeftRP.inputs:execIn"),
            ("OnTick.outputs:tick", "RightRP.inputs:execIn"),
            ("LeftRP.outputs:execOut", "RGBHelper.inputs:execIn"),
            ("LeftRP.outputs:execOut", "DepthHelper.inputs:execIn"),
            ("LeftRP.outputs:execOut", "CamInfoRGB.inputs:execIn"),
            ("LeftRP.outputs:execOut", "CamInfoDepth.inputs:execIn"),
            ("LeftRP.outputs:renderProductPath", "RGBHelper.inputs:renderProductPath"),
            ("LeftRP.outputs:renderProductPath", "DepthHelper.inputs:renderProductPath"),
            ("LeftRP.outputs:renderProductPath", "CamInfoRGB.inputs:renderProductPath"),
            ("LeftRP.outputs:renderProductPath", "CamInfoDepth.inputs:renderProductPath"),
            ("RightRP.outputs:renderProductPath", "CamInfoRGB.inputs:renderProductPathRight"),
            ("RightRP.outputs:renderProductPath", "CamInfoDepth.inputs:renderProductPathRight"),
            # Tick → IMU read → ROS publish
            ("OnTick.outputs:tick", "ReadIMU.inputs:execIn"),
            ("ReadIMU.outputs:execOut", "PubIMU.inputs:execIn"),
            ("ReadIMU.outputs:angVel", "PubIMU.inputs:angularVelocity"),
            ("ReadIMU.outputs:linAcc", "PubIMU.inputs:linearAcceleration"),
            ("ReadIMU.outputs:orientation", "PubIMU.inputs:orientation"),
            ("SimTime.outputs:simulationTime", "PubIMU.inputs:timeStamp"),
        ],
        keys.SET_VALUES: [
            # Render product 해상도
            ("LeftRP.inputs:width", ZED_RES_W),
            ("LeftRP.inputs:height", ZED_RES_H),
            ("RightRP.inputs:width", ZED_RES_W),
            ("RightRP.inputs:height", ZED_RES_H),
            # RGB helper
            ("RGBHelper.inputs:type", "rgb"),
            ("RGBHelper.inputs:topicName", ZED_RGB_TOPIC),
            ("RGBHelper.inputs:frameId", ZED_LEFT_FRAME_ID),
            ("RGBHelper.inputs:frameSkipCount", ZED_FRAME_SKIP),
            # Depth helper (ground-truth 32FC1 m)
            ("DepthHelper.inputs:type", "depth"),
            ("DepthHelper.inputs:topicName", ZED_DEPTH_TOPIC),
            ("DepthHelper.inputs:frameId", ZED_LEFT_FRAME_ID),
            ("DepthHelper.inputs:frameSkipCount", ZED_FRAME_SKIP),
            # CameraInfo (stereo — left + right intrinsics 같이 발행)
            ("CamInfoRGB.inputs:topicName", ZED_RGB_INFO_TOPIC),
            ("CamInfoRGB.inputs:topicNameRight", ZED_RGB_INFO_TOPIC + "_right"),
            ("CamInfoRGB.inputs:frameId", ZED_LEFT_FRAME_ID),
            ("CamInfoRGB.inputs:frameIdRight", ZED_RIGHT_FRAME_ID),
            ("CamInfoRGB.inputs:frameSkipCount", ZED_FRAME_SKIP),
            ("CamInfoDepth.inputs:topicName", ZED_DEPTH_INFO_TOPIC),
            ("CamInfoDepth.inputs:topicNameRight", ZED_DEPTH_INFO_TOPIC + "_right"),
            ("CamInfoDepth.inputs:frameId", ZED_LEFT_FRAME_ID),
            ("CamInfoDepth.inputs:frameIdRight", ZED_RIGHT_FRAME_ID),
            ("CamInfoDepth.inputs:frameSkipCount", ZED_FRAME_SKIP),
            # IMU
            ("PubIMU.inputs:topicName", ZED_IMU_TOPIC),
            ("PubIMU.inputs:frameId", ZED_IMU_FRAME_ID),
        ],
    },
)

# cameraPrim 은 OGN 의 type="target" relationship — SET_VALUES 로 안 잡힘. 별도 설정.
_LEFT_CAM = CAMERA_PATH + "/base_link/ZED_X/CameraLeft"
_RIGHT_CAM = CAMERA_PATH + "/base_link/ZED_X/CameraRight"
for _node_name, _target in [
    ("/World/ZedROS2Graph/LeftRP",  _LEFT_CAM),
    ("/World/ZedROS2Graph/RightRP", _RIGHT_CAM),
]:
    _np = stage.GetPrimAtPath(_node_name)
    _rel = _np.GetRelationship("inputs:cameraPrim") or \
           _np.CreateRelationship("inputs:cameraPrim", custom=False)
    _rel.SetTargets([Sdf.Path(_target)])

# IMU read 의 imuPrim relationship — kit command 로 재생성된 IMU prim path 사용.
_imu_node_path = "/World/ZedROS2Graph/ReadIMU"
_imu_node_prim = stage.GetPrimAtPath(_imu_node_path)
_imu_rel = _imu_node_prim.GetRelationship("inputs:imuPrim") or \
           _imu_node_prim.CreateRelationship("inputs:imuPrim", custom=False)
_imu_rel.SetTargets([Sdf.Path(IMU_PRIM_PATH)])

print("[OK] ROS2 OmniGraph: TFGraph + JointGraph + ClockGraph + ZedROS2Graph 생성")
print(f"     Cameras   : {_LEFT_CAM}, {_RIGHT_CAM}")
print(f"     IMU       : {IMU_PRIM_PATH}")
print(f"     Resolution: {ZED_RES_W}x{ZED_RES_H} @ {60//(ZED_FRAME_SKIP+1)} Hz")
print(f"     Topics    : {ZED_RGB_TOPIC}")
print(f"                 {ZED_DEPTH_TOPIC}")
print(f"                 {ZED_RGB_INFO_TOPIC}, {ZED_DEPTH_INFO_TOPIC}")
print(f"                 {ZED_IMU_TOPIC}")

# ---- Ground truth dump --------------------------------------------------------
# sim scene/perception 진단용 기준값. camera pose, 벽, 작업영역 corner 를
# 동일 기준(World/real base)으로 읽을 수 있게 저장한다.
import json as _json                                          # noqa: E402

def _world_pose_dict(prim_path):
    """Stage 의 prim world transform 을 (translation, rotation xyzw) dict 로."""
    _p = stage.GetPrimAtPath(prim_path)
    if not _p.IsValid() or not _p.IsA(UsdGeom.Xformable):
        return None
    _xf = UsdGeom.Xformable(_p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    _t = _xf.ExtractTranslation()
    _q = _xf.ExtractRotation().GetQuat()
    _i = _q.GetImaginary()
    return {
        "translation": [float(_t[0]), float(_t[1]), float(_t[2])],
        "rotation_xyzw": [float(_i[0]), float(_i[1]), float(_i[2]), float(_q.GetReal())],
    }

def _quat_mul_xyzw(a, b):
    """xyzw quaternion composition: result = a * b."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ]

def _camera_optical_world_pose_dict(prim_path):
    """USD Camera prim pose 를 ROS optical frame convention 으로 변환.

    Isaac USD camera prim 과 ROS2CameraHelper 의 optical frame 은 local X축 기준
    180도 차이가 난다. calibration GT 는 ROS optical frame 기준이어야 한다.
    """
    pose = _world_pose_dict(prim_path)
    if pose is None:
        return None
    pose["rotation_xyzw"] = _quat_mul_xyzw(
        pose["rotation_xyzw"],
        [1.0, 0.0, 0.0, 0.0],  # local Rx(180deg)
    )
    return pose

_gt = {
    "schema_version": 1,
    "units": "meters / quaternion(xyzw)",
    "physics_dt_hz": 60,
    "wall": {
        "world_pose": _world_pose_dict("/World/wall") or {
            "translation": WALL_CENTER.tolist(),
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "size_xyz": WALL_SIZE.tolist(),
        "front_surface_x": _WALL_FRONT_X,
        "front_surface_normal": [-1.0, 0.0, 0.0],
    },
    "work_area": {
        "size_wh": [WORK_AREA_W, WORK_AREA_H],
        "tape_width": TAPE_W,
        "corners": [
            {"id": _cid, "world": [float(c) for c in _w]}
            for _cid, _w in WORK_AREA_CORNERS
        ],
    },
    "camera_optical_world_pose": _camera_optical_world_pose_dict(_LEFT_CAM),
    "camera_right_optical_world_pose": _camera_optical_world_pose_dict(_RIGHT_CAM),
    # MoveIt 의 robot base 는 URDF link0 frame 이며, launch static TF 에서
    # world(real base) -> link0(+90deg) 관계를 제공한다.
    "robot_base_world_pose": {
        "translation": [0.0, 0.0, 0.0],
        "rotation_xyzw": [0.0, 0.0, 0.7071067811865475, 0.7071067811865476],
    },
    "topics": {
        "rgb": ZED_RGB_TOPIC,
        "rgb_camera_info": ZED_RGB_INFO_TOPIC,
        "depth": ZED_DEPTH_TOPIC,
        "depth_camera_info": ZED_DEPTH_INFO_TOPIC,
        "imu": ZED_IMU_TOPIC,
    },
    "frames": {
        "left_camera_optical": ZED_LEFT_FRAME_ID,
        "right_camera_optical": ZED_RIGHT_FRAME_ID,
        "imu": ZED_IMU_FRAME_ID,
    },
}
_GT_PATH = "/home/minjea/sketch_robot_ws/ground_truth.json"
try:
    with open(_GT_PATH, "w") as _f:
        _json.dump(_gt, _f, indent=2)
    print(f"[OK] ground_truth.json 저장: {_GT_PATH}")
except Exception as _e:
    print(f"[ERROR] ground_truth.json 저장 실패: {_e}")

print("=" * 60)
print("Isaac Sim RB10 씬 준비 완료 (Phase 5 옵션 C — Isaac Sim native ROS2)")
print(f"  RB10:        link0=(0,0,0), articulation={ARTICULATION_PATH}")
print(f"  Table:       center={TABLE_CENTER.tolist()} size={TABLE_SIZE.tolist()}")
print(f"  Steel plate: center={PLATE_CENTER.tolist()} size={PLATE_SIZE.tolist()}")
print(f"  Wall:        center={WALL_CENTER.tolist()} size={WALL_SIZE.tolist()} "
      f"(front x={_WALL_FRONT_X}, normal=-X)")
print(f"  Work area:   {WORK_AREA_W}m × {WORK_AREA_H}m yellow outline at wall center")
print(f"  Camera EYE:  {tuple(CAMERA_EYE)} → target={tuple(CAMERA_TARGET)}")
print(f"  Mount seg1:  center={MOUNT_SEG1_CENTER.tolist()} size={MOUNT_SEG1_SIZE.tolist()}")
print(f"  Mount seg2:  center={MOUNT_SEG2_CENTER.tolist()} size={MOUNT_SEG2_SIZE.tolist()}")
print(f"  Mount ball:  center={MOUNT_BALL_CENTER.tolist()} r={MOUNT_BALL_RADIUS}")
print(f"  Mount bolt:  center={MOUNT_BOLT_CENTER.tolist()} r={MOUNT_BOLT_RADIUS} h={MOUNT_BOLT_HEIGHT}")
print(f"  EOAT:        tcp -> AFT200 -> RR-00A_B(no-camera)+D405, tcp local {TOOL_AXIS}, "
      f"tip offset={EOAT_TIP_OFFSET:.6f}m, mesh={EOAT_STL_PATH}")
print(f"  ROS topics:  /joint_states, /joint_command, /tf, /clock,")
print(f"               /zed/zed_node/rgb/color/rect/image (+camera_info),")
print(f"               /zed/zed_node/depth/depth_registered (+camera_info),")
print(f"               /zed/zed_node/imu/data")
print(f"  Ground truth: {_GT_PATH}")
print(f"  Pose 적용: WORK_POSE (real RB10 joint pose; scene/collision coords are base-frame)")
print("=" * 60)
