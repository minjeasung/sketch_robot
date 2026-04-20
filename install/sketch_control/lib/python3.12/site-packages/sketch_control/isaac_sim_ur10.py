"""
Isaac Sim UR10 벽에 쓰기 씬 + ROS2 브릿지 (OmniGraph 전용, rclpy 미사용)

실행: ~/sketch_robot_ws/run_isaac_sim.sh

씬 구성:
  - UR10: 베이스 기둥(0.8m) 위에 설치, 벽을 향해 작업
  - Wall: 로봇 전방 벽면 (1.5m x 1.2m)
  - Camera: 로봇 뒤-옆-위에서 촬영

퍼블리시 (OmniGraph):
  /camera/image_raw, /camera/camera_info, /tf, /joint_states
구독 (OmniGraph):
  /joint_command
"""
import numpy as np
import time as _time
import yaml
import carb
import omni
import omni.kit.app
import omni.graph.core as og
from pxr import UsdGeom, UsdPhysics, Gf, Sdf, PhysxSchema
import usdrt.Sdf

# ---- objects.yaml 로드 (Isaac Sim 은 ament_index_python 없을 수 있어 절대경로) ----
OBJECTS_CFG_PATH = "/home/minjea/sketch_robot_ws/src/sketch_control/config/objects.yaml"
with open(OBJECTS_CFG_PATH, "r") as _f:
    OBJECTS_CFG = yaml.safe_load(_f)

_FACE_MAP = {
    "+x": (np.array([1.0, 0.0, 0.0]), 0),
    "-x": (np.array([-1.0, 0.0, 0.0]), 0),
    "+y": (np.array([0.0, 1.0, 0.0]), 1),
    "-y": (np.array([0.0, -1.0, 0.0]), 1),
    "+z": (np.array([0.0, 0.0, 1.0]), 2),
    "-z": (np.array([0.0, 0.0, -1.0]), 2),
}


def _surface_plane(obj):
    normal, axis = _FACE_MAP[obj["sketch_face"]]
    pos = np.array(obj["position"], dtype=float)
    half = np.array(obj["size"], dtype=float) / 2.0
    return pos + normal * half[axis], normal

# ---- ROS2 bridge 활성화 + 확인 ------------------------------------------------
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

# ---- Isaac Sim core import (5.0/4.5 호환) ------------------------------------
try:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation as Robot
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.nucleus import get_assets_root_path
except ImportError:
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.objects import FixedCuboid
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path

# ---- 월드 설정 ----------------------------------------------------------------
stage = omni.usd.get_context().get_stage()
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# ---- 베이스 기둥 (UR10 아래) ---------------------------------------------------
robot_base = world.scene.add(
    FixedCuboid(
        prim_path="/World/RobotBase", name="robot_base",
        position=np.array([0.0, 0.0, -0.4]),
        scale=np.array([0.3, 0.3, 0.8]),
        color=np.array([0.5, 0.5, 0.5]),
    )
)

# ---- UR10 로딩 ----------------------------------------------------------------
assets_root = get_assets_root_path()
ur10_usd = assets_root + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
add_reference_to_stage(usd_path=ur10_usd, prim_path="/World/UR10")

# UR10 은 USD 레퍼런스 내부 어딘가에 ArticulationRoot 가 있음 — 실제 경로 탐색
ur10_prim = stage.GetPrimAtPath("/World/UR10")

ARTICULATION_PATH = None
# UR10 서브트리 전체에서 ArticulationRootAPI 가진 prim 탐색
for _p in stage.Traverse():
    _path_str = _p.GetPath().pathString
    if _path_str.startswith("/World/UR10") and _p.HasAPI(UsdPhysics.ArticulationRootAPI):
        ARTICULATION_PATH = _path_str
        print(f"[OK] ArticulationRoot 발견: {ARTICULATION_PATH}")
        break

if ARTICULATION_PATH is None:
    # USD 원본에 없으면 /World/UR10 에 직접 적용 (single root 보장)
    UsdPhysics.ArticulationRootAPI.Apply(ur10_prim)
    ARTICULATION_PATH = "/World/UR10"
    print(f"[FIX] ArticulationRoot 없음 → {ARTICULATION_PATH} 에 Apply")

robot = world.scene.add(
    Robot(prim_path=ARTICULATION_PATH, name="ur10")
)

# ---- 작업 대상 물체들 (YAML 기반) ---------------------------------------------
for _obj in OBJECTS_CFG["objects"]:
    if not _obj.get("enabled", True):
        continue
    if _obj["shape"] != "box":
        print(f"[WARN] shape={_obj['shape']} 미지원, {_obj['name']} 스킵")
        continue
    _prim_path = f"/World/{_obj['name']}"
    world.scene.add(FixedCuboid(
        prim_path=_prim_path,
        name=_obj["name"],
        position=np.array(_obj["position"]),
        scale=np.array(_obj["size"]),
        color=np.array(_obj["color"]),
    ))
    _obj_prim = stage.GetPrimAtPath(_prim_path)
    if not _obj_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(_obj_prim)
    print(f"[OK] 물체 생성: {_obj['name']} pos={_obj['position']} face={_obj['sketch_face']}")

# Active target 의 표면 평면 (paint 시스템용)
_active_name = OBJECTS_CFG.get("active_target", "wall")
_active_obj = next(o for o in OBJECTS_CFG["objects"]
                   if o["name"] == _active_name and o.get("enabled", True))
ACTIVE_PLANE_POINT, ACTIVE_PLANE_NORMAL = _surface_plane(_active_obj)
print(f"[INFO] Active target: {_active_name} | plane_pt={ACTIVE_PLANE_POINT} normal={ACTIVE_PLANE_NORMAL}")

# ---- 카메라 -------------------------------------------------------------------
CAMERA_PATH = "/World/SketchCamera"
CAMERA_EYE = Gf.Vec3d(-1.1, -1.5, 0.8)
CAMERA_TARGET = Gf.Vec3d(float(ACTIVE_PLANE_POINT[0]),
                          float(ACTIVE_PLANE_POINT[1]),
                          float(ACTIVE_PLANE_POINT[2]))

cam_prim = UsdGeom.Camera.Define(stage, CAMERA_PATH)
cam_prim.GetFocalLengthAttr().Set(16.0)
cam_prim.GetHorizontalApertureAttr().Set(24.0)
cam_xform = UsdGeom.Xformable(cam_prim.GetPrim())


def _look_at_matrix(eye, target, up=Gf.Vec3d(0, 0, 1)):
    fwd = (target - eye); fwd.Normalize()
    right = Gf.Cross(fwd, up); right.Normalize()
    new_up = Gf.Cross(right, fwd)
    return Gf.Matrix4d(
        right[0], right[1], right[2], 0,
        new_up[0], new_up[1], new_up[2], 0,
        -fwd[0], -fwd[1], -fwd[2], 0,
        eye[0], eye[1], eye[2], 1,
    )


for _op in cam_xform.GetOrderedXformOps():
    cam_xform.GetPrim().RemoveProperty(_op.GetOpName())
cam_xform.AddTransformOp().Set(_look_at_matrix(CAMERA_EYE, CAMERA_TARGET))

# ---- 붓 EoAT (UR10 마지막 링크에 부착) -----------------------------------------
BRUSH_LENGTH = 0.15  # 15cm

# UR10 USD 내부에서 tool0 또는 ee_link prim 경로 탐색
_tool0_path = None
for _p in stage.Traverse():
    _name = _p.GetName().lower()
    if _name in ("tool0", "ee_link", "flange") and _p.GetPath().pathString.startswith("/World/UR10"):
        _tool0_path = _p.GetPath().pathString
        break
if _tool0_path is None:
    _tool0_path = "/World/UR10"
    for _p in stage.Traverse():
        _path_str = _p.GetPath().pathString
        if _path_str.startswith("/World/UR10") and "wrist_3_link" in _path_str.lower():
            _tool0_path = _path_str
            break
print(f"[INFO] 붓 부착 대상 prim: {_tool0_path}")

# 기존 Brush/brush_tip 제거 후 재생성
_brush_path = _tool0_path + "/Brush"
_tip_path = _tool0_path + "/brush_tip"
if stage.GetPrimAtPath(_brush_path).IsValid():
    stage.RemovePrim(_brush_path)
if stage.GetPrimAtPath(_tip_path).IsValid():
    stage.RemovePrim(_tip_path)

# 붓: 로컬 +Z 방향으로 뻗음 (이 USD flange 의 +Z 가 시각적으로 벽 방향)
brush = UsdGeom.Cylinder.Define(stage, _brush_path)
brush.CreateHeightAttr(BRUSH_LENGTH)
brush.CreateRadiusAttr(0.01)
brush.CreateAxisAttr("Z")
brush.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.0, 0.0)])
_brush_xf = UsdGeom.Xformable(brush.GetPrim())
_brush_xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, BRUSH_LENGTH / 2.0))

# brush_tip: +Z 방향 15cm
_tip = UsdGeom.Xform.Define(stage, _tip_path)
UsdGeom.Xformable(_tip.GetPrim()).AddTranslateOp().Set(
    Gf.Vec3d(0, 0, BRUSH_LENGTH))
BRUSH_TIP_PATH = _tip_path
print(f"[OK] 붓 부착: axis=Z, 길이={BRUSH_LENGTH*100:.0f}cm, tip={BRUSH_TIP_PATH}")

# ---- 물리 충돌 강제 적용 (world.reset() 전) ------------------------------------
# 각 활성 물체에 Collision 재확인
for _obj in OBJECTS_CFG["objects"]:
    if not _obj.get("enabled", True):
        continue
    _p = stage.GetPrimAtPath(f"/World/{_obj['name']}")
    if _p.IsValid():
        UsdPhysics.CollisionAPI(_p).CreateCollisionEnabledAttr(True)

# RobotBase collider
base_prim = stage.GetPrimAtPath("/World/RobotBase")
if base_prim.IsValid() and not base_prim.HasAPI(UsdPhysics.CollisionAPI):
    UsdPhysics.CollisionAPI.Apply(base_prim)
    print("[FIX] RobotBase CollisionAPI 적용")

# UR10 의 PhysxArticulation 설정도 USD 원본에 맡김 (Apply 하지 않음)

# ==== OmniGraph 생성 ============================================================

def _node_type(name_50, name_45):
    try:
        if og.get_node_type(name_50) is not None:
            return name_50
    except Exception:
        pass
    return name_45

NT_TICK = "omni.graph.action.OnPlaybackTick"
NT_CREATE_RP = _node_type("isaacsim.core.nodes.IsaacCreateRenderProduct",
                          "omni.isaac.core_nodes.IsaacCreateRenderProduct")
NT_CAM = _node_type("isaacsim.ros2.bridge.ROS2CameraHelper",
                     "omni.isaac.ros2_bridge.ROS2CameraHelper")
NT_CAMINFO = _node_type("isaacsim.ros2.bridge.ROS2CameraInfoHelper",
                         "omni.isaac.ros2_bridge.ROS2CameraInfoHelper")
NT_SIM_TIME = _node_type("isaacsim.core.nodes.IsaacReadSimulationTime",
                          "omni.isaac.core_nodes.IsaacReadSimulationTime")
NT_PUB_TF = _node_type("isaacsim.ros2.bridge.ROS2PublishTransformTree",
                         "omni.isaac.ros2_bridge.ROS2PublishTransformTree")
NT_ARTIC_STATE = _node_type("isaacsim.core.nodes.IsaacArticulationState",
                             "omni.isaac.core_nodes.IsaacArticulationState")
NT_PUB_JS = _node_type("isaacsim.ros2.bridge.ROS2PublishJointState",
                         "omni.isaac.ros2_bridge.ROS2PublishJointState")
NT_SUB_JS = _node_type("isaacsim.ros2.bridge.ROS2SubscribeJointState",
                         "omni.isaac.ros2_bridge.ROS2SubscribeJointState")
NT_ARTIC_CTRL = _node_type("isaacsim.core.nodes.IsaacArticulationController",
                            "omni.isaac.core_nodes.IsaacArticulationController")

keys = og.Controller.Keys
CAM_W, CAM_H = 640, 480

# ---- CameraGraph --------------------------------------------------------------
og.Controller.edit(
    {"graph_path": "/World/CameraGraph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("Tick", NT_TICK), ("CreateRP", NT_CREATE_RP),
            ("CamRgb", NT_CAM), ("CamInfo", NT_CAMINFO),
        ],
        keys.CONNECT: [
            ("Tick.outputs:tick", "CreateRP.inputs:execIn"),
            ("CreateRP.outputs:execOut", "CamRgb.inputs:execIn"),
            ("CreateRP.outputs:execOut", "CamInfo.inputs:execIn"),
            ("CreateRP.outputs:renderProductPath", "CamRgb.inputs:renderProductPath"),
            ("CreateRP.outputs:renderProductPath", "CamInfo.inputs:renderProductPath"),
        ],
        keys.SET_VALUES: [
            ("CreateRP.inputs:cameraPrim", [Sdf.Path(CAMERA_PATH)]),
            ("CreateRP.inputs:width", CAM_W), ("CreateRP.inputs:height", CAM_H),
            ("CamRgb.inputs:topicName", "/camera/image_raw"),
            ("CamRgb.inputs:type", "rgb"), ("CamRgb.inputs:frameId", "sketch_camera"),
            ("CamInfo.inputs:topicName", "/camera/camera_info"),
            ("CamInfo.inputs:frameId", "sketch_camera"),
        ],
    },
)

# ---- TFGraph ------------------------------------------------------------------
og.Controller.edit(
    {"graph_path": "/World/TFGraph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("Tick", NT_TICK), ("SimTime", NT_SIM_TIME), ("PubTF", NT_PUB_TF),
        ],
        keys.CONNECT: [
            ("Tick.outputs:tick", "PubTF.inputs:execIn"),
            ("SimTime.outputs:simulationTime", "PubTF.inputs:timeStamp"),
        ],
        keys.SET_VALUES: [
            ("PubTF.inputs:parentPrim", [Sdf.Path("/World")]),
            ("PubTF.inputs:targetPrims", [Sdf.Path("/World/SketchCamera")]),
            ("PubTF.inputs:topicName", "/tf"),
        ],
    },
)

# ---- JointGraph (NVIDIA 공식 예제 패턴) ----------------------------------------
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

# ==== 초기화 완료 ===============================================================
world.reset()

# ---- Articulation 검증 -------------------------------------------------------
try:
    _dofs = robot.num_dof
    _joint_names = robot.dof_names
    print(f"[OK] UR10 articulation 유효: {_dofs} DOF, joints={_joint_names}")
except Exception as _e:
    print(f"[ERROR] UR10 articulation 실패: {_e}")
    print("      → UR10 USD 자체 문제 또는 ArticulationRoot 깨짐")

# 초기 관절 자세: 캘리브레이션된 "벽 찌르기" 자세
# 붓이 벽을 향하도록 수동 튜닝된 값
READY_POSE = np.array([-0.08, -1.6, 1.76, -1.76, -1.9, 3.14])
try:
    robot.set_joint_positions(READY_POSE)
    print(f"[INIT] UR10 READY_POSE 설정: {READY_POSE.tolist()}")
except Exception as _e:
    print(f"[ERROR] READY_POSE 설정 실패: {_e}")

# tool0 prim 참조 (paint callback 에서 사용)
_tool0_prim = stage.GetPrimAtPath(_tool0_path)

# ---- Paint 상수 (초기 검증 로그에서도 사용) -----------------------------------
PAINT_THRESHOLD = 0.015
PAINT_DOT_RADIUS = 0.003

# ---- brush_tip prim 초기 검증 -------------------------------------------------
_tip_prim_check = stage.GetPrimAtPath(BRUSH_TIP_PATH)
if _tip_prim_check.IsValid():
    _xfc_init = UsdGeom.XformCache()
    _wm_init = _xfc_init.GetLocalToWorldTransform(_tip_prim_check)
    _tip_init_pos = _wm_init.ExtractTranslation()
    print(f"[INIT] brush_tip prim 경로: {BRUSH_TIP_PATH}")
    print(f"[INIT] brush_tip world 위치: "
          f"({_tip_init_pos[0]:.3f}, {_tip_init_pos[1]:.3f}, {_tip_init_pos[2]:.3f})")
    print(f"[INIT] active plane point: "
          f"({ACTIVE_PLANE_POINT[0]:.3f}, {ACTIVE_PLANE_POINT[1]:.3f}, "
          f"{ACTIVE_PLANE_POINT[2]:.3f})  normal={ACTIVE_PLANE_NORMAL}")
    _init_delta = np.array([
        _tip_init_pos[0] - ACTIVE_PLANE_POINT[0],
        _tip_init_pos[1] - ACTIVE_PLANE_POINT[1],
        _tip_init_pos[2] - ACTIVE_PLANE_POINT[2],
    ])
    _init_signed = float(np.dot(_init_delta, ACTIVE_PLANE_NORMAL))
    print(f"[INIT] 평면까지 signed_dist = {_init_signed*1000:+.1f}mm "
          f"(접촉범위 -5mm ~ +{PAINT_THRESHOLD*1000:.0f}mm)")
else:
    print(f"[ERROR] brush_tip prim 없음! 페인트 callback 작동 안 함")
    print(f"        기대 경로: {BRUSH_TIP_PATH}")
    for _alt in ["/World/UR10/tool0/brush_tip",
                 "/World/UR10/ee_link/brush_tip",
                 "/World/UR10/wrist_3_link/brush_tip"]:
        if stage.GetPrimAtPath(_alt).IsValid():
            print(f"        대체 존재: {_alt}")

# ======================== 붓 페인팅 (연속 선 + 진단 로그) ========================
import time as _pytime

PAINT_LINE_RADIUS = 0.004
PAINT_CONTACT_THRESHOLD = 0.020      # 평면 바깥 20mm 이내면 접촉
PAINT_BEHIND_THRESHOLD = 0.010       # 평면 안쪽 10mm 까지 허용
PAINT_MIN_STEP = 0.002
PAINT_MAX_STEP = 0.050
PAINT_SNAP_OFFSET = 0.002
PAINT_COLOR = Gf.Vec3f(1.0, 0.0, 0.0)
PAINT_MAX_SEGMENTS = 3000

# 벽 평면 (active target 기준)
WALL_PLANE_POINT = np.array(ACTIVE_PLANE_POINT, dtype=float)
WALL_NORMAL = np.array(ACTIVE_PLANE_NORMAL, dtype=float)

paint_state = {
    "segment_count": 0,
    "last_contact_pos": None,
    "container_created": False,
    "last_log_time": 0.0,
}


def _paint_dbg(msg):
    now = _pytime.time()
    if now - paint_state["last_log_time"] > 0.5:
        print(f"[PAINT_DBG] {msg}")
        paint_state["last_log_time"] = now


def _ensure_paint_container():
    if not paint_state["container_created"]:
        UsdGeom.Xform.Define(stage, "/World/PaintStrokes")
        paint_state["container_created"] = True


def _project_to_wall(world_pos):
    to_point = world_pos - WALL_PLANE_POINT
    dist_along_normal = float(np.dot(to_point, WALL_NORMAL))
    projected = world_pos - WALL_NORMAL * dist_along_normal
    projected = projected + WALL_NORMAL * PAINT_SNAP_OFFSET
    return projected


def _add_segment(p1, p2):
    if paint_state["segment_count"] >= PAINT_MAX_SEGMENTS:
        return
    mid = (p1 + p2) * 0.5
    diff = p2 - p1
    length = float(np.linalg.norm(diff))
    if length < 1e-6:
        return
    direction = diff / length

    paint_state["segment_count"] += 1
    idx = paint_state["segment_count"]
    seg_path = f"/World/PaintStrokes/seg_{idx:05d}"
    cyl = UsdGeom.Cylinder.Define(stage, seg_path)
    cyl.CreateHeightAttr(length)
    cyl.CreateRadiusAttr(PAINT_LINE_RADIUS)
    cyl.CreateAxisAttr("Z")
    cyl.CreateDisplayColorAttr([PAINT_COLOR])

    z_axis = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if dot > 0.9999:
        rot_quat = Gf.Quatd(1.0, 0.0, 0.0, 0.0)
    elif dot < -0.9999:
        rot_quat = Gf.Quatd(0.0, 1.0, 0.0, 0.0)
    else:
        axis = np.cross(z_axis, direction)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(dot)
        s = np.sin(angle / 2.0)
        rot_quat = Gf.Quatd(
            float(np.cos(angle / 2.0)),
            float(axis[0] * s), float(axis[1] * s), float(axis[2] * s),
        )

    xform = UsdGeom.Xformable(cyl.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2])))
    xform.AddOrientOp().Set(rot_quat)


def on_paint_step(step):
    tip_prim = stage.GetPrimAtPath(BRUSH_TIP_PATH)
    if not tip_prim.IsValid():
        _paint_dbg(f"brush_tip prim 없음: {BRUSH_TIP_PATH}")
        return

    xfc = UsdGeom.XformCache()
    world_m = xfc.GetLocalToWorldTransform(tip_prim)
    tip_pos_gf = world_m.ExtractTranslation()
    tip_pos = np.array([float(tip_pos_gf[0]), float(tip_pos_gf[1]), float(tip_pos_gf[2])])

    to_point = tip_pos - WALL_PLANE_POINT
    signed_dist = float(np.dot(to_point, WALL_NORMAL))
    # signed_dist > 0: 평면 바깥(법선 방향)
    # signed_dist < 0: 평면 안쪽(침투)

    _paint_dbg(
        f"tip=({tip_pos[0]:+.3f},{tip_pos[1]:+.3f},{tip_pos[2]:+.3f}) "
        f"signed_dist={signed_dist*1000:+.1f}mm "
        f"범위=[-{PAINT_BEHIND_THRESHOLD*1000:.0f}mm, +{PAINT_CONTACT_THRESHOLD*1000:.0f}mm] "
        f"segments={paint_state['segment_count']}"
    )

    if not (-PAINT_BEHIND_THRESHOLD <= signed_dist <= PAINT_CONTACT_THRESHOLD):
        paint_state["last_contact_pos"] = None
        return

    current = _project_to_wall(tip_pos)
    last = paint_state["last_contact_pos"]

    if last is None:
        _ensure_paint_container()
        paint_state["last_contact_pos"] = current
        _paint_dbg(f"첫 접촉 at ({current[0]:.3f},{current[1]:.3f},{current[2]:.3f})")
        return

    gap = float(np.linalg.norm(current - last))
    if gap < PAINT_MIN_STEP:
        return
    if gap > PAINT_MAX_STEP:
        paint_state["last_contact_pos"] = current
        return

    _ensure_paint_container()
    _add_segment(last, current)
    paint_state["last_contact_pos"] = current


# 기존 callback 제거 후 재등록
try:
    world.remove_physics_callback("paint_brush")
except Exception:
    pass
world.add_physics_callback("paint_brush", on_paint_step)
print("[OK] 붓 페인팅 callback 등록 (연속 선, 진단 로그 포함)")

print("=" * 60)
print("Isaac Sim UR10 씬 준비 완료 (objects.yaml 기반)")
print(f"  UR10: (0, 0, 0) 원점")
print(f"  Active target: {_active_name} (face={_active_obj['sketch_face']})")
print(f"  카메라: {CAMERA_EYE} -> {CAMERA_TARGET}")
print("  토픽: /camera/image_raw, /camera/camera_info, /tf, /joint_states")
print("=" * 60)
