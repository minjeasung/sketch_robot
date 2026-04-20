"""
Isaac Sim UR10 벽에 글씨 쓰기
실행: isaacsim --exec ~/sketch_robot_ws/sketch_ur10.py
"""
import json
import os
import numpy as np

import omni
import omni.kit.app

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from pxr import UsdGeom, Gf, Sdf, UsdShade, UsdPhysics

WAYPOINT_FILE = "/tmp/sketch_waypoints.json"
DONE_FILE = "/tmp/sketch_done.flag"

# 레이아웃 설정
ROBOT_POS = (-0.4, 0.0, 0.75)  # 테이블 위, 벽에서 충분히 떨어짐
WALL_X = 0.65                   # 벽 x위치
IK_X = 0.63                     # EE가 벽 표면 바로 앞에 닿도록
EE_FRAME = "ee_link"            # UR10 Lula 표준 엔드이펙터 프레임

# 벽의 실제 z 범위 (center=0.75, scale=1.5 → z=0.0~1.5)
WALL_Z_MIN = 0.0
WALL_Z_MAX = 1.5
# 글씨 쓰기 안전 영역 (벽 내부 마진 확보)
DRAW_Z_MIN = 0.15
DRAW_Z_MAX = 1.35
DRAW_Y_MIN = -0.35
DRAW_Y_MAX = 0.35

# 팔꿈치가 벽 반대쪽(뒤쪽)으로 가는 초기 자세
# shoulder_pan=0(정면), shoulder_lift=-π/4(약간 위), elbow=π/2(팔꿈치 뒤로 접힘)
INITIAL_JOINT_POS = np.array([0.0, -np.pi / 4, np.pi / 2, -np.pi / 4, -np.pi / 2, 0.0])


def get_ee_world_pos(robot):
    """엔드이펙터의 실제 월드 좌표를 가져온다"""
    stage = omni.usd.get_context().get_stage()
    ee_frame = state.get("ee_frame", EE_FRAME)
    # UR10 prim 트리에서 EE 프레임 찾기
    ee_prim = stage.GetPrimAtPath(f"/World/UR10/{ee_frame}")
    if not ee_prim.IsValid():
        # 중첩 경로일 수 있으므로 재귀 탐색
        from pxr import Usd
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/UR10")):
            if prim.GetName() == ee_frame:
                ee_prim = prim
                break
    if ee_prim.IsValid():
        xformable = UsdGeom.Xformable(ee_prim)
        world_tf = xformable.ComputeLocalToWorldTransform(0)
        pos = world_tf.ExtractTranslation()
        return np.array([pos[0], pos[1], pos[2]])
    return None


def create_trail_dot(stage, pos, idx):
    """EE 실제 위치 기반으로 벽 표면에 펜 자국 생성"""
    path = f"/World/Trail/dot_{idx}"
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.GetRadiusAttr().Set(0.008)
    # EE의 y, z를 사용하되 x는 벽 표면에 고정
    wall_surface_x = WALL_X - 0.012
    sphere.AddTranslateOp().Set(Gf.Vec3d(wall_surface_x, float(pos[1]), float(pos[2])))

    mat_path = f"/World/Trail/mat_{idx}"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.1, 0.4))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    sphere.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(mat)


def add_collision(prim):
    """프림에 CollisionAPI 추가"""
    UsdPhysics.CollisionAPI.Apply(prim)


state = {
    "waypoints": [],
    "current_wp_idx": 0,
    "moving": False,
    "phase": "idle",       # "approach" | "draw" | "idle"
    "interp_step": 0,
    "interp_steps": 15,
    "ur10": None,
    "ik_solver": None,
    "initialized": False,
    "trail_idx": 0,
}


def on_update(e):
    on_physics_step(0.016)


def on_physics_step(step_size):
    s = state
    if not s["initialized"]:
        return

    ur10 = s["ur10"]
    ik = s["ik_solver"]
    stage = omni.usd.get_context().get_stage()

    # 새 waypoint 파일 감지
    if not s["moving"] and os.path.exists(WAYPOINT_FILE):
        try:
            with open(WAYPOINT_FILE, 'r') as f:
                data = json.load(f)
            if data.get("execute"):
                s["waypoints"] = data["waypoints"]
                s["current_wp_idx"] = 0
                s["moving"] = True
                s["phase"] = "draw"
                s["interp_step"] = 0
                os.remove(WAYPOINT_FILE)

                # Trail 초기화
                trail_prim = stage.GetPrimAtPath("/World/Trail")
                if trail_prim.IsValid():
                    stage.RemovePrim("/World/Trail")
                UsdGeom.Xform.Define(stage, "/World/Trail")
                s["trail_idx"] = 0

                print(f"글씨 쓰기 시작: {len(s['waypoints'])}개 포인트")
        except Exception as e:
            print(f"파일 읽기 오류: {e}")

    if not s["moving"]:
        return

    # IK로 각 waypoint에 대한 타겟 설정 및 이동
    if s["current_wp_idx"] < len(s["waypoints"]):
        wp = s["waypoints"][s["current_wp_idx"]]
        # 벽 안전 영역으로 클램핑
        clamped_y = np.clip(wp[1], DRAW_Y_MIN, DRAW_Y_MAX)
        clamped_z = np.clip(wp[2], DRAW_Z_MIN, DRAW_Z_MAX)
        target_pos = np.array([IK_X, clamped_y, clamped_z])

        # 벽을 향한 orientation (x축 양의 방향)
        target_orient = euler_angles_to_quat(np.array([0, np.pi / 2, 0]))

        # 현재 joint를 seed로 사용 → 팔꿈치-뒤 자세 유지
        current_joints = ur10.get_joint_positions()
        if current_joints is None:
            current_joints = INITIAL_JOINT_POS.copy()

        # IK solver 호출
        ik_result = ik.compute_inverse_kinematics(
            target_position=target_pos,
            target_orientation=target_orient,
        )

        # 반환값 타입에 따라 joint positions 추출
        target_joints = None
        if isinstance(ik_result, tuple):
            # (ArticulationAction, success_bool) 형태
            action, ik_success = ik_result[0], ik_result[1]
            if ik_success and hasattr(action, 'joint_positions') and action.joint_positions is not None:
                target_joints = np.array(action.joint_positions)
            elif ik_success and isinstance(action, np.ndarray):
                target_joints = action
        elif hasattr(ik_result, 'joint_positions') and ik_result.joint_positions is not None:
            # ArticulationAction 객체
            target_joints = np.array(ik_result.joint_positions)
        elif isinstance(ik_result, np.ndarray):
            target_joints = ik_result

        if target_joints is not None:

            # 보간 이동
            if s["interp_step"] < s["interp_steps"]:
                t = s["interp_step"] / s["interp_steps"]
                t = t * t * (3 - 2 * t)  # smoothstep
                new_joints = current_joints * (1 - t) + target_joints * t
                # IK 결과는 6DOF만 사용
                ur10.set_joint_positions(new_joints[:6])
                s["interp_step"] += 1

                # EE 실제 위치로 trail dot 생성
                if s["interp_step"] % 3 == 0:
                    ee_pos = get_ee_world_pos(ur10)
                    if ee_pos is not None:
                        create_trail_dot(stage, ee_pos, s["trail_idx"])
                        s["trail_idx"] += 1
            else:
                # 현재 waypoint 도착 - EE 실제 위치로 dot
                ur10.set_joint_positions(target_joints[:6])
                ee_pos = get_ee_world_pos(ur10)
                if ee_pos is not None:
                    create_trail_dot(stage, ee_pos, s["trail_idx"])
                    s["trail_idx"] += 1

                # 다음 waypoint로
                s["current_wp_idx"] += 1
                s["interp_step"] = 0

                if s["current_wp_idx"] >= len(s["waypoints"]):
                    s["moving"] = False
                    s["phase"] = "idle"
                    print("글씨 쓰기 완료!")
                    with open(DONE_FILE, 'w') as f:
                        f.write("done")
        else:
            # IK 해가 없으면 스킵
            print(f"IK 실패 (wp {s['current_wp_idx']}): pos={wp}")
            s["current_wp_idx"] += 1
            s["interp_step"] = 0
            if s["current_wp_idx"] >= len(s["waypoints"]):
                s["moving"] = False
                s["phase"] = "idle"
                print("글씨 쓰기 완료! (일부 포인트 스킵됨)")
                with open(DONE_FILE, 'w') as f:
                    f.write("done")


def setup_scene():
    stage = omni.usd.get_context().get_stage()
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # === 테이블 (로봇 받침대) ===
    table = UsdGeom.Cube.Define(stage, "/World/Table")
    table.GetSizeAttr().Set(1.0)
    txf = UsdGeom.Xformable(table.GetPrim())
    txf.AddTranslateOp().Set(Gf.Vec3d(ROBOT_POS[0], 0.0, ROBOT_POS[2] / 2.0))
    txf.AddScaleOp().Set(Gf.Vec3f(0.4, 0.4, ROBOT_POS[2]))

    tmat = UsdShade.Material.Define(stage, "/World/TableMat")
    tshader = UsdShade.Shader.Define(stage, "/World/TableMat/shader")
    tshader.CreateIdAttr("UsdPreviewSurface")
    tshader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.35, 0.25, 0.15))
    tshader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    tmat.CreateSurfaceOutput().ConnectToSource(tshader.ConnectableAPI(), "surface")
    table.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(table.GetPrim()).Bind(tmat)
    add_collision(table.GetPrim())

    # === 벽 ===
    wall = UsdGeom.Cube.Define(stage, "/World/Wall")
    wall.GetSizeAttr().Set(1.0)
    wxf = UsdGeom.Xformable(wall.GetPrim())
    wxf.AddTranslateOp().Set(Gf.Vec3d(WALL_X, 0.0, 0.75))
    wxf.AddScaleOp().Set(Gf.Vec3f(0.015, 1.0, 1.5))

    wmat = UsdShade.Material.Define(stage, "/World/WallMat")
    wshader = UsdShade.Shader.Define(stage, "/World/WallMat/shader")
    wshader.CreateIdAttr("UsdPreviewSurface")
    wshader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.96, 0.94, 0.88))
    wshader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    wmat.CreateSurfaceOutput().ConnectToSource(wshader.ConnectableAPI(), "surface")
    wall.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(wall.GetPrim()).Bind(wmat)
    add_collision(wall.GetPrim())

    # === UR10 (테이블 위) ===
    assets_root = get_assets_root_path()
    if assets_root:
        ur10_usd = assets_root + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
    else:
        ur10_usd = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/UniversalRobots/ur10/ur10.usd"

    add_reference_to_stage(usd_path=ur10_usd, prim_path="/World/UR10")
    ur10_prim = stage.GetPrimAtPath("/World/UR10")
    uxf = UsdGeom.Xformable(ur10_prim)
    uxf.ClearXformOpOrder()
    uxf.AddTranslateOp().Set(Gf.Vec3d(*ROBOT_POS))

    # === 카메라: 정면 약간 좌측에서 바라봄 ===
    camera = UsdGeom.Camera.Define(stage, "/World/FrontCamera")
    cxf = UsdGeom.Xformable(camera.GetPrim())
    cxf.AddTranslateOp().Set(Gf.Vec3d(-0.3, -1.5, 1.0))
    cxf.AddRotateXYZOp().Set(Gf.Vec3f(65.0, 0.0, 10.0))
    camera.GetFocalLengthAttr().Set(20.0)

    # Trail 그룹
    UsdGeom.Xform.Define(stage, "/World/Trail")

    state["world"] = world
    for f in [WAYPOINT_FILE, DONE_FILE]:
        if os.path.exists(f):
            os.remove(f)

    print("=" * 50)
    print("UR10 벽 글씨 쓰기 준비 완료!")
    print("1) FrontCamera 선택 (뷰포트 카메라 아이콘)")
    print("2) Play(▶) 클릭")
    print("3) python3 ~/sketch_robot_ws/sketch_ui_standalone.py")
    print("=" * 50)


def on_play():
    try:
        from omni.isaac.manipulators import SingleManipulator
        from omni.isaac.manipulators.grippers import SurfaceGripper
        from omni.isaac.core.utils.types import ArticulationAction

        # Articulation 초기화
        ur10 = Articulation(prim_path="/World/UR10")
        ur10.initialize()

        # 팔꿈치가 벽 반대쪽으로 가는 초기 자세 설정
        ur10.set_joint_positions(INITIAL_JOINT_POS)
        state["ur10"] = ur10
        print(f"UR10 초기화 완료! joints: {ur10.dof_names}")
        print(f"초기 자세 설정: 팔꿈치 뒤쪽 배치")

        # Lula IK solver 초기화
        from omni.isaac.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver

        # Isaac Sim 내장 UR10 IK 설정 파일 경로
        from omni.isaac.motion_generation import interface_config_loader
        mg_config = interface_config_loader.load_supported_motion_policy_config("UR10", "RMPflow")

        kinematics_solver = LulaKinematicsSolver(
            robot_description_path=mg_config["robot_description_path"],
            urdf_path=mg_config["urdf_path"],
        )

        # 사용 가능한 프레임 이름 출력
        frame_names = kinematics_solver.get_all_frame_names()
        print(f"사용 가능한 프레임: {frame_names}")

        # EE 프레임 확인 및 선택
        ee_frame = EE_FRAME
        if ee_frame not in frame_names:
            # UR10 Lula에서 쓸 수 있는 프레임 후보
            for candidate in ["ee_link", "wrist_3_link", "flange", "tool0"]:
                if candidate in frame_names:
                    ee_frame = candidate
                    break
            else:
                ee_frame = frame_names[-1]
            print(f"경고: '{EE_FRAME}' 없음 → '{ee_frame}' 사용")
        print(f"EE 프레임: {ee_frame}")

        ik_solver = ArticulationKinematicsSolver(
            robot_articulation=ur10,
            kinematics_solver=kinematics_solver,
            end_effector_frame_name=ee_frame,
        )
        state["ee_frame"] = ee_frame

        state["ik_solver"] = ik_solver
        state["initialized"] = True
        print("IK Solver 초기화 완료!")

    except Exception as e:
        print(f"초기화 오류: {e}")
        import traceback
        traceback.print_exc()


setup_scene()

import omni.timeline
timeline = omni.timeline.get_timeline_interface()

def on_timeline_event(e):
    if e.type == int(omni.timeline.TimelineEventType.PLAY):
        on_play()

timeline_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(on_timeline_event)
app = omni.kit.app.get_app()
update_sub = app.get_update_event_stream().create_subscription_to_pop(on_update)
