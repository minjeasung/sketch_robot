"""
MoveIt Executor - 4-stage 용접 모션 (Plan + FollowJointTrajectory 실행)

대상 로봇: Rainbow Robotics RB10-1300 (rb10_1300e_u, rbpodo_ros2 driver)
EE link: tcp / Base frame: link0 / Planning group: mainpulation (SRDF 오타)

4-stage 파이프라인:
  Stage 1: 자유 plan (현재 자세 → safety pose)  [OMPL via /move_action]
  Stage 2: cartesian (safety → 표면 첫 점)        [/compute_cartesian_path]
  Stage 3: cartesian path (표면 위 스케치 추종)
  Stage 4: cartesian (표면 끝 → retreat pose)
  Stage 5: 자유 plan (retreat → READY_POSE)

EoAT 체인:
  tcp -> AFT200 force/torque sensor -> RR-00A_B EOAT(no-camera) -> D405

제공된 AFT200 URDF/roller STEP 은 CAD local +Z 방향으로 뻗지만, 실제 장착은
TCP local -Y 방향이다. 따라서 planning 에서는 CAD +Z 를 TCP -Y 로 해석한다.
"""
import copy
import json
import math
import struct
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from tf2_ros import Buffer, TransformListener

from geometry_msgs.msg import Point, PoseArray, Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray

from moveit_msgs.srv import GetCartesianPath, GetPositionIK, ApplyPlanningScene
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    PositionIKRequest, RobotState,
    CollisionObject, AttachedCollisionObject,
    PlanningScene, PlanningSceneWorld,
    Constraints, OrientationConstraint, PositionConstraint, JointConstraint,
)
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory

from sketch_control.targets import (
    load_objects_config, get_surface_plane, get_target, ee_quat_for_target,
)
from sketch_control.rotation_utils import (
    quat_apply, quat_from_matrix, quat_from_two_vectors, quat_multiply,
)


# SRDF 의 그룹명이 "mainpulation" 으로 오타. 그대로 유지 (RB10 공식 SRDF 기준).
PLANNING_GROUP = "mainpulation"
EE_LINK = "tcp"
BASE_FRAME = "link0"

CONTACT_CLEARANCE = 0.002  # MoveIt 상에서는 벽과 2mm clearance 유지
CONTACT_PLANE_TOL = 0.015  # perception/TF noise 허용 범위
WORK_AREA_W = 0.50
WORK_AREA_H = 0.40
WORK_AREA_MARGIN = 0.02

SAFETY_OFFSET = 0.08   # 접촉 전 normal 방향 안전거리
RETREAT_OFFSET = 0.08  # Stage 4 후퇴점: 표면 normal 방향 8cm
MIN_APPROACH_NORMAL_ALIGN = 0.97
MIN_CARTESIAN_FRACTION = 0.99
MIN_RETREAT_CARTESIAN_FRACTION = 0.80
WORLD_COLLISION_PADDING = 0.02
TARGET_COLLISION_MARGIN = 0.20
TARGET_COLLISION_MIN_THICKNESS = 0.02
TARGET_COLLISION_MAX_THICKNESS = 0.08
OBSTACLES_TOPIC = "/perception/obstacles"
PLANES_TOPIC = "/perception/planes"
PLANE_LABELS_TOPIC = "/perception/plane_labels"
WORK_AREA_PLANE_TOPIC = "/perception/work_area_plane"
WORK_AREA_REFINED_PLANE_TOPIC = "/perception/work_area_plane_refined"
WORK_AREA_CORNERS_TOPIC = "/perception/work_area_corners"
FT_STATUS_TOPIC = "/ft/status"
FT_ZERO_TOPIC = "/ft/zero"
JOINT_COMMAND_TOPIC = "/joint_command"
MAX_DYNAMIC_OBSTACLES = 80
DYNAMIC_OBSTACLE_PREFIX = "zed_obstacle_"
ROBOT_SELF_FILTER_PADDING = 0.10
ROBOT_LINK_FRAMES = ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "tcp"]
ROBOT_LINK_CAPSULE_RADIUS = {
    ("link0", "link1"): 0.20,
    ("link1", "link2"): 0.20,
    ("link2", "link3"): 0.18,
    ("link3", "link4"): 0.16,
    ("link4", "link5"): 0.15,
    ("link5", "link6"): 0.15,
    ("link6", "tcp"): 0.20,
}
PLANNER_ID = "RRTConnect"
ALLOWED_PLANNING_TIME = 5.0
PLANNING_ATTEMPTS = 5

LATCHED_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# Runtime speed policy.
# Stage 1 is intentionally slow: it is the longest free-space move near humans
# and can otherwise look abrupt on the real RB10.
STAGE1_SPEED_SCALE = 0.10
STAGE2_SPEED_SCALE = 0.08
STAGE3_SPEED_SCALE = 0.25
STAGE4_SPEED_SCALE = 0.10
STAGE5_SPEED_SCALE = 0.20
STAGE1_IK_TIMEOUT_S = 1.0
STAGE1_JOINT_GOAL_TOL = 0.02
STAGE1_LARGE_JOINT_DELTA_WARN_RAD = 1.2
FT_FORCE_STALE_SEC = 0.75
FT_ABORT_FORCE_N = 30.0
FT_AUTO_ZERO_BEFORE_SKETCH = True
FT_AUTO_ZERO_TIMEOUT_SEC = 6.0
FT_AUTO_ZERO_STALE_SEC = 1.0
FT_AUTO_ZERO_CHECK_PERIOD = 0.05

# 스케치 시작 전에는 시야/해 안정화를 위해 READY_POSE 에서 시작한다.
START_FROM_READY_BEFORE_SKETCH = True

# 작업 종료 후에는 READY_POSE 로 크게 복귀하지 않는다.
# Stage 4 에서 작업 위치 근처에서 벽 normal 방향으로만 짧게 빠진다.
RETURN_TO_READY_AFTER_SKETCH = False

# Stage 1 단독 디버그용. production 의 SAFETY_OFFSET 과 분리.
# 토치 미장착 + 첫 실로봇 검증이라 일반보다 보수적으로 잡음.
DEBUG_STAGE1_OFFSET = 0.15  # meters, surface normal 방향 후퇴 거리
# 실제 RB10 pendant/base 와 rbpodo URDF link0 는 Z축 기준 90도 차이가 있다.
# launch 의 world->link0 static TF(+90deg)가 실제 base(world) 좌표를 URDF link0
# 좌표로 변환한다. 디버그 평면은 실제 base/world 기준 x=+0.80 이다.
DEBUG_STAGE1_SURFACE_NORMAL = (-1.0, 0.0, 0.0)

# Joint-space jog 디버그용. motion pipeline 단독 검증.
# 한 joint 만 작게 움직여 OMPL / IK / Cartesian goal 의존성 모두 우회.
JOG_JOINT_INDEX = 5      # wrist3 (가장 국소적, 충돌 위험 최소)
JOG_DELTA_RAD = 0.05     # ≈ 2.9°. 시각적으로 보이지만 무시할 수준
JOG_DURATION_SEC = 10.0  # 10초에 걸쳐 움직임 → 인간 반응 충분
JOG_NUM_POINTS = 50      # 0.2초 간격 보간

# Isaac Sim 의 RB10 OmniGraph 는 /joint_command(sensor_msgs/JointState)를
# 직접 구독한다. FollowJointTrajectory action 은 실로봇/ros2_control 용으로
# 남겨두되, 시뮬레이션 기본 실행 backend 는 joint_command 로 둔다.
EXECUTION_BACKEND = "joint_command"  # "joint_command" or "follow_joint_trajectory"
# Isaac Sim 에는 FollowJointTrajectory 대신 /joint_command 를 재생한다.
# 50Hz + trajectory time interpolation 으로 계단형 target jump 를 줄인다.
JOINT_COMMAND_TIMER_PERIOD = 0.02

# RB10 joint 운동학 순서 (URDF 기준).
# 주의: /joint_states 토픽은 알파벳 순으로 발행됨 (base, elbow, shoulder, wrist1, wrist2, wrist3) —
# 이 dict 는 이름 매핑이라 순서 무관, 안전.
# 실제 RB10 pendant/driver 에서 읽은 작업 시작 자세.
READY_POSE_JOINTS = {
    "base":     0.0005,   # pendant: +0.03 deg
    "shoulder": -0.9343,  # pendant: -53.53 deg
    "elbow":    2.4246,   # pendant: +138.92 deg
    "wrist1":  -1.6293,   # pendant: -93.35 deg
    "wrist2":   1.5675,   # pendant: +89.81 deg
    "wrist3":   0.0000,
}

# 실제 RB10 joint.yaml / MoveIt URDF 기준. Isaac 에도 같은 범위 안의 명령만 보낸다.
JOINT_LIMITS = {
    "base": (-6.28, 6.28),
    "shoulder": (-3.14, 3.14),
    "elbow": (-3.14, 3.14),
    "wrist1": (-3.14, 3.14),
    "wrist2": (-3.14, 3.14),
    "wrist3": (-3.14, 3.14),
}
JOINT_LIMIT_MARGIN = 0.01

CALIBRATION_POSE_JOINTS = dict(READY_POSE_JOINTS)
# MoveIt RB10 URDF currently limits wrist1 to about -pi. The original Isaac
# calibration candidate is slightly below that, so keep the operator preset
# inside MoveIt bounds.
CALIBRATION_POSE_JOINTS["wrist1"] = -3.13

PRESET_POSES = {
    "ready": ("READY_POSE", READY_POSE_JOINTS, STAGE5_SPEED_SCALE),
    "work": ("READY_POSE", READY_POSE_JOINTS, STAGE5_SPEED_SCALE),
    "view": ("READY_POSE", READY_POSE_JOINTS, STAGE5_SPEED_SCALE),
    "calib": ("CALIB_POSE", CALIBRATION_POSE_JOINTS, STAGE5_SPEED_SCALE),
    "calibration": ("CALIB_POSE", CALIBRATION_POSE_JOINTS, STAGE5_SPEED_SCALE),
}

# MoveIt 내부 planning frame 은 URDF link0 이지만, world/World 는 실제 RB10 base.
# launch static TF(world->link0 +90deg) 로 둘을 연결한다.
ROBOT_ORIGIN = (0.0, 0.0, 0.0)

# ---- EoAT 형상 (tcp -> AFT200 -> roller) --------------------------------------
# CAD 는 +Z 로 뻗지만 실제 장착은 TCP local -Y.
TOOL_AXIS = "-y"

# AFT200 collision STL 기반. CAD +Z 52.2mm 를 TCP -Y 로 회전해 사용.
AFT200_LENGTH = 0.0522
AFT200_COLLISION_STL_PATH = "/home/minjea/Downloads/aft200_description/meshes/collision/aft200.stl"
AFT200_SIZE = (0.104, AFT200_LENGTH, 0.082)  # TCP frame box size: x, y, z
AFT200_CENTER = (-0.0116, -AFT200_LENGTH / 2.0, 0.0)

# RR-00A_B__EOAT.step 의 CAD +Z forward reach. AFT200 뒤에 붙는 롤러 중심까지.
EOAT_NO_CAMERA_COLLISION_STL_PATH = (
    "/home/minjea/sketch_robot_ws/src/eoat_description/meshes/"
    "rr_00a_b_eoat_no_camera_collision.stl"
)
EOAT_MESH_FORWARD_LENGTH = 0.2305
EOAT_MESH_CENTER_OFFSET = np.array([
    0.0,
    -(AFT200_LENGTH + EOAT_MESH_FORWARD_LENGTH / 2.0),
    0.0,
], dtype=float)
ROLLER_FORWARD_REACH = 0.209475
ROLLER_SUPPORT_RADIUS = 0.012
ROLLER_LENGTH = 0.18
ROLLER_RADIUS = 0.025
ROLLER_LONG_AXIS = "+x"  # 벽면 가로(real base Y) 방향.

# TCP → 롤러 회전축 중심까지의 거리.
# Cartesian / IK 가 "tip" 으로 삼는 점 = 롤러 회전축 중심.
EOAT_TIP_OFFSET = AFT200_LENGTH + ROLLER_FORWARD_REACH
EOAT_TOTAL_REACH = EOAT_TIP_OFFSET

# Intel RealSense D405 attached on the roller EOAT.
# Camera front(+X in RealSense camera_link) points along TCP local -Y.
D405_SIZE = (0.042, 0.023, 0.042)  # TCP frame bbox: x(width), y(depth), z(height)
D405_COLLISION_CENTER = (0.0, -0.06870, 0.04375)

EOAT_TOUCH_LINKS = ["tcp", "link6"]


def _cylinder_axis_quat(axis):
    """SolidPrimitive.CYLINDER (default +z) 를 axis 방향으로 회전시키는 quaternion (x,y,z,w).
    shortest-arc quaternion 으로 동적 계산하여 부호 실수 방지."""
    target = {"+x": [1, 0, 0], "-x": [-1, 0, 0],
              "+y": [0, 1, 0], "-y": [0, -1, 0],
              "+z": [0, 0, 1], "-z": [0, 0, -1]}[axis]
    src = np.array([0.0, 0.0, 1.0])
    tgt = np.array(target, dtype=float)
    if np.allclose(src, tgt):
        return (0.0, 0.0, 0.0, 1.0)
    if np.allclose(src, -tgt):
        # 180도 뒤집기 — 회전축은 src 와 직교한 임의 축. X 선택.
        return (1.0, 0.0, 0.0, 0.0)
    q = quat_from_two_vectors(src, tgt)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _axis_offset(axis, distance):
    """axis ('+x'/'-y'/...) 방향으로 distance 만큼 떨어진 점 (x, y, z)."""
    sign = -1.0 if axis.startswith("-") else 1.0
    a = axis[1]
    d = distance * sign
    if a == "x":
        return (d, 0.0, 0.0)
    if a == "y":
        return (0.0, d, 0.0)
    if a == "z":
        return (0.0, 0.0, d)
    raise ValueError(f"unknown axis: {axis}")


_AFT200_MESH_CACHE = None
_EOAT_NO_CAMERA_MESH_CACHE = None


def _cad_z_to_tcp_minus_y_np(x, y, z):
    """CAD +Z forward mesh vertex -> TCP local -Y frame."""
    return np.array([float(x), float(-z), float(y)], dtype=float)


def _load_stl_mesh(stl_path, transform_vertex):
    with open(stl_path, "rb") as f:
        data = f.read()
    if len(data) < 84:
        raise RuntimeError(f"STL 파일이 너무 짧음: {stl_path}")

    tri_count = struct.unpack("<I", data[80:84])[0]
    expected_len = 84 + tri_count * 50
    if expected_len != len(data):
        raise RuntimeError(f"binary STL 길이 불일치: {stl_path}")

    mesh = Mesh()
    off = 84
    for _ in range(tri_count):
        off += 12  # normal
        idx = []
        for _v in range(3):
            x, y, z = struct.unpack("<fff", data[off:off + 12])
            off += 12
            p = transform_vertex(x, y, z)
            mesh.vertices.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
            idx.append(len(mesh.vertices) - 1)
        off += 2
        tri = MeshTriangle()
        tri.vertex_indices = idx
        mesh.triangles.append(tri)
    return mesh


def _load_aft200_mesh():
    """Load AFT200 collision STL as a MoveIt mesh in tcp frame."""
    global _AFT200_MESH_CACHE
    if _AFT200_MESH_CACHE is not None:
        return copy.deepcopy(_AFT200_MESH_CACHE)

    mesh = _load_stl_mesh(AFT200_COLLISION_STL_PATH, _cad_z_to_tcp_minus_y_np)
    _AFT200_MESH_CACHE = mesh
    return copy.deepcopy(mesh)


def _load_eoat_no_camera_mesh():
    """Load no-camera EOAT collision STL as a MoveIt mesh in tcp frame."""
    global _EOAT_NO_CAMERA_MESH_CACHE
    if _EOAT_NO_CAMERA_MESH_CACHE is not None:
        return copy.deepcopy(_EOAT_NO_CAMERA_MESH_CACHE)

    mesh = _load_stl_mesh(
        EOAT_NO_CAMERA_COLLISION_STL_PATH,
        lambda x, y, z: _cad_z_to_tcp_minus_y_np(x, y, z) + EOAT_MESH_CENTER_OFFSET,
    )
    _EOAT_NO_CAMERA_MESH_CACHE = mesh
    return copy.deepcopy(mesh)


class MoveItExecutor(Node):
    def __init__(self):
        super().__init__("moveit_executor")

        # I/O
        self.create_subscription(PoseArray, "/sketch_waypoints", self.on_waypoints, 10)
        self.create_subscription(Bool, "/sketch_execute", self.on_execute, 10)
        self.create_subscription(JointState, "/joint_states", self.on_joint_state, 10)
        self.create_subscription(
            MarkerArray, OBSTACLES_TOPIC, self.on_dynamic_obstacles, 10)
        self.create_subscription(
            PoseArray, PLANES_TOPIC, self.on_perception_planes, 10)
        self.create_subscription(
            String, PLANE_LABELS_TOPIC, self.on_plane_labels, 10)
        self.create_subscription(
            PoseStamped, WORK_AREA_PLANE_TOPIC, self.on_active_surface, LATCHED_QOS)
        self.create_subscription(
            PoseStamped, WORK_AREA_REFINED_PLANE_TOPIC,
            self.on_refined_active_surface, LATCHED_QOS)
        self.create_subscription(
            PoseArray, WORK_AREA_CORNERS_TOPIC, self.on_work_area_corners, LATCHED_QOS)
        self.create_subscription(String, FT_STATUS_TOPIC, self.on_ft_status, 10)
        # 디버그용 — 실로봇 검증 시 Stage 5 단독 호출용
        self.create_subscription(
            Bool, "/debug_trigger_stage5", self.on_debug_trigger_stage5, 10)
        # 디버그용 — 실로봇 검증 시 Stage 1 단독 호출용
        self.create_subscription(
            Bool, "/debug_trigger_stage1", self.on_debug_trigger_stage1, 10)
        # 디버그용 — motion pipeline 단독 검증 (OMPL/IK 우회 jog)
        self.create_subscription(
            Bool, "/debug_trigger_jog", self.on_debug_trigger_jog, 10)
        # 수동 프리셋 이동 — perception 전에 카메라 시야 확보/캘리브 자세 복귀용.
        self.create_subscription(
            String, "/robot_pose_preset", self.on_robot_pose_preset, 10)
        self.create_subscription(
            Bool, "/go_ready_pose", self.on_go_ready_pose, 10)
        self.create_subscription(
            Bool, "/go_calibration_pose", self.on_go_calibration_pose, 10)
        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)
        self.joint_cmd_pub = self.create_publisher(
            JointState, JOINT_COMMAND_TOPIC, 10)
        self.ft_zero_pub = self.create_publisher(Bool, FT_ZERO_TOPIC, 10)

        # MoveIt endpoints (계획만 사용)
        self.cartesian_client = self.create_client(
            GetCartesianPath, "/compute_cartesian_path")
        self.ik_client = self.create_client(
            GetPositionIK, "/compute_ik")

        # MoveGroup action client (Stage 1: 자유 경로 planning)
        self.move_action_client = ActionClient(self, MoveGroup, "/move_action")

        # ApplyPlanningScene service client (실제 MoveIt collision detection 등록)
        self.apply_scene_client = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene")

        # FollowJointTrajectory action client (RB10 driver 의 joint_trajectory_controller)
        self.traj_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )

        # 다음 stage 로 넘기는 데 쓰는 상태
        self._safety_tcp_pose = None
        self._retreat_tcp_pose = None
        self._stage3_tcp_wps = None  # Stage 2 끝났을 때 stage 3 가 쓸 waypoints
        self._stage3_tip_wps = None
        self._stage1_retried = False
        self._stage1_goal_constraints = None
        self._joint_goal_context = None

        self.current_waypoints = []
        self.current_joint_state = None
        self._joint_command_timer = None
        self.scene_initialized = False
        self.scene_confirmed = False
        self.executing = False
        self.dynamic_obstacles = []
        self._dynamic_obstacle_ids = set()
        self._stale_dynamic_obstacle_ids = set()
        self._dynamic_obstacle_signature = None
        self.perception_planes = []
        self.perception_plane_labels = []
        self.dynamic_surface_point = None
        self.dynamic_surface_normal = None
        self.dynamic_surface_source = "fallback"
        self.dynamic_surface_source_time = 0.0
        self.dynamic_work_area_corners = None
        self._pending_surface_msg = None
        self._pending_surface_source = "zed"
        self._pending_corners_msg = None
        self.ft_normal_force_n = None
        self.ft_contact = False
        self.ft_bias_ready = False
        self.ft_abort_force_n = FT_ABORT_FORCE_N
        self.ft_status_time = 0.0
        self.ft_state = "unknown"
        self._ft_auto_zero_timer = None

        # ---- Target registry ----
        self.cfg = load_objects_config()
        self.active_target_name = self.cfg.get("active_target", "wall")
        self._enabled_ids = set(
            o["name"] for o in self.cfg["objects"] if o.get("enabled", True)
        )

        # PlanningScene 모니터링
        self.create_subscription(
            PlanningScene, "/monitored_planning_scene",
            self.on_scene_update, 10)

        self.create_timer(1.0, self.publish_scene_periodic)

        # ---- 진단: 현재 tcp TF 를 3초마다 출력 (수동 캘리브레이션 용) ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(0.5, self._retry_pending_surface_tf)
        self.create_timer(3.0, self._log_current_tcp)

        self.get_logger().info(
            f"MoveIt Executor 노드 시작 "
            f"(planning=MoveIt, execution={EXECUTION_BACKEND})")

    def _log_current_tcp(self):
        """world → tcp TF 를 주기적으로 로그. 수동 캘리브레이션 시 사용."""
        try:
            tf = self.tf_buffer.lookup_transform(
                "world", EE_LINK, rclpy.time.Time(),
                timeout=Duration(seconds=0.3),
            )
        except Exception:
            # world 프레임 없으면 World 대문자 시도
            try:
                tf = self.tf_buffer.lookup_transform(
                    "World", EE_LINK, rclpy.time.Time(),
                    timeout=Duration(seconds=0.3),
                )
            except Exception:
                return

        q = tf.transform.rotation
        p = tf.transform.translation
        x, y, z, w = q.x, q.y, q.z, q.w
        local_x = (1 - 2 * (y * y + z * z),
                   2 * (x * y + z * w),
                   2 * (x * z - y * w))
        local_y = (2 * (x * y - z * w),
                   1 - 2 * (x * x + z * z),
                   2 * (y * z + x * w))
        local_z = (2 * (x * z + y * w),
                   2 * (y * z - x * w),
                   1 - 2 * (x * x + y * y))
        self.get_logger().info(
            f"[TCP_NOW] pos=({p.x:+.3f},{p.y:+.3f},{p.z:+.3f}) "
            f"quat=({q.x:+.3f},{q.y:+.3f},{q.z:+.3f},{q.w:+.3f})"
        )
        self.get_logger().info(
            f"          local_X_in_world=({local_x[0]:+.2f},{local_x[1]:+.2f},{local_x[2]:+.2f})"
        )
        self.get_logger().info(
            f"          local_Y_in_world=({local_y[0]:+.2f},{local_y[1]:+.2f},{local_y[2]:+.2f})"
        )
        self.get_logger().info(
            f"          local_Z_in_world=({local_z[0]:+.2f},{local_z[1]:+.2f},{local_z[2]:+.2f})"
        )

    # ---- callbacks ----------------------------------------------------------
    def on_waypoints(self, msg: PoseArray):
        # sketch_to_waypoints 는 Isaac World 좌표로 waypoint 를 만든다.
        # MoveIt planning frame 은 URDF link0 이다. World/world 는 실제 RB10 base 이므로
        # static TF(world->link0 +90deg) 변환을 거쳐 보관한다.
        frame = self._canonical_world_frame(msg.header.frame_id or BASE_FRAME)
        transform = None
        if frame != BASE_FRAME:
            transform = self._lookup_transform_to_base(frame, timeout_s=0.5)
            if transform is None:
                self.get_logger().error(
                    f"/sketch_waypoints TF 실패 ({BASE_FRAME}<-{frame}) "
                    "-> waypoint 폐기")
                return

        converted = []
        for p in msg.poses:
            bp = self._transform_pose_msg_to_base(p, transform)
            converted.append(bp)
        self.current_waypoints = converted
        self.get_logger().info(
            f"{len(self.current_waypoints)}개 웨이포인트 수신 "
            f"({frame}->{BASE_FRAME}: 첫 점 "
            f"x={self.current_waypoints[0].position.x:.2f} "
            f"y={self.current_waypoints[0].position.y:.2f} "
            f"z={self.current_waypoints[0].position.z:.2f})")

    def on_joint_state(self, msg: JointState):
        self.current_joint_state = msg

    def on_ft_status(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        self.ft_state = str(payload.get("state", "unknown"))
        self.ft_status_time = time.monotonic()
        if "bias_ready" in payload:
            self.ft_bias_ready = bool(payload.get("bias_ready", False))
        if not payload.get("ok", False):
            self.ft_normal_force_n = None
            self.ft_contact = False
            return
        self.ft_normal_force_n = float(payload.get("normal_force_n", 0.0))
        self.ft_contact = bool(payload.get("contact", False))
        self.ft_abort_force_n = float(
            payload.get("abort_force_n", FT_ABORT_FORCE_N))

    def on_active_surface(self, msg: PoseStamped):
        if (
            self.dynamic_surface_source == "d405_refined"
            and time.monotonic() - self.dynamic_surface_source_time <= 2.0
        ):
            return
        if self.executing:
            self.get_logger().warn(
                "[SURFACE] 실행 중 ZED surface 갱신 무시 "
                "(현재 plan/collision 기준 고정)",
                throttle_duration_sec=2.0)
            return
        self._set_active_surface(msg, "zed")

    def on_refined_active_surface(self, msg: PoseStamped):
        if self.executing:
            self.get_logger().warn(
                "[SURFACE] 실행 중 D405 refined surface 갱신 무시 "
                "(다음 plan부터 적용)",
                throttle_duration_sec=2.0)
            return
        self._set_active_surface(msg, "d405_refined")

    def _set_active_surface(self, msg: PoseStamped, source: str):
        frame = msg.header.frame_id or BASE_FRAME
        pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=float)
        q_msg = msg.pose.orientation
        normal_local = quat_apply([
            q_msg.x, q_msg.y, q_msg.z, q_msg.w
        ], [0.0, 0.0, 1.0])

        frame = self._canonical_world_frame(frame)
        if frame == BASE_FRAME:
            point = pos
            normal = normal_local
        else:
            tf = self._lookup_transform_to_base(frame, timeout_s=0.2)
            if tf is None:
                self._pending_surface_msg = msg
                self._pending_surface_source = source
                self.get_logger().warn(
                    f"[SURFACE] TF 실패 ({BASE_FRAME}<-{frame})",
                    throttle_duration_sec=2.0)
                return
            t = tf.transform.translation
            q = tf.transform.rotation
            q_tf = [q.x, q.y, q.z, q.w]
            point = quat_apply(q_tf, pos) + np.array([t.x, t.y, t.z])
            normal = quat_apply(q_tf, normal_local)

        normal = np.asarray(normal, dtype=float)
        normal /= np.linalg.norm(normal) + 1e-12
        self.dynamic_surface_point = point
        self.dynamic_surface_normal = normal
        self.dynamic_surface_source = source
        self.dynamic_surface_source_time = time.monotonic()
        self._pending_surface_msg = None
        self.scene_confirmed = False
        self.scene_initialized = False
        self.get_logger().info(
            f"[SURFACE] dynamic plane 갱신({source}): point=({point[0]:+.3f},"
            f"{point[1]:+.3f},{point[2]:+.3f}) normal=({normal[0]:+.2f},"
            f"{normal[1]:+.2f},{normal[2]:+.2f})",
            throttle_duration_sec=2.0)

    def on_work_area_corners(self, msg: PoseArray):
        if self.executing:
            self.get_logger().warn(
                "[SURFACE] 실행 중 work_area corners 갱신 무시 "
                "(현재 plan/collision 기준 고정)",
                throttle_duration_sec=2.0)
            return
        if len(msg.poses) < 4:
            return
        frame = msg.header.frame_id or BASE_FRAME
        pts = np.array([
            [p.position.x, p.position.y, p.position.z]
            for p in msg.poses[:4]
        ], dtype=float)
        frame = self._canonical_world_frame(frame)
        if frame == BASE_FRAME:
            self.dynamic_work_area_corners = pts
            self.scene_confirmed = False
            self.scene_initialized = False
            return
        tf = self._lookup_transform_to_base(frame, timeout_s=0.2)
        if tf is None:
            self._pending_corners_msg = msg
            self.get_logger().warn(
                f"[SURFACE] corners TF 실패 ({BASE_FRAME}<-{frame})",
                throttle_duration_sec=2.0)
            return
        t = tf.transform.translation
        q = tf.transform.rotation
        self.dynamic_work_area_corners = (
            quat_apply([q.x, q.y, q.z, q.w], pts)
            + np.array([t.x, t.y, t.z])
        )
        self._pending_corners_msg = None
        self.scene_confirmed = False
        self.scene_initialized = False

    def _retry_pending_surface_tf(self):
        """Surface/corner 메시지를 TF 준비 전에 받았을 때 나중에 다시 변환."""
        if self._pending_surface_msg is not None:
            self._set_active_surface(
                self._pending_surface_msg,
                self._pending_surface_source,
            )
        if self._pending_corners_msg is not None:
            self.on_work_area_corners(self._pending_corners_msg)

    def on_perception_planes(self, msg: PoseArray):
        frame = self._canonical_world_frame(msg.header.frame_id or BASE_FRAME)
        transform = None
        if frame != BASE_FRAME:
            transform = self._lookup_transform_to_base(frame, timeout_s=0.2)
            if transform is None:
                self.get_logger().warn(
                    f"[SCENE] planes TF 실패 ({BASE_FRAME}<-{frame})",
                    throttle_duration_sec=2.0)
                return

        planes = []
        for idx, pose in enumerate(msg.poses):
            point = np.array([
                pose.position.x,
                pose.position.y,
                pose.position.z,
            ], dtype=float)
            q_pose = [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
            normal = quat_apply(q_pose, [0.0, 0.0, 1.0])
            if transform is not None:
                t = transform.transform.translation
                q = transform.transform.rotation
                q_tf = [q.x, q.y, q.z, q.w]
                point = quat_apply(q_tf, point) + np.array([t.x, t.y, t.z])
                normal = quat_apply(q_tf, normal)
            normal = np.asarray(normal, dtype=float)
            normal /= np.linalg.norm(normal) + 1e-12
            planes.append({
                "index": idx,
                "point": point,
                "normal": normal,
            })

        self.perception_planes = planes
        if planes:
            self.scene_confirmed = False
            self.scene_initialized = False
            self.get_logger().info(
                f"[SCENE] perception planes 갱신: {len(planes)}개",
                throttle_duration_sec=2.0)

    def on_plane_labels(self, msg: String):
        try:
            payload = json.loads(msg.data)
            labels = payload.get("planes", [])
            if not isinstance(labels, list):
                raise ValueError("planes is not a list")
        except Exception as e:
            self.get_logger().warn(f"[SCENE] plane_labels parse 실패: {e}")
            return
        self.perception_plane_labels = labels
        if self.perception_planes:
            self.scene_confirmed = False
            self.scene_initialized = False

    def on_dynamic_obstacles(self, msg: MarkerArray):
        obstacles = []
        new_ids = set()
        count = 0
        for marker in msg.markers:
            if marker.action != Marker.ADD or marker.type != Marker.CUBE:
                continue
            if count >= MAX_DYNAMIC_OBSTACLES:
                break
            if marker.scale.x <= 0.0 or marker.scale.y <= 0.0 or marker.scale.z <= 0.0:
                continue
            converted = self._marker_to_base_collision(marker, count)
            if converted is None:
                continue
            if self._is_robot_self_obstacle(
                    converted["position"], converted["size"]):
                continue
            obstacles.append(converted)
            new_ids.add(converted["id"])
            count += 1

        stale = self._dynamic_obstacle_ids - new_ids
        signature = tuple(
            (
                ob["id"],
                *(round(float(v), 3) for v in ob["position"]),
                *(round(float(v), 3) for v in ob["size"]),
            )
            for ob in obstacles
        )
        if (
            stale
            or new_ids != self._dynamic_obstacle_ids
            or signature != self._dynamic_obstacle_signature
        ):
            self.dynamic_obstacles = obstacles
            self._stale_dynamic_obstacle_ids.update(stale)
            self._dynamic_obstacle_ids = new_ids
            self._dynamic_obstacle_signature = signature
            self.scene_confirmed = False
            self.scene_initialized = False
            self.get_logger().info(
                f"[SCENE] ZED dynamic obstacles 갱신: {len(obstacles)}개")

    def _marker_to_base_collision(self, marker, index):
        frame = marker.header.frame_id or BASE_FRAME
        pos = np.array([
            marker.pose.position.x,
            marker.pose.position.y,
            marker.pose.position.z,
        ], dtype=float)
        q_marker = np.array([
            marker.pose.orientation.x,
            marker.pose.orientation.y,
            marker.pose.orientation.z,
            marker.pose.orientation.w,
        ], dtype=float)
        if np.linalg.norm(q_marker) < 1e-9:
            q_marker = np.array([0.0, 0.0, 0.0, 1.0])

        frame = self._canonical_world_frame(frame)
        if frame == BASE_FRAME:
            base_pos = pos
            base_q = q_marker
        else:
            tf = self._lookup_transform_to_base(frame, timeout_s=0.2)
            if tf is None:
                self.get_logger().warn(
                    f"[SCENE] obstacle TF 실패 ({BASE_FRAME}<-{frame})")
                return None
            t = tf.transform.translation
            q = tf.transform.rotation
            q_tf = [q.x, q.y, q.z, q.w]
            base_pos = quat_apply(q_tf, pos) + np.array([t.x, t.y, t.z])
            base_q = quat_multiply(q_tf, q_marker)

        return {
            "id": f"{DYNAMIC_OBSTACLE_PREFIX}{index:03d}",
            "position": base_pos,
            "orientation": base_q,
            "size": np.array([
                marker.scale.x,
                marker.scale.y,
                marker.scale.z,
            ], dtype=float),
        }

    @staticmethod
    def _canonical_world_frame(frame):
        # Isaac Sim publishes the physical world as "World". Treat lowercase
        # "world" from legacy UI/perception code as the same physical frame,
        # not as MoveIt's virtual SRDF frame.
        return "World" if frame == "world" else frame

    def _lookup_transform_to_base(self, source_frame, timeout_s=0.2):
        try:
            return self.tf_buffer.lookup_transform(
                BASE_FRAME,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=timeout_s),
            )
        except Exception:
            return None

    @staticmethod
    def _transform_pose_msg_to_base(pose, transform):
        if transform is None:
            return copy.deepcopy(pose)
        return MoveItExecutor._transform_xyz_quat_to_pose(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
            ],
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            transform,
        )

    @staticmethod
    def _transform_xyz_quat_to_pose(position, orientation, transform):
        t = transform.transform.translation
        q = transform.transform.rotation
        q_tf = [q.x, q.y, q.z, q.w]
        pos = np.asarray(position, dtype=float)
        base_pos = quat_apply(q_tf, pos) + np.array([t.x, t.y, t.z])
        base_q = quat_multiply(q_tf, np.asarray(orientation, dtype=float))
        out = Pose()
        out.position.x = float(base_pos[0])
        out.position.y = float(base_pos[1])
        out.position.z = float(base_pos[2])
        out.orientation.x = float(base_q[0])
        out.orientation.y = float(base_q[1])
        out.orientation.z = float(base_q[2])
        out.orientation.w = float(base_q[3])
        return out

    def _is_robot_self_obstacle(self, point_base, size=None):
        """로봇 본체 표면을 ZED dynamic obstacle 로 재등록하지 않도록 차단."""
        extra = 0.0
        if size is not None:
            extra = 0.5 * float(np.linalg.norm(np.asarray(size, dtype=float)))

        frames = self._lookup_robot_link_points()
        if len(frames) < 3:
            self.get_logger().warn(
                "[SCENE] robot self-filter TF 부족 -> dynamic obstacle skip",
                throttle_duration_sec=2.0)
            return True

        p = np.asarray(point_base, dtype=float)
        for pair, radius in ROBOT_LINK_CAPSULE_RADIUS.items():
            if pair[0] not in frames or pair[1] not in frames:
                continue
            dist = self._distance_to_segment(p, frames[pair[0]], frames[pair[1]])
            if dist <= radius + ROBOT_SELF_FILTER_PADDING + extra:
                return True
        return False

    def _lookup_robot_link_points(self):
        out = {}
        for frame in ROBOT_LINK_FRAMES:
            if frame == BASE_FRAME:
                out[frame] = np.zeros(3, dtype=float)
                continue
            try:
                tf = self.tf_buffer.lookup_transform(
                    BASE_FRAME, frame, rclpy.time.Time(),
                    timeout=Duration(seconds=0.05))
            except Exception:
                continue
            t = tf.transform.translation
            out[frame] = np.array([t.x, t.y, t.z], dtype=float)
        return out

    @staticmethod
    def _distance_to_segment(point, a, b):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            return float(np.linalg.norm(point - a))
        t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
        closest = a + t * ab
        return float(np.linalg.norm(point - closest))

    def on_execute(self, msg: Bool):
        if not msg.data or not self.current_waypoints:
            self.get_logger().warn("실행할 웨이포인트가 없습니다")
            return
        if self.current_joint_state is None:
            self.get_logger().warn("joint_state 미수신 -> 실행 보류")
            return
        if self.executing:
            self.get_logger().warn("이미 실행 중")
            return

        if not self._joint_state_within_limits(
                self.current_joint_state, "SKETCH start"):
            self.get_logger().error(
                "현재 joint_state 가 MoveIt/실로봇 limit 밖입니다. "
                "스케치 실행을 중단합니다.")
            return

        # 첫 Submit 안전성: READY_POSE 가 아니면 먼저 collision-aware joint plan 으로
        # READY_POSE 로 복귀한 뒤 같은 sketch execute 를 다시 시작한다.
        if START_FROM_READY_BEFORE_SKETCH and not self._is_at_ready_pose():
            self.get_logger().warn(
                "현재 자세가 READY_POSE 와 다름 -> READY_POSE 먼저 이동 후 "
                "스케치 제어를 시작합니다.")
            self.executing = True
            self._plan_joint_goal(
                "PRE_SKETCH_READY_POSE",
                READY_POSE_JOINTS,
                STAGE5_SPEED_SCALE,
                finalize_cb=self._pre_sketch_ready_done,
            )
            return

        self._sync_active_target_plane_from_waypoints(self.current_waypoints)

        # Stage 2/3 가 쓸 롤러 중심 + tcp waypoints 미리 계산
        densified_tip, tcp_wps, target, n = self._compute_snapped_tcp_waypoints()
        if not self._validate_contact_waypoints(densified_tip, target):
            self.get_logger().error(
                "waypoint safety validation 실패 -> 실행 중단")
            return
        if self.ft_status_time <= 0.0:
            self.get_logger().warn(
                "[FT GUARD] /ft/status 미수신. 이번 실행은 힘 기반 "
                "과압 정지 없이 진행됩니다.")
        elif time.monotonic() - self.ft_status_time > FT_AUTO_ZERO_STALE_SEC:
            self.get_logger().warn(
                "[FT GUARD] /ft/status 가 오래됨. 이번 실행은 힘 기반 "
                "과압 정지 없이 진행될 수 있습니다.")
        elif not self.ft_bias_ready:
            self.get_logger().warn(
                "[FT GUARD] FT bias not ready. 스케치 시작 전 자동 zero 를 "
                "시도합니다.")
        self._stage3_tip_wps = densified_tip
        self._stage3_tcp_wps = tcp_wps

        # Stage 1 의 목표: 첫 롤러 중심점 → normal 방향 SAFETY_OFFSET 후퇴
        fixed_q = self._active_ee_quat(target)
        first_tip = densified_tip[0]
        safety_tip = self._offset_along_normal(first_tip, SAFETY_OFFSET)
        safety_tcp = self._brush_tip_to_tcp(safety_tip)
        safety_tcp.orientation.x = float(fixed_q[0])
        safety_tcp.orientation.y = float(fixed_q[1])
        safety_tcp.orientation.z = float(fixed_q[2])
        safety_tcp.orientation.w = float(fixed_q[3])
        self._safety_tcp_pose = safety_tcp

        # Stage 4 의 목표: 마지막 롤러 중심점 → normal 방향 RETREAT_OFFSET 후퇴
        last_tip = densified_tip[-1]
        retreat_tip = self._offset_along_normal(last_tip, RETREAT_OFFSET)
        retreat_tcp = self._brush_tip_to_tcp(retreat_tip)
        retreat_tcp.orientation.x = float(fixed_q[0])
        retreat_tcp.orientation.y = float(fixed_q[1])
        retreat_tcp.orientation.z = float(fixed_q[2])
        retreat_tcp.orientation.w = float(fixed_q[3])
        self._retreat_tcp_pose = retreat_tcp

        self.get_logger().info("=" * 60)
        self.get_logger().info("=== STAGE 1: free-space approach (OMPL) ===")
        self.get_logger().info(
            f"safety_tcp=({safety_tcp.position.x:.3f},"
            f"{safety_tcp.position.y:.3f},{safety_tcp.position.z:.3f})")
        self.executing = True
        self._begin_stage1_with_optional_ft_zero()

    def _cancel_ft_auto_zero_timer(self):
        if self._ft_auto_zero_timer is not None:
            self._ft_auto_zero_timer.cancel()
            self.destroy_timer(self._ft_auto_zero_timer)
            self._ft_auto_zero_timer = None

    def _begin_stage1_with_optional_ft_zero(self):
        """Sketch 실행 직전 F/T bias 를 자동 갱신한다.

        센서가 아예 없거나 /ft/status 가 안 들어오는 Isaac-only 테스트에서는
        기존처럼 진행한다. 센서가 살아 있으면 접촉 없는 상태에서 zero 완료를
        기다린 뒤 Stage 1 을 시작한다.
        """
        if not FT_AUTO_ZERO_BEFORE_SKETCH:
            self.stage1_approach_free()
            return

        now = time.monotonic()
        if self.ft_status_time <= 0.0:
            self.get_logger().warn(
                "[FT ZERO] /ft/status 미수신 -> 자동 zero 생략")
            self.stage1_approach_free()
            return
        if now - self.ft_status_time > FT_AUTO_ZERO_STALE_SEC:
            self.get_logger().warn(
                "[FT ZERO] /ft/status stale -> 자동 zero 생략")
            self.stage1_approach_free()
            return
        if self.ft_contact:
            self.get_logger().error(
                "[FT ZERO] 이미 접촉 상태로 판단됨 -> zero 금지, 실행 중단. "
                "EOAT 를 작업면에서 떼고 다시 실행하세요.")
            self.executing = False
            return

        self._cancel_ft_auto_zero_timer()
        request_time = time.monotonic()
        msg = Bool()
        msg.data = True
        self.ft_zero_pub.publish(msg)
        self.ft_bias_ready = False
        self.ft_normal_force_n = None
        self.ft_contact = False
        self.ft_state = "zero_requested"
        self.get_logger().info(
            f"[FT ZERO] 스케치 시작 전 자동 zero 요청 "
            f"(timeout={FT_AUTO_ZERO_TIMEOUT_SEC:.1f}s)")

        def _wait_zero_done():
            now_inner = time.monotonic()
            if (
                self.ft_bias_ready
                and self.ft_status_time >= request_time
                and now_inner - self.ft_status_time <= FT_AUTO_ZERO_STALE_SEC
            ):
                self._cancel_ft_auto_zero_timer()
                self.get_logger().info("[FT ZERO] 완료 -> STAGE 1 시작")
                self.stage1_approach_free()
                return

            if now_inner - request_time > FT_AUTO_ZERO_TIMEOUT_SEC:
                self._cancel_ft_auto_zero_timer()
                self.executing = False
                self.get_logger().error(
                    "[FT ZERO] timeout -> 실행 중단. "
                    "AFT200 wrench 토픽과 /ft/status 를 확인하세요.")

        self._ft_auto_zero_timer = self.create_timer(
            FT_AUTO_ZERO_CHECK_PERIOD, _wait_zero_done)

    def _pre_sketch_ready_done(self, success):
        self._joint_goal_context = None
        self.executing = False
        if not success:
            self.get_logger().error(
                "PRE_SKETCH_READY_POSE 이동 실패 -> 스케치 실행 중단. "
                "RViz 현재 자세가 joint limit/충돌/도달성 조건을 만족하는지 확인 필요.")
            return
        self.get_logger().info(
            "PRE_SKETCH_READY_POSE 완료 -> 같은 스케치 경로 실행을 시작합니다.")
        timer_ref = {}

        def _restart_after_joint_state_update():
            timer = timer_ref.pop("timer", None)
            if timer is not None:
                timer.cancel()
                self.destroy_timer(timer)
            msg = Bool()
            msg.data = True
            self.on_execute(msg)

        timer_ref["timer"] = self.create_timer(
            0.25, _restart_after_joint_state_update)

    def on_debug_trigger_stage1(self, msg: Bool):
        """디버그용 — 실로봇 검증 시 Stage 1 (자유공간 approach) 단독 호출용.

        current_waypoints[0] 을 surface waypoint 로 보고, surface normal 방향으로
        DEBUG_STAGE1_OFFSET 만큼 떨어진 safe_pose 계산.
        현재 joint state → safe_pose 를 Stage 1 planner 로 plan, 성공 시 실행.
        Stage 2~5 는 트리거하지 않는다.

        publish_test_waypoint 로 /sketch_waypoints 먼저 publish 한 후 실행할 것.
        """
        if not msg.data:
            return
        if self.executing:
            self.get_logger().warn(
                "이미 실행 중 -> /debug_trigger_stage1 무시")
            return
        if not self.current_waypoints:
            self.get_logger().error(
                "/sketch_waypoints 가 비어있음. "
                "publish_test_waypoint 먼저 실행 필요.")
            return
        if self.current_joint_state is None:
            self.get_logger().error("joint_state 미수신 -> 실행 보류")
            return

        self.get_logger().info(
            "[DEBUG] /debug_trigger_stage1 수신 -> Stage 1 단독 실행")

        waypoint = self.current_waypoints[0]

        nx, ny, nz = DEBUG_STAGE1_SURFACE_NORMAL
        safe_pose = Pose()
        safe_pose.position.x = waypoint.position.x + DEBUG_STAGE1_OFFSET * nx
        safe_pose.position.y = waypoint.position.y + DEBUG_STAGE1_OFFSET * ny
        safe_pose.position.z = waypoint.position.z + DEBUG_STAGE1_OFFSET * nz
        safe_pose.orientation = copy.deepcopy(waypoint.orientation)

        self.get_logger().info(
            f"[DEBUG] Stage 1 target safe_pose: "
            f"({safe_pose.position.x:.3f}, {safe_pose.position.y:.3f}, "
            f"{safe_pose.position.z:.3f})"
        )
        self.get_logger().info(
            f"[DEBUG] safe_pose orientation (xyzw): "
            f"({safe_pose.orientation.x:.4f}, {safe_pose.orientation.y:.4f}, "
            f"{safe_pose.orientation.z:.4f}, {safe_pose.orientation.w:.4f})"
        )

        self._safety_tcp_pose = safe_pose
        self.executing = True
        self.stage1_approach_free(on_complete=self._debug_stage1_done)

    def _debug_stage1_done(self):
        self.get_logger().info("[DEBUG] Stage 1 단독 실행 완료")
        self.executing = False

    def on_debug_trigger_jog(self, msg: Bool):
        """매우 작은 joint-space 동작으로 motion pipeline 검증.

        OMPL / IK / Cartesian goal / planning scene 의존성 모두 없음.
        한 joint (wrist3) 에 JOG_DELTA_RAD 를 JOG_DURATION_SEC 동안 적용.
        Trajectory 직접 빌드 → execute_trajectory_direct 호출.
        """
        if not msg.data:
            return
        if self.executing:
            self.get_logger().warn(
                "이미 실행 중 -> /debug_trigger_jog 무시")
            return

        self.get_logger().info(
            "[DEBUG] /debug_trigger_jog 수신 -> joint-space jog 단독 실행")

        if self.current_joint_state is None or \
           len(self.current_joint_state.position) == 0:
            self.get_logger().error(
                "current_joint_state 미수신. /joint_states 흐름 확인.")
            return

        n = len(self.current_joint_state.position)
        if JOG_JOINT_INDEX >= n:
            self.get_logger().error(
                f"JOG_JOINT_INDEX={JOG_JOINT_INDEX} 가 joint 수({n}) 초과")
            return

        current = list(self.current_joint_state.position)
        target = list(current)
        target[JOG_JOINT_INDEX] = current[JOG_JOINT_INDEX] + JOG_DELTA_RAD

        target_joint_name = self.current_joint_state.name[JOG_JOINT_INDEX]
        self.get_logger().info(
            f"[DEBUG] jog target joint: {target_joint_name} "
            f"({current[JOG_JOINT_INDEX]:.4f} -> {target[JOG_JOINT_INDEX]:.4f}, "
            f"delta={JOG_DELTA_RAD:+.4f} rad)")
        self.get_logger().info(
            f"[DEBUG] jog duration: {JOG_DURATION_SEC}s, "
            f"points: {JOG_NUM_POINTS}, "
            f"평균 각속도: {JOG_DELTA_RAD / JOG_DURATION_SEC:.4f} rad/s "
            f"(≈ {(JOG_DELTA_RAD / JOG_DURATION_SEC) * 57.3:.2f}°/s)")

        traj = JointTrajectory()
        traj.joint_names = list(self.current_joint_state.name)

        avg_v = JOG_DELTA_RAD / JOG_DURATION_SEC
        for i in range(JOG_NUM_POINTS):
            alpha = i / (JOG_NUM_POINTS - 1)  # 0.0 ~ 1.0
            point = JointTrajectoryPoint()
            point.positions = [
                current[j] + alpha * (target[j] - current[j])
                for j in range(n)
            ]
            if i == 0 or i == JOG_NUM_POINTS - 1:
                point.velocities = [0.0] * n
            else:
                point.velocities = [0.0] * n
                point.velocities[JOG_JOINT_INDEX] = avg_v
            t = alpha * JOG_DURATION_SEC
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t - int(t)) * 1e9)
            traj.points.append(point)

        rt = RobotTrajectory()
        rt.joint_trajectory = traj

        self.executing = True
        try:
            if self.execute_trajectory_direct(rt, on_complete=self._jog_done):
                self.get_logger().info("[DEBUG] jog trajectory sent.")
        except Exception as e:
            self.get_logger().error(f"jog trajectory 전송 실패: {e}")
            self.executing = False

    def _jog_done(self):
        """Jog 완료 콜백 (Stage chain 진입 없음, executing 플래그만 해제)."""
        self.get_logger().info("[DEBUG] jog 단독 실행 완료")
        self.executing = False

    def on_debug_trigger_stage5(self, msg: Bool):
        """디버그용 — 실로봇 검증 시 Stage 5 (READY_POSE 복귀) 단독 호출용.
        Stage 1~4 거치지 않고 바로 Stage 5 만 trigger."""
        if not msg.data:
            return
        if self.executing:
            self.get_logger().warn(
                "이미 실행 중 -> /debug_trigger_stage5 무시")
            return
        self.get_logger().info(
            "[DEBUG] /debug_trigger_stage5 수신 -> Stage 5 단독 실행")
        self.executing = True
        self.stage5_return_to_ready()

    def on_go_ready_pose(self, msg: Bool):
        if msg.data:
            self._start_preset_motion("ready")

    def on_go_calibration_pose(self, msg: Bool):
        if msg.data:
            self._start_preset_motion("calib")

    def on_robot_pose_preset(self, msg: String):
        name = (msg.data or "").strip().lower()
        if name.endswith("_pose"):
            name = name[:-5]
        self._start_preset_motion(name)

    def _start_preset_motion(self, name):
        if name not in PRESET_POSES:
            self.get_logger().warn(
                f"알 수 없는 pose preset '{name}'. "
                f"사용 가능: {sorted(PRESET_POSES.keys())}")
            return
        if self.executing:
            self.get_logger().warn(
                f"이미 실행 중 -> preset '{name}' 무시")
            return
        if self.current_joint_state is None:
            self.get_logger().warn(
                f"joint_state 미수신 -> preset '{name}' 실행 보류")
            return

        label, joints, speed_scale = PRESET_POSES[name]
        self.get_logger().info(
            f"[PRESET] '{name}' 요청 -> {label} "
            f"(speed_scale={speed_scale:.2f})")
        self.executing = True
        self._plan_joint_goal(
            label,
            joints,
            speed_scale,
            finalize_cb=lambda success: self._preset_motion_finalize(
                label, success),
        )

    def _preset_motion_finalize(self, label, success):
        self._joint_goal_context = None
        if success:
            self.get_logger().info(f"[PRESET] {label} 이동 완료")
        else:
            self.get_logger().warn(f"[PRESET] {label} 이동 실패")
        self.executing = False

    def on_scene_update(self, msg):
        """모니터링 용. scene_confirmed 는 ApplyPlanningScene 결과로 설정."""
        scene_ids = set(obj.id for obj in msg.world.collision_objects)
        objects_ok = self._enabled_ids.issubset(scene_ids)
        eoat_ok = any(
            ao.object.id == "eoat"
            for ao in msg.robot_state.attached_collision_objects)
        if objects_ok and eoat_ok and not getattr(self, "_monitor_logged", False):
            self._monitor_logged = True
            self.get_logger().info(
                f"[INFO] /monitored_planning_scene 에 {sorted(self._enabled_ids)} + eoat 보임 "
                "(apply 결과로 confirmed 됨)")

    # ---- PlanningScene (물체들 + EoAT AttachedCollisionObject) ----------------
    def publish_scene_periodic(self):
        if self.scene_confirmed:
            return

        ps = PlanningScene()
        ps.is_diff = True
        ps.world = PlanningSceneWorld()

        # 현재 executor 가 이전에 등록했던 obstacle 중 사라진 것만 제거한다.
        # 없는 object 를 매번 REMOVE 하면 /apply_planning_scene 이 success=False 를
        # 반환해서 scene 검증이 계속 실패한다.
        for stale_id in sorted(self._stale_dynamic_obstacle_ids):
            co = CollisionObject()
            co.id = stale_id
            co.header.frame_id = BASE_FRAME
            co.operation = CollisionObject.REMOVE
            ps.world.collision_objects.append(co)
        self._stale_dynamic_obstacle_ids.clear()

        # --- 활성 물체 전부 ---
        world_to_base = self._lookup_transform_to_base("World", timeout_s=0.05)
        if world_to_base is None:
            self.get_logger().warn(
                f"[SCENE] {BASE_FRAME}<-World TF 미수신 -> scene publish 보류",
                throttle_duration_sec=2.0)
            return

        for obj in self.cfg["objects"]:
            if not obj.get("enabled", True):
                continue
            if obj["name"] == self.active_target_name:
                dynamic_target = self._dynamic_target_collision_object(obj)
                if dynamic_target is not None:
                    ps.world.collision_objects.append(dynamic_target)
                    continue

            co = CollisionObject()
            co.id = obj["name"]
            co.header.frame_id = BASE_FRAME
            prim = SolidPrimitive()
            prim.type = SolidPrimitive.BOX
            padding = 0.0 if obj["name"] == self.active_target_name \
                else WORLD_COLLISION_PADDING
            prim.dimensions = [
                float(v) + 2.0 * padding for v in obj["size"]
            ]
            pose = self._transform_xyz_quat_to_pose(
                obj["position"], [0.0, 0.0, 0.0, 1.0], world_to_base)
            co.primitives.append(prim)
            co.primitive_poses.append(pose)
            co.operation = CollisionObject.ADD
            ps.world.collision_objects.append(co)

        # --- ZED 인식 잔여 장애물 voxel ---
        for obstacle in self.dynamic_obstacles:
            co = CollisionObject()
            co.id = obstacle["id"]
            co.header.frame_id = BASE_FRAME
            prim = SolidPrimitive()
            prim.type = SolidPrimitive.BOX
            prim.dimensions = [
                float(v) + 2.0 * WORLD_COLLISION_PADDING
                for v in obstacle["size"]
            ]
            pose = Pose()
            pose.position.x = float(obstacle["position"][0])
            pose.position.y = float(obstacle["position"][1])
            pose.position.z = float(obstacle["position"][2])
            pose.orientation.x = float(obstacle["orientation"][0])
            pose.orientation.y = float(obstacle["orientation"][1])
            pose.orientation.z = float(obstacle["orientation"][2])
            pose.orientation.w = float(obstacle["orientation"][3])
            co.primitives.append(prim)
            co.primitive_poses.append(pose)
            co.operation = CollisionObject.ADD
            ps.world.collision_objects.append(co)

        # --- EoAT (tcp -> AFT200 -> EOAT no-camera mesh -> D405, tcp 에 attached) ---
        eoat_aco = AttachedCollisionObject()
        eoat_aco.link_name = EE_LINK  # "tcp"
        eoat_aco.object.id = "eoat"
        eoat_aco.object.header.frame_id = EE_LINK

        # mesh[0] — AFT200 F/T sensor. Isaac Sim 과 같은 실제 collision STL.
        try:
            aft_mesh = _load_aft200_mesh()
            aft_mesh_pose = Pose()
            aft_mesh_pose.orientation.w = 1.0
            eoat_aco.object.meshes.append(aft_mesh)
            eoat_aco.object.mesh_poses.append(aft_mesh_pose)
        except Exception as e:
            self.get_logger().warn(
                f"AFT200 mesh 로드 실패 -> bbox primitive fallback 사용: {e}")
            aft_prim = SolidPrimitive()
            aft_prim.type = SolidPrimitive.BOX
            aft_prim.dimensions = list(AFT200_SIZE)
            aft_pose = Pose()
            aft_pose.position.x = AFT200_CENTER[0]
            aft_pose.position.y = AFT200_CENTER[1]
            aft_pose.position.z = AFT200_CENTER[2]
            aft_pose.orientation.w = 1.0
            eoat_aco.object.primitives.append(aft_prim)
            eoat_aco.object.primitive_poses.append(aft_pose)

        # mesh[1] — RR-00A_B EOAT no-camera collision mesh.
        # CAD +Z 를 TCP -Y 로 회전하고, AFT200 뒤에 붙는 위치까지 변환해 둔 mesh.
        try:
            eoat_mesh = _load_eoat_no_camera_mesh()
            eoat_mesh_pose = Pose()
            eoat_mesh_pose.orientation.w = 1.0
            eoat_aco.object.meshes.append(eoat_mesh)
            eoat_aco.object.mesh_poses.append(eoat_mesh_pose)
        except Exception as e:
            self.get_logger().warn(
                f"EOAT no-camera mesh 로드 실패 -> roller primitive fallback 사용: {e}")

            support_prim = SolidPrimitive()
            support_prim.type = SolidPrimitive.CYLINDER
            support_prim.dimensions = [ROLLER_FORWARD_REACH, ROLLER_SUPPORT_RADIUS]
            support_pose = Pose()
            sx, sy, sz = _axis_offset(
                TOOL_AXIS, AFT200_LENGTH + ROLLER_FORWARD_REACH / 2.0)
            support_pose.position.x = sx
            support_pose.position.y = sy
            support_pose.position.z = sz
            sqx, sqy, sqz, sqw = _cylinder_axis_quat(TOOL_AXIS)
            support_pose.orientation.x = sqx
            support_pose.orientation.y = sqy
            support_pose.orientation.z = sqz
            support_pose.orientation.w = sqw
            eoat_aco.object.primitives.append(support_prim)
            eoat_aco.object.primitive_poses.append(support_pose)

            roller_prim = SolidPrimitive()
            roller_prim.type = SolidPrimitive.CYLINDER
            roller_prim.dimensions = [ROLLER_LENGTH, ROLLER_RADIUS]
            roller_pose = Pose()
            cx, cy, cz = _axis_offset(TOOL_AXIS, EOAT_TIP_OFFSET)
            roller_pose.position.x = cx
            roller_pose.position.y = cy
            roller_pose.position.z = cz
            lqx, lqy, lqz, lqw = _cylinder_axis_quat(ROLLER_LONG_AXIS)
            roller_pose.orientation.x = lqx
            roller_pose.orientation.y = lqy
            roller_pose.orientation.z = lqz
            roller_pose.orientation.w = lqw
            eoat_aco.object.primitives.append(roller_prim)
            eoat_aco.object.primitive_poses.append(roller_pose)

        # primitive[1] — Intel RealSense D405. It is physically attached to the
        # EOAT, so MoveIt must treat it as robot geometry.
        d405_prim = SolidPrimitive()
        d405_prim.type = SolidPrimitive.BOX
        d405_prim.dimensions = list(D405_SIZE)
        d405_pose = Pose()
        d405_pose.position.x = D405_COLLISION_CENTER[0]
        d405_pose.position.y = D405_COLLISION_CENTER[1]
        d405_pose.position.z = D405_COLLISION_CENTER[2]
        d405_pose.orientation.w = 1.0
        eoat_aco.object.primitives.append(d405_prim)
        eoat_aco.object.primitive_poses.append(d405_pose)

        eoat_aco.object.operation = CollisionObject.ADD
        # 장착 플랜지 쪽 접촉만 허용. link5 는 손목 충돌을 잡기 위해 제외.
        eoat_aco.touch_links = list(EOAT_TOUCH_LINKS)
        ps.robot_state.attached_collision_objects.append(eoat_aco)
        ps.robot_state.is_diff = True

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
                "PlanningScene: 물체 + EoAT(AFT200-mesh+EOAT-no-camera-mesh+D405) publish + apply 시도")
            self.scene_initialized = True

    def _dynamic_target_collision_object(self, obj):
        """Perception 기반 활성 target collision slab 생성.

        objects.yaml 의 target box 는 nominal fallback 이고, 실제 작업에서는
        ZED 가 발행한 work_area_plane/work_area_corners 를 우선 사용한다.
        local +Z face 가 작업 표면이 되도록 center 를 normal 반대쪽으로 둔다.
        """
        if self.dynamic_surface_point is None or self.dynamic_surface_normal is None:
            return None

        normal = np.asarray(self.dynamic_surface_normal, dtype=float)
        normal /= np.linalg.norm(normal) + 1e-12
        size = np.asarray(obj.get("size", [1.0, 0.02, 1.0]), dtype=float)
        thickness = float(np.min(size)) if size.size else TARGET_COLLISION_MIN_THICKNESS
        thickness = float(np.clip(
            thickness,
            TARGET_COLLISION_MIN_THICKNESS,
            TARGET_COLLISION_MAX_THICKNESS,
        ))
        matched_extent = self._matched_perception_plane_extent(
            normal, self.dynamic_surface_point)

        area_basis = self._dynamic_work_area_basis()
        if area_basis is not None:
            surface_center, u_axis, v_axis, half_u, half_v = area_basis
            surface_center = self._project_point_to_active_surface(
                surface_center, normal)
            width = 2.0 * (half_u + TARGET_COLLISION_MARGIN)
            height = 2.0 * (half_v + TARGET_COLLISION_MARGIN)
            source = "work_area_corners"
            if matched_extent is not None:
                width = max(width, matched_extent[0])
                height = max(height, matched_extent[1])
                source += f"+{matched_extent[2]}_plane_extent"
        else:
            surface_center = np.asarray(self.dynamic_surface_point, dtype=float)
            u_axis, v_axis = self._plane_basis_from_normal(normal)
            if matched_extent is not None:
                width, height = matched_extent[:2]
                source = f"{matched_extent[2]}_plane_extent"
            else:
                tangent_sizes = sorted([float(v) for v in size], reverse=True)
                width = tangent_sizes[0] if tangent_sizes else WORK_AREA_W
                height = tangent_sizes[1] if len(tangent_sizes) > 1 else WORK_AREA_H
                source = "target_surface"

        u_axis = np.asarray(u_axis, dtype=float)
        u_axis /= np.linalg.norm(u_axis) + 1e-12
        v_axis = np.asarray(v_axis, dtype=float)
        v_axis /= np.linalg.norm(v_axis) + 1e-12
        if float(np.dot(np.cross(u_axis, v_axis), normal)) < 0.0:
            v_axis = -v_axis

        center = np.asarray(surface_center, dtype=float) - normal * (thickness / 2.0)
        q = quat_from_matrix(np.column_stack([u_axis, v_axis, normal]))

        co = CollisionObject()
        co.id = obj["name"]
        co.header.frame_id = BASE_FRAME
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(width), float(height), float(thickness)]
        pose = Pose()
        pose.position.x = float(center[0])
        pose.position.y = float(center[1])
        pose.position.z = float(center[2])
        pose.orientation.x = float(q[0])
        pose.orientation.y = float(q[1])
        pose.orientation.z = float(q[2])
        pose.orientation.w = float(q[3])
        co.primitives.append(prim)
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD

        self.get_logger().info(
            f"[SCENE] active target collision 동적 갱신({source}): "
            f"center=({center[0]:+.3f},{center[1]:+.3f},{center[2]:+.3f}) "
            f"size=({width:.3f},{height:.3f},{thickness:.3f}) "
            f"normal=({normal[0]:+.2f},{normal[1]:+.2f},{normal[2]:+.2f})",
            throttle_duration_sec=2.0,
        )
        return co

    def _matched_perception_plane_extent(self, normal, surface_point):
        """선택된 surface 와 같은 평면으로 보이는 scanner plane 크기 반환."""
        if not self.perception_planes:
            return None
        n = np.asarray(normal, dtype=float)
        n /= np.linalg.norm(n) + 1e-12
        p = np.asarray(surface_point, dtype=float)

        best = None
        for plane in self.perception_planes:
            pn = np.asarray(plane["normal"], dtype=float)
            pn /= np.linalg.norm(pn) + 1e-12
            align = abs(float(np.dot(pn, n)))
            if align < 0.90:
                continue
            plane_dist = abs(float(np.dot(p - plane["point"], pn)))
            if plane_dist > 0.10:
                continue

            label = self._plane_label_for_index(plane["index"])
            if label is None:
                continue
            try:
                dims = [
                    float(v)
                    for v in label.get("size", [])[:2]
                    if float(v) > 0.02
                ]
            except (TypeError, ValueError):
                continue
            if len(dims) < 2:
                continue
            dims = sorted(dims, reverse=True)
            kind = str(label.get("type", "plane"))
            # 같은 무한 평면 위에서는 centroid 가 멀 수 있으므로 plane distance
            # 와 normal 정렬을 우선한다. wall label 은 동률일 때만 약간 선호.
            score = plane_dist + (1.0 - align) * 0.25
            if kind == "wall":
                score -= 0.01
            if best is None or score < best[0]:
                best = (score, dims[0], dims[1], kind)

        if best is None:
            return None
        return best[1], best[2], best[3]

    def _plane_label_for_index(self, index):
        if index < 0 or index >= len(self.perception_plane_labels):
            return None
        label = self.perception_plane_labels[index]
        return label if isinstance(label, dict) else None

    @staticmethod
    def _plane_basis_from_normal(normal):
        n = np.asarray(normal, dtype=float)
        n /= np.linalg.norm(n) + 1e-12
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(ref, n))) > 0.95:
            ref = np.array([1.0, 0.0, 0.0], dtype=float)
        u_axis = np.cross(n, ref)
        u_axis /= np.linalg.norm(u_axis) + 1e-12
        v_axis = np.cross(n, u_axis)
        v_axis /= np.linalg.norm(v_axis) + 1e-12
        return u_axis, v_axis

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
                "[OK] PlanningScene apply 성공 (MoveIt 에 wall+EoAT(AFT200-mesh+EOAT-no-camera-mesh+D405) 등록됨)")
        elif not resp.success:
            self.get_logger().warn("ApplyPlanningScene 실패 (success=False)")

    # ---- joint limit / trajectory safety ------------------------------------
    def _joint_state_within_limits(self, joint_state, label):
        if joint_state is None or not joint_state.name:
            self.get_logger().error(f"[JOINT LIMIT] {label}: joint_state 없음")
            return False
        ok = True
        positions = list(joint_state.position)
        for name, pos in zip(joint_state.name, positions):
            if name not in JOINT_LIMITS:
                continue
            lower, upper = JOINT_LIMITS[name]
            value = float(pos)
            if value < lower - JOINT_LIMIT_MARGIN or value > upper + JOINT_LIMIT_MARGIN:
                self.get_logger().error(
                    f"[JOINT LIMIT] {label}: {name}={value:+.4f} rad "
                    f"outside [{lower:+.2f}, {upper:+.2f}]")
                ok = False
        return ok

    def _trajectory_within_joint_limits(self, jt, label):
        if not jt.joint_names:
            self.get_logger().error(f"[JOINT LIMIT] {label}: joint_names 없음")
            return False

        violations = []
        for point_idx, point in enumerate(jt.points):
            for joint_idx, name in enumerate(jt.joint_names):
                if name not in JOINT_LIMITS or joint_idx >= len(point.positions):
                    continue
                lower, upper = JOINT_LIMITS[name]
                value = float(point.positions[joint_idx])
                if value < lower - JOINT_LIMIT_MARGIN or value > upper + JOINT_LIMIT_MARGIN:
                    violations.append((point_idx, name, value, lower, upper))
                    if len(violations) >= 5:
                        break
            if len(violations) >= 5:
                break

        if not violations:
            return True

        for point_idx, name, value, lower, upper in violations:
            self.get_logger().error(
                f"[JOINT LIMIT] {label}: point#{point_idx} "
                f"{name}={value:+.4f} rad outside [{lower:+.2f}, {upper:+.2f}]")
        self.get_logger().error(
            f"[JOINT LIMIT] {label}: trajectory rejected before robot command. "
            "MoveIt/Cartesian result would exceed real RB10 joint limits.")
        return False

    # ---- 궤적 실행 (FollowJointTrajectory action) ---------------------------
    def execute_trajectory_direct(self, traj, on_complete=None,
                                  force_guard=False, label="trajectory"):
        """RB10 driver 의 FollowJointTrajectory action 으로 trajectory 전송.
        on_complete: action 성공 후 호출할 callback (다음 stage 트리거용).
        함수명은 호출처 호환을 위해 유지."""
        if EXECUTION_BACKEND == "joint_command":
            return self._execute_trajectory_joint_command(
                traj, on_complete=on_complete,
                force_guard=force_guard, label=label)
        if force_guard:
            self.get_logger().warn(
                f"[FT GUARD] {label}: FollowJointTrajectory backend 에서는 "
                "실행 중 force interlock 이 즉시 cancel 되지 않을 수 있음")
        return self._execute_trajectory_follow_joint(
            traj, on_complete=on_complete)

    def _point_time_sec(self, point):
        return (
            float(point.time_from_start.sec) +
            float(point.time_from_start.nanosec) * 1e-9
        )

    def _cancel_joint_command_timer(self):
        if self._joint_command_timer is not None:
            self._joint_command_timer.cancel()
            self.destroy_timer(self._joint_command_timer)
            self._joint_command_timer = None

    def _ft_guard_triggered(self, label):
        if self.ft_normal_force_n is None:
            return False
        if time.monotonic() - self.ft_status_time > FT_FORCE_STALE_SEC:
            return False
        if not self.ft_bias_ready:
            self.get_logger().warn(
                f"[FT GUARD] {label}: FT bias not ready; guard inactive",
                throttle_duration_sec=2.0)
            return False
        if self.ft_normal_force_n >= self.ft_abort_force_n:
            self.get_logger().error(
                f"[FT GUARD] {label}: normal force "
                f"{self.ft_normal_force_n:.1f}N >= "
                f"{self.ft_abort_force_n:.1f}N -> trajectory stop")
            return True
        return False

    def _joint_command_from_positions(self, joint_names, positions):
        """Isaac Sim /joint_command JointState 생성.

        trajectory joint_names 는 MoveIt 순서이고, Isaac JointGraph 는 현재
        /joint_states 순서도 받을 수 있으므로 이름 기준으로 재정렬한다.
        """
        if len(joint_names) != len(positions):
            raise ValueError(
                f"trajectory joint_names({len(joint_names)})와 "
                f"positions({len(positions)}) 길이가 다름")

        cmd_by_name = dict(zip(joint_names, positions))
        if self.current_joint_state is not None and self.current_joint_state.name:
            names = list(self.current_joint_state.name)
            current_by_name = dict(zip(
                self.current_joint_state.name,
                self.current_joint_state.position,
            ))
            positions = [
                float(cmd_by_name.get(name, current_by_name.get(name, 0.0)))
                for name in names
            ]
        else:
            names = list(joint_names)
            positions = [float(v) for v in positions]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = positions
        return msg

    def _joint_command_from_point(self, joint_names, point):
        """Trajectory point 를 Isaac Sim /joint_command JointState 로 변환."""
        return self._joint_command_from_positions(joint_names, point.positions)

    def _sample_trajectory_positions(self, points, times, idx, elapsed):
        """시간 elapsed 에서 trajectory position 을 보간한다.

        MoveIt 이 velocity 를 채워주면 cubic Hermite 보간을 써서 waypoint
        경계의 속도 불연속을 줄이고, velocity 가 없으면 선형 보간으로 fallback.
        """
        if elapsed <= times[0]:
            return [float(v) for v in points[0].positions]
        if elapsed >= times[-1]:
            return [float(v) for v in points[-1].positions]

        while idx < len(points) - 2 and times[idx + 1] <= elapsed:
            idx += 1

        p0 = points[idx]
        p1 = points[idx + 1]
        t0 = times[idx]
        t1 = times[idx + 1]
        dt = t1 - t0
        if dt <= 1e-9:
            return [float(v) for v in p1.positions], idx

        a = max(0.0, min(1.0, (elapsed - t0) / dt))
        p0_pos = [float(v) for v in p0.positions]
        p1_pos = [float(v) for v in p1.positions]

        has_vel = (
            len(p0.velocities) == len(p0_pos) and
            len(p1.velocities) == len(p1_pos)
        )
        if has_vel:
            a2 = a * a
            a3 = a2 * a
            h00 = 2.0 * a3 - 3.0 * a2 + 1.0
            h10 = a3 - 2.0 * a2 + a
            h01 = -2.0 * a3 + 3.0 * a2
            h11 = a3 - a2
            positions = [
                h00 * p0_pos[i] +
                h10 * dt * float(p0.velocities[i]) +
                h01 * p1_pos[i] +
                h11 * dt * float(p1.velocities[i])
                for i in range(len(p0_pos))
            ]
        else:
            positions = [
                p0_pos[i] + a * (p1_pos[i] - p0_pos[i])
                for i in range(len(p0_pos))
            ]
        return positions, idx

    def _execute_trajectory_joint_command(self, traj, on_complete=None,
                                          force_guard=False,
                                          label="trajectory"):
        """Isaac Sim JointGraph 가 구독하는 /joint_command 로 trajectory 재생."""
        jt = traj.joint_trajectory
        if not jt.points:
            self.get_logger().warn("빈 궤적")
            if on_complete is None:
                self.executing = False
            return False
        if not jt.joint_names:
            self.get_logger().error("trajectory joint_names 비어 있음")
            self.executing = False
            return False
        if not self._trajectory_within_joint_limits(jt, "joint_command"):
            self.executing = False
            return False

        self._cancel_joint_command_timer()

        start_time = time.monotonic()
        point_times = [self._point_time_sec(p) for p in jt.points]
        end_time = max(point_times)
        state = {"idx": 0, "done": False}

        self.get_logger().info(
            f"/joint_command 재생: {len(jt.points)} 포인트, "
            f"duration={end_time:.2f}s, "
            f"publish_rate={1.0 / JOINT_COMMAND_TIMER_PERIOD:.0f}Hz, "
            f"interpolation=on, ft_guard={'on' if force_guard else 'off'}")

        def _tick():
            if state["done"]:
                return

            elapsed = time.monotonic() - start_time
            if force_guard and self._ft_guard_triggered(label):
                state["done"] = True
                self._cancel_joint_command_timer()
                self.executing = False
                return
            try:
                sampled = self._sample_trajectory_positions(
                    jt.points, point_times, state["idx"], elapsed)
                if isinstance(sampled, tuple):
                    positions, state["idx"] = sampled
                else:
                    positions = sampled
                self.joint_cmd_pub.publish(
                    self._joint_command_from_positions(jt.joint_names, positions))
            except Exception as e:
                self.get_logger().error(f"/joint_command publish 실패: {e}")
                state["done"] = True
                self._cancel_joint_command_timer()
                self.executing = False
                return

            if elapsed >= end_time:
                state["done"] = True
                try:
                    self.joint_cmd_pub.publish(
                        self._joint_command_from_point(
                            jt.joint_names, jt.points[-1]))
                except Exception:
                    pass
                self._cancel_joint_command_timer()
                self.get_logger().info(
                    ">>> 궤적 실행 완료 (/joint_command playback)")
                if on_complete is not None:
                    on_complete()
                else:
                    self.executing = False

        self._joint_command_timer = self.create_timer(
            JOINT_COMMAND_TIMER_PERIOD, _tick)
        _tick()
        return True

    def _execute_trajectory_follow_joint(self, traj, on_complete=None):
        """FollowJointTrajectory action 으로 trajectory 전송."""
        if not traj.joint_trajectory.points:
            self.get_logger().warn("빈 궤적")
            if on_complete is None:
                self.executing = False
            return False
        if not self._trajectory_within_joint_limits(
                traj.joint_trajectory, "follow_joint_trajectory"):
            self.executing = False
            return False

        if not self.traj_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("FollowJointTrajectory action server 없음")
            self.executing = False
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj.joint_trajectory  # trajectory_msgs/JointTrajectory 그대로
        # path/goal tolerances 는 비워둠 (controller 디폴트 사용).
        # 필요시 차후에 GoalTolerance 추가.

        self.get_logger().info(
            f"FollowJointTrajectory 전송: {len(traj.joint_trajectory.points)} 포인트")

        send_future = self.traj_action_client.send_goal_async(goal)

        def _goal_response(fut):
            try:
                handle = fut.result()
            except Exception as e:
                self.get_logger().error(f"send_goal 실패: {e}")
                self.executing = False
                return
            if not handle.accepted:
                self.get_logger().error("Trajectory goal rejected")
                self.executing = False
                return
            result_future = handle.get_result_async()
            result_future.add_done_callback(_result_done)

        def _result_done(fut):
            try:
                res = fut.result().result
            except Exception as e:
                self.get_logger().error(f"trajectory result 실패: {e}")
                self.executing = False
                return
            # FollowJointTrajectory.Result.error_code: 0 = SUCCESSFUL
            if res.error_code != 0:
                self.get_logger().error(
                    f"Trajectory 실행 에러 code={res.error_code}: {res.error_string}")
                self.executing = False
                return
            self.get_logger().info(">>> 궤적 실행 완료 (action SUCCESS)")
            if on_complete is not None:
                on_complete()
            else:
                self.executing = False

        send_future.add_done_callback(_goal_response)
        return True

    # ---- helpers for 4-stage approach ---------------------------------------
    def _active_surface_plane(self, target=None):
        if self.dynamic_surface_point is not None and self.dynamic_surface_normal is not None:
            return self.dynamic_surface_point, self.dynamic_surface_normal
        if target is None:
            target = get_target(self.cfg, self.active_target_name)
        return get_surface_plane(target)

    def _active_ee_quat(self, target):
        if self._stage3_tcp_wps:
            q = self._stage3_tcp_wps[0].orientation
            return np.array([q.x, q.y, q.z, q.w], dtype=float)
        return ee_quat_for_target(target)

    def _offset_along_normal(self, pose, distance):
        """pose 를 active target 의 표면 normal 방향으로 distance 만큼 후퇴."""
        _, n = self._active_surface_plane()
        out = copy.deepcopy(pose)
        out.position.x += distance * float(n[0])
        out.position.y += distance * float(n[1])
        out.position.z += distance * float(n[2])
        return out

    def _make_pose_constraints(self, pose, link_name=EE_LINK, frame=BASE_FRAME):
        """Pose 를 MoveGroup goal 의 Constraints 로 변환."""
        c = Constraints()

        pc = PositionConstraint()
        pc.header.frame_id = frame
        pc.link_name = link_name
        pc.target_point_offset.x = 0.0
        pc.target_point_offset.y = 0.0
        pc.target_point_offset.z = 0.0
        sp = SolidPrimitive()
        sp.type = SolidPrimitive.SPHERE
        sp.dimensions = [0.001]  # 1mm tolerance (tight)
        pc.constraint_region.primitives.append(sp)
        region_pose = Pose()
        region_pose.position = pose.position
        region_pose.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(region_pose)
        pc.weight = 1.0
        c.position_constraints.append(pc)

        oc = OrientationConstraint()
        oc.header.frame_id = frame
        oc.link_name = link_name
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = 0.02
        oc.absolute_y_axis_tolerance = 0.02
        oc.absolute_z_axis_tolerance = 0.02
        oc.weight = 1.0
        c.orientation_constraints.append(oc)

        return c

    def _make_joint_goal_constraints(
            self, joint_state, tolerance=0.02, joint_names=None):
        """IK 결과 joint_state 를 MoveGroup joint goal constraints 로 변환."""
        c = Constraints()
        goal_map = dict(zip(joint_state.name, joint_state.position))
        current_names = set(self.current_joint_state.name) \
            if self.current_joint_state is not None else set(goal_map.keys())
        names = joint_names or joint_state.name
        for name in names:
            if name not in goal_map or name not in current_names:
                continue
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(goal_map[name])
            jc.tolerance_above = float(tolerance)
            jc.tolerance_below = float(tolerance)
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        return c

    def _log_stage1_joint_delta(self, goal_joint_state):
        if self.current_joint_state is None:
            return
        current = dict(zip(
            self.current_joint_state.name,
            self.current_joint_state.position,
        ))
        goal = dict(zip(goal_joint_state.name, goal_joint_state.position))
        deltas = []
        for name in READY_POSE_JOINTS.keys():
            if name in current and name in goal:
                d = abs(float(goal[name]) - float(current[name]))
                deltas.append((name, d))
        if not deltas:
            return
        max_name, max_delta = max(deltas, key=lambda item: item[1])
        self.get_logger().info(
            f"[STAGE 1 IK] nearest joint goal: max_delta="
            f"{math.degrees(max_delta):.1f}deg ({max_name})")
        if max_delta > STAGE1_LARGE_JOINT_DELTA_WARN_RAD:
            self.get_logger().warn(
                f"[STAGE 1 IK] 큰 관절 이동 감지: {max_name} "
                f"{math.degrees(max_delta):.1f}deg. "
                "MoveIt 이 collision-free plan 은 찾지만 최단/최소 관절 이동을 "
                "수학적으로 보장하지는 않음.")

    def _compute_snapped_tcp_waypoints(self):
        """current_waypoints 를 densify + tcp 변환한 결과 반환.
        새 perception 흐름: sketch_to_waypoints_node 가 이미 wall_plane 좌표를
        roller center contact-clearance plane 으로 변환해서 보냄.
        따라서 yaml 기반 snap 불필요. waypoint 그대로 사용.
        target/n 은 caller 호환을 위해 yaml 에서 계속 반환 (offset_along_normal 에서 사용).
        Returns: (snapped_tip_wps, tcp_wps, target, n)
        """
        target = get_target(self.cfg, self.active_target_name)
        _sp, n = self._active_surface_plane(target)
        snapped = [copy.deepcopy(wp) for wp in self.current_waypoints]
        densified = self._densify_waypoints(snapped, spacing_m=0.005)
        tcp_wps = [self._brush_tip_to_tcp(wp) for wp in densified]
        self.get_logger().info(
            f"표면 스냅 SKIP (새 perception 흐름): N={len(snapped)} "
            f"첫점=({snapped[0].position.x:.3f},{snapped[0].position.y:.3f},"
            f"{snapped[0].position.z:.3f})")
        return densified, tcp_wps, target, n

    def _sync_active_target_plane_from_waypoints(self, waypoints):
        """Perception 기반 waypoint plane 에 맞춰 active target collision 위치 보정.

        objects.yaml 은 실험실 기준 nominal wall 이고, 실제 wall plane 은 ZED 가
        매번 인식한다. sketch_to_waypoints 는 roller center 를 발행하므로,
        mean(waypoints) 에서 roller radius+clearance 를 빼 active target 의
        collision surface 를 normal 축 방향으로 갱신한다.
        """
        if not waypoints:
            return
        try:
            target = get_target(self.cfg, self.active_target_name)
        except Exception as e:
            self.get_logger().warn(f"active target lookup 실패: {e}")
            return

        if self.dynamic_surface_point is not None:
            return
        expected = ROLLER_RADIUS + CONTACT_CLEARANCE
        pts = np.array([
            [p.position.x, p.position.y, p.position.z] for p in waypoints
        ], dtype=float)
        normal = self._infer_surface_normal_from_waypoints(waypoints, target)
        surface_point = np.mean(pts, axis=0) - normal * expected

        self.dynamic_surface_point = surface_point
        self.dynamic_surface_normal = normal

        axis = int(np.argmax(np.abs(normal)))
        half_axis = float(target["size"][axis]) / 2.0
        old_center_axis = float(target["position"][axis])
        if abs(float(normal[axis])) > 0.5:
            new_center_axis = (
                float(surface_point[axis]) - float(normal[axis]) * half_axis
            )
            target["position"][axis] = new_center_axis
        else:
            new_center_axis = old_center_axis

        self.scene_confirmed = False
        self.scene_initialized = False
        self.get_logger().warn(
            f"[SCENE SYNC] {self.active_target_name} plane 을 waypoint 로 복구: "
            f"point=({surface_point[0]:+.3f},{surface_point[1]:+.3f},"
            f"{surface_point[2]:+.3f}) normal=({normal[0]:+.2f},"
            f"{normal[1]:+.2f},{normal[2]:+.2f}), "
            f"axis={['x','y','z'][axis]}, center "
            f"{old_center_axis:+.3f}->{new_center_axis:+.3f} "
            f"(work_area_plane TF miss fallback)")

    def _infer_surface_normal_from_waypoints(self, waypoints, target):
        """sketch_to_waypoints 가 넣은 EE orientation 으로 free-space normal 복구.

        sketch_to_waypoints convention:
          local TOOL_AXIS = forward = -surface_normal
        따라서 surface normal 은 waypoint orientation 의 -TOOL_AXIS 방향이다.
        orientation 이 없거나 이상하면 objects.yaml 의 nominal normal 로 fallback.
        """
        normals = []
        for wp in waypoints:
            try:
                tool_axis = self._local_axis_in_world(wp.orientation, TOOL_AXIS)
            except Exception:
                continue
            n = -np.asarray(tool_axis, dtype=float)
            norm = float(np.linalg.norm(n))
            if norm > 1e-6:
                normals.append(n / norm)
        if normals:
            normal = np.mean(normals, axis=0)
            norm = float(np.linalg.norm(normal))
            if norm > 1e-6:
                return normal / norm
        _plane_point, normal = get_surface_plane(target)
        normal = np.asarray(normal, dtype=float)
        return normal / (np.linalg.norm(normal) + 1e-12)

    def _validate_contact_waypoints(self, waypoints, target):
        """작업영역/벽 clearance 검증.

        waypoints 는 wall surface point 가 아니라 roller center 이다. 모든 점은
        wall surface 에서 ROLLER_RADIUS + CONTACT_CLEARANCE 만큼 free-space 방향에
        있어야 하며, yellow work area 에 해당하는 0.6m x 0.6m 영역 안이어야 한다.
        """
        if not waypoints:
            self.get_logger().error("waypoint 없음")
            return False
        plane_point, normal = self._active_surface_plane(target)
        normal = np.asarray(normal, dtype=float)
        expected = ROLLER_RADIUS + CONTACT_CLEARANCE

        axis = int(np.argmax(np.abs(normal)))
        lateral_axes = [i for i in range(3) if i != axis]
        area_basis = self._dynamic_work_area_basis()
        half_limits = {
            lateral_axes[0]: WORK_AREA_W / 2.0 + WORK_AREA_MARGIN,
            lateral_axes[1]: WORK_AREA_H / 2.0 + WORK_AREA_MARGIN,
        }
        names = ["x", "y", "z"]
        bad = []
        for i, wp in enumerate(waypoints):
            p = np.array([wp.position.x, wp.position.y, wp.position.z])
            clearance = float(np.dot(p - plane_point, normal))
            if abs(clearance - expected) > CONTACT_PLANE_TOL:
                bad.append(
                    f"#{i}: clearance {clearance:.3f}m "
                    f"(expected {expected:.3f}±{CONTACT_PLANE_TOL:.3f})")
                continue
            if area_basis is not None:
                center, u_axis, v_axis, half_u, half_v = area_basis
                du = float(np.dot(p - center, u_axis))
                dv = float(np.dot(p - center, v_axis))
                if abs(du) > half_u or abs(dv) > half_v:
                    bad.append(
                        f"#{i}: outside dynamic work area "
                        f"(u={du:.3f}/{half_u:.3f}, v={dv:.3f}/{half_v:.3f})")
            else:
                for ax in lateral_axes:
                    if abs(float(p[ax] - plane_point[ax])) > half_limits[ax]:
                        bad.append(
                            f"#{i}: {names[ax]}={p[ax]:.3f} outside work area")
                        break
        if bad:
            for msg in bad[:5]:
                self.get_logger().error("[WAYPOINT SAFETY] " + msg)
            if len(bad) > 5:
                self.get_logger().error(
                    f"[WAYPOINT SAFETY] ... and {len(bad)-5} more")
            return False
        area_desc = (
            "dynamic work area"
            if area_basis is not None
            else f"work area={WORK_AREA_W:.2f}x{WORK_AREA_H:.2f}m"
        )
        self.get_logger().info(
            f"[WAYPOINT SAFETY] {len(waypoints)}점 검증 OK: "
            f"roller center clearance={expected*1000:.1f}mm, "
            f"{area_desc}")
        return True

    def _dynamic_work_area_basis(self):
        corners = self.dynamic_work_area_corners
        if corners is None or len(corners) < 4:
            return None
        tl, tr, br, bl = np.asarray(corners[:4], dtype=float)
        center = (tl + tr + br + bl) / 4.0
        u_vec = ((tr - tl) + (br - bl)) / 2.0
        v_vec = ((bl - tl) + (br - tr)) / 2.0
        width = float(np.linalg.norm(u_vec))
        height = float(np.linalg.norm(v_vec))
        if width < 1e-6 or height < 1e-6:
            return None
        return (
            center,
            u_vec / width,
            v_vec / height,
            width / 2.0 + WORK_AREA_MARGIN,
            height / 2.0 + WORK_AREA_MARGIN,
        )

    def _project_point_to_active_surface(self, point, normal=None):
        """Keep ZED-derived work-area lateral center, but use active/D405 depth."""
        p = np.asarray(point, dtype=float)
        if self.dynamic_surface_point is None:
            return p
        n = (
            np.asarray(normal, dtype=float)
            if normal is not None
            else np.asarray(self.dynamic_surface_normal, dtype=float)
        )
        n /= np.linalg.norm(n) + 1e-12
        plane_p = np.asarray(self.dynamic_surface_point, dtype=float)
        return p - float(np.dot(p - plane_p, n)) * n

    def _check_normal_motion(self, start_pose, end_pose, normal, expected_align,
                             label):
        start = np.array([
            start_pose.position.x,
            start_pose.position.y,
            start_pose.position.z,
        ], dtype=float)
        end = np.array([
            end_pose.position.x,
            end_pose.position.y,
            end_pose.position.z,
        ], dtype=float)
        delta = end - start
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            self.get_logger().error(f"[{label}] 이동 거리 0 -> 중단")
            return False
        direction = delta / dist
        align = float(np.dot(direction, np.asarray(normal, dtype=float)))
        if align < expected_align:
            self.get_logger().error(
                f"[{label}] normal 방향 정렬 실패: align={align:+.3f}, "
                f"required>={expected_align:.3f}")
            return False
        self.get_logger().info(
            f"[{label}] normal motion OK: dist={dist*100:.1f}cm, "
            f"align={align:+.3f}")
        return True

    # ---- Stage 1: free-space approach via MoveGroup action -------------------
    def stage1_approach_free(self, on_complete=None):
        # on_complete: 실행 완료 후 호출할 콜백. None 이면 production 기본 (Stage 2 chain).
        self._stage1_on_complete = on_complete or self.stage2_approach_linear
        # PlanningScene 검증 대기 (attached torch + wall 등록 확인)
        import time
        wait_start = time.time()
        while not self.scene_confirmed and (time.time() - wait_start) < 5.0:
            self.get_logger().info("PlanningScene 검증 대기 중...")
            time.sleep(0.5)
        if not self.scene_confirmed:
            self.get_logger().warn(
                "PlanningScene 미검증 (5초 타임아웃). 충돌 회피 약할 수 있음.")
        else:
            self.get_logger().info("PlanningScene 검증 OK")

        if not self.move_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("MoveGroup action server 없음 (/move_action)")
            self.executing = False
            return
        if not self._joint_state_within_limits(
                self.current_joint_state, "STAGE 1 start"):
            self.get_logger().error(
                "STAGE 1 시작 joint_state 가 limit 밖 -> 실행 중단")
            self.executing = False
            return

        self._request_stage1_nearest_ik()

    def _request_stage1_nearest_ik(self):
        """현재 joint state 를 seed 로 쓰는 IK 를 먼저 풀어 Stage 1 wrist flip 을 줄인다."""
        if not self.ik_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                "/compute_ik 서비스 없음 -> Stage 1 pose goal fallback")
            self._send_stage1_plan(
                [self._make_pose_constraints(self._safety_tcp_pose)],
                "pose goal fallback",
            )
            return

        req = GetPositionIK.Request()
        req.ik_request.group_name = PLANNING_GROUP
        req.ik_request.robot_state.joint_state = self.current_joint_state
        req.ik_request.robot_state.is_diff = False
        req.ik_request.avoid_collisions = True
        req.ik_request.ik_link_name = EE_LINK
        req.ik_request.pose_stamped.header.frame_id = BASE_FRAME
        req.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        req.ik_request.pose_stamped.pose = self._safety_tcp_pose
        req.ik_request.timeout = Duration(seconds=STAGE1_IK_TIMEOUT_S).to_msg()

        future = self.ik_client.call_async(req)
        future.add_done_callback(self._stage1_ik_done)

    def _stage1_ik_done(self, future):
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().warn(
                f"STAGE 1 IK service 실패: {e} -> pose goal fallback")
            self._send_stage1_plan(
                [self._make_pose_constraints(self._safety_tcp_pose)],
                "pose goal fallback",
            )
            return

        if resp.error_code.val != 1:
            self.get_logger().warn(
                f"STAGE 1 IK 실패 error_code={resp.error_code.val} "
                "-> pose goal fallback")
            self._send_stage1_plan(
                [self._make_pose_constraints(self._safety_tcp_pose)],
                "pose goal fallback",
            )
            return

        self._log_stage1_joint_delta(resp.solution.joint_state)
        joint_goal = self._make_joint_goal_constraints(
            resp.solution.joint_state,
            tolerance=STAGE1_JOINT_GOAL_TOL,
            joint_names=list(READY_POSE_JOINTS.keys()),
        )
        if not joint_goal.joint_constraints:
            self.get_logger().warn(
                "STAGE 1 IK 결과에 사용 가능한 joint 없음 -> pose goal fallback")
            self._send_stage1_plan(
                [self._make_pose_constraints(self._safety_tcp_pose)],
                "pose goal fallback",
            )
            return

        self._send_stage1_plan([joint_goal], "nearest IK joint goal")

    def _send_stage1_plan(self, goal_constraints, label, planner_id=PLANNER_ID):
        goal = MoveGroup.Goal()
        goal.request.group_name = PLANNING_GROUP
        rs = RobotState()
        rs.joint_state = self.current_joint_state
        rs.is_diff = False
        goal.request.start_state = rs
        goal.request.goal_constraints = goal_constraints
        goal.request.planner_id = planner_id
        goal.request.allowed_planning_time = ALLOWED_PLANNING_TIME
        goal.request.num_planning_attempts = PLANNING_ATTEMPTS
        goal.request.max_velocity_scaling_factor = STAGE1_SPEED_SCALE
        goal.request.max_acceleration_scaling_factor = STAGE1_SPEED_SCALE

        goal.planning_options.plan_only = True
        goal.planning_options.planning_scene_diff.is_diff = True

        self._stage1_goal_constraints = goal_constraints
        self.get_logger().info(
            f"STAGE 1 planning request: {label}, "
            f"speed_scale={STAGE1_SPEED_SCALE:.2f}")
        future = self.move_action_client.send_goal_async(goal)
        future.add_done_callback(self._stage1_goal_response)

    def _retry_stage1_with_default_planner(self):
        """planner_id 를 비워서 MoveGroup 의 기본 planner 로 재시도."""
        constraints = self._stage1_goal_constraints or [
            self._make_pose_constraints(self._safety_tcp_pose)
        ]
        self._send_stage1_plan(
            constraints,
            "same goal with default planner",
            planner_id="",
        )

    def _stage1_goal_response(self, future):
        try:
            handle = future.result()
        except Exception as e:
            self.get_logger().error(f"STAGE 1 send_goal 실패: {e}")
            self.executing = False
            return
        if not handle.accepted:
            self.get_logger().error("STAGE 1 goal rejected")
            self.executing = False
            return
        self.get_logger().info("STAGE 1 goal accepted, planning...")
        handle.get_result_async().add_done_callback(self._stage1_result)

    def _stage1_result(self, future):
        try:
            result = future.result().result
        except Exception as e:
            self.get_logger().error(f"STAGE 1 result 실패: {e}")
            self.executing = False
            return
        if result.error_code.val != 1:  # MoveItErrorCodes.SUCCESS = 1
            self.get_logger().error(
                f"STAGE 1 planning 실패 error_code={result.error_code.val}")
            # planner_id mismatch 가능성 — 빈 planner_id 로 한 번 재시도
            if not getattr(self, "_stage1_retried", False):
                self._stage1_retried = True
                self.get_logger().warn(
                    f"planner_id='{PLANNER_ID}' 실패 — 서버 기본 planner 로 재시도")
                self._retry_stage1_with_default_planner()
                return
            self._stage1_retried = False
            self.executing = False
            return

        # 성공 시 retry flag 리셋
        self._stage1_retried = False

        traj = result.planned_trajectory
        n_points = len(traj.joint_trajectory.points)
        self.get_logger().info(f"STAGE 1 planning OK: {n_points} 포인트")

        # MoveGroup request 의 max_velocity_scaling_factor 가 이미 timing 에
        # 반영되어 있다. 여기서 다시 같은 scale 을 적용하면 0.1 x 0.1 이 되어
        # 접근 동작이 100초 이상 걸린다.
        traj = self._rescale_trajectory(traj, scale=1.0)
        on_complete = getattr(self, "_stage1_on_complete", None) \
            or self.stage2_approach_linear
        if not self.execute_trajectory_direct(traj, on_complete=on_complete):
            self.executing = False

    # ---- Stage 2: linear approach to first surface point (cartesian) --------
    def stage2_approach_linear(self):
        self.get_logger().info("=== STAGE 2: linear approach (cartesian) ===")
        if not self.cartesian_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/compute_cartesian_path 서비스 없음 (Stage 2)")
            self.executing = False
            return

        target = get_target(self.cfg, self.active_target_name)
        _, n = self._active_surface_plane(target)

        req = GetCartesianPath.Request()
        req.header.frame_id = BASE_FRAME
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = PLANNING_GROUP
        req.link_name = EE_LINK

        rs = RobotState()
        rs.joint_state = self.current_joint_state
        rs.is_diff = False
        req.start_state = rs

        # 목표: 첫 접촉-clearance 롤러 중심점의 tcp 좌표
        req.waypoints = [self._stage3_tcp_wps[0]]
        req.max_step = 0.005
        req.jump_threshold = 5.0
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = STAGE2_SPEED_SCALE
        req.max_acceleration_scaling_factor = STAGE2_SPEED_SCALE

        # Orientation constraint (Stage 3 와 동일)
        ee_q = self._active_ee_quat(target)
        brush_dir_world = -np.asarray(n, dtype=float)
        free_axis = int(np.argmax(np.abs(brush_dir_world)))
        tol = [0.2, 0.2, 0.2]
        tol[free_axis] = 3.14
        oc = OrientationConstraint()
        oc.header.frame_id = BASE_FRAME
        oc.link_name = EE_LINK
        oc.orientation.x = float(ee_q[0])
        oc.orientation.y = float(ee_q[1])
        oc.orientation.z = float(ee_q[2])
        oc.orientation.w = float(ee_q[3])
        oc.absolute_x_axis_tolerance = tol[0]
        oc.absolute_y_axis_tolerance = tol[1]
        oc.absolute_z_axis_tolerance = tol[2]
        oc.weight = 1.0
        req.path_constraints = Constraints()
        req.path_constraints.orientation_constraints.append(oc)

        # 디버그: stage 2 의 시작 → 끝 거리와 방향 검증
        end_pose = self._stage3_tcp_wps[0]
        # 현재 tcp 위치는 정확히 모르지만, 직전 stage 1 의 의도된 도착점 (safety_tcp_pose) 로 근사
        start_pose = self._safety_tcp_pose
        delta = np.array([
            end_pose.position.x - start_pose.position.x,
            end_pose.position.y - start_pose.position.y,
            end_pose.position.z - start_pose.position.z,
        ])
        dist = float(np.linalg.norm(delta))
        direction = delta / (dist + 1e-9)
        _, n_target = self._active_surface_plane(target)
        align = float(np.dot(direction, -np.asarray(n_target)))  # +1 이 완벽한 normal 진입
        self.get_logger().info(
            f"[STAGE 2 DEBUG] dist={dist*100:.1f}cm "
            f"direction=({direction[0]:+.2f},{direction[1]:+.2f},{direction[2]:+.2f}) "
            f"normal_align={align:+.3f} (1.0=perfect)")
        if align < MIN_APPROACH_NORMAL_ALIGN:
            self.get_logger().error(
                f"STAGE 2 접근 방향이 surface normal 과 맞지 않음 "
                f"({align:+.3f} < {MIN_APPROACH_NORMAL_ALIGN}) -> 중단")
            self.executing = False
            return

        future = self.cartesian_client.call_async(req)
        future.add_done_callback(self._stage2_done)

    def _stage2_done(self, future):
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(f"STAGE 2 서비스 실패: {e}")
            self.executing = False
            return
        fraction = resp.fraction
        self.get_logger().info(f"STAGE 2 cartesian: {fraction*100:.1f}%")
        if fraction < MIN_CARTESIAN_FRACTION:
            self.get_logger().error(
                f"STAGE 2 fraction {fraction*100:.1f}% < "
                f"{MIN_CARTESIAN_FRACTION*100:.1f}% -> 중단")
            self.executing = False
            return
        traj = self._rescale_trajectory(resp.solution, scale=1.0)
        if not self.execute_trajectory_direct(
                traj, on_complete=self.plan_cartesian,
                force_guard=True, label="STAGE 2 approach"):
            self.executing = False

    # ---- roller center/tool tip → tcp 오프셋 변환 -----------------------------
    @staticmethod
    def _local_axis_in_world(q, axis):
        """쿼터니언 q 기준 로컬 axis ('+x'/'-x'/...) 의 world 방향 단위벡터."""
        x, y, z, w = q.x, q.y, q.z, q.w
        if axis in ("+x", "-x"):
            v = np.array([1 - 2*(y*y + z*z), 2*(x*y + z*w), 2*(x*z - y*w)])
        elif axis in ("+y", "-y"):
            v = np.array([2*(x*y - z*w), 1 - 2*(x*x + z*z), 2*(y*z + x*w)])
        elif axis in ("+z", "-z"):
            v = np.array([2*(x*z + y*w), 2*(y*z - x*w), 1 - 2*(x*x + y*y)])
        else:
            raise ValueError(f"unknown axis: {axis}")
        if axis.startswith("-"):
            v = -v
        return v

    @staticmethod
    def _brush_tip_to_tcp(pose):
        """tool tip (= 롤러 회전축 중심) 기준 좌표를 tcp 기준으로 변환.
        AFT200+roller 는 tcp 의 로컬 TOOL_AXIS 방향으로 EOAT_TIP_OFFSET 만큼 뻗음."""
        axis_world = MoveItExecutor._local_axis_in_world(
            pose.orientation, TOOL_AXIS)
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        new_pos = pos - EOAT_TIP_OFFSET * axis_world
        new_pose = Pose()
        new_pose.position.x = float(new_pos[0])
        new_pose.position.y = float(new_pos[1])
        new_pose.position.z = float(new_pos[2])
        new_pose.orientation = copy.deepcopy(pose.orientation)
        return new_pose

    # ---- 3D 웨이포인트 스플라인 밀집화 ------------------------------------------
    @staticmethod
    def _densify_waypoints(waypoints, spacing_m=0.005):
        """3D poses 를 거리 기준 선형 보간으로 등간격 재샘플."""
        if len(waypoints) < 2:
            return waypoints
        positions = np.array([[p.position.x, p.position.y, p.position.z]
                              for p in waypoints])
        # 중복/매우 가까운 점 제거
        diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        keep = np.concatenate([[True], diffs > 1e-6])
        positions = positions[keep]
        if len(positions) < 2:
            return waypoints
        dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        t = np.concatenate([[0], np.cumsum(dists)])
        total = t[-1]
        if total < 1e-6:
            return waypoints
        n = max(len(waypoints), int(total / spacing_m))
        t_new = np.linspace(0, total, n)
        new_wps = []
        ref_ori = waypoints[0].orientation
        xyz = [np.interp(t_new, t, positions[:, i]) for i in range(3)]
        for idx in range(len(t_new)):
            p = Pose()
            p.position.x = float(xyz[0][idx])
            p.position.y = float(xyz[1][idx])
            p.position.z = float(xyz[2][idx])
            p.orientation = copy.deepcopy(ref_ori)
            new_wps.append(p)
        return new_wps

    # ---- Stage 3: Cartesian path 계획 + 실행 --------------------------------
    def plan_cartesian(self):
        if not self.cartesian_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/compute_cartesian_path 서비스 없음")
            return

        # 새 perception 흐름: sketch_to_waypoints_node 가 이미 wall_plane 좌표를
        # roller center contact-clearance plane 으로 변환해서 보냄.
        # target/n 은 OrientationConstraint 계산을 위해 아래에서 계속 yaml 에서 받음.
        target = get_target(self.cfg, self.active_target_name)
        _sp, n = self._active_surface_plane(target)
        snapped = self._stage3_tip_wps or [
            copy.deepcopy(wp) for wp in self.current_waypoints
        ]
        if not self._validate_contact_waypoints(snapped, target):
            self.executing = False
            return
        self.get_logger().info(
            f"표면 스냅 SKIP (새 perception 흐름): N={len(snapped)} "
            f"첫점=({snapped[0].position.x:.3f},{snapped[0].position.y:.3f},"
            f"{snapped[0].position.z:.3f})")

        # 3D 스플라인 밀집화
        densified = snapped
        self.get_logger().info(
            f"웨이포인트 밀집화: {len(snapped)} -> {len(densified)}")

        # roller center/tool tip → tcp 오프셋 적용
        tcp_wps = [self._brush_tip_to_tcp(wp) for wp in densified]

        # [EOAT_CHECK] 첫 waypoint 변환 검증
        _first_tip = densified[0]
        _first_tcp = tcp_wps[0]
        _dist = np.linalg.norm([
            _first_tip.position.x - _first_tcp.position.x,
            _first_tip.position.y - _first_tcp.position.y,
            _first_tip.position.z - _first_tcp.position.z,
        ])
        self.get_logger().info(
            f"[EOAT_CHECK] roller_center=({_first_tip.position.x:.3f},"
            f"{_first_tip.position.y:.3f},{_first_tip.position.z:.3f}) → "
            f"tcp=({_first_tcp.position.x:.3f},{_first_tcp.position.y:.3f},"
            f"{_first_tcp.position.z:.3f}) 거리={_dist*100:.1f}cm "
            f"(기대={EOAT_TIP_OFFSET*100:.1f}cm)"
        )

        req = GetCartesianPath.Request()
        req.header.frame_id = BASE_FRAME
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = PLANNING_GROUP
        req.link_name = EE_LINK

        rs = RobotState()
        rs.joint_state = self.current_joint_state
        rs.is_diff = False
        req.start_state = rs

        req.waypoints = tcp_wps
        req.max_step = 0.005  # 5mm
        req.jump_threshold = 0.0  # 점프 제한 없음
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = STAGE3_SPEED_SCALE
        req.max_acceleration_scaling_factor = STAGE3_SPEED_SCALE

        # ---- OrientationConstraint: tcp 의 EE 자세 유지 ----
        # tool 축(tcp 의 TOOL_AXIS) 중심 회전만 자유. 나머지는 tight.
        ee_q = self._active_ee_quat(target)
        brush_dir_world = -np.asarray(n, dtype=float)
        free_axis = int(np.argmax(np.abs(brush_dir_world)))
        tol = [0.2, 0.2, 0.2]
        tol[free_axis] = 3.14
        oc = OrientationConstraint()
        oc.header.frame_id = BASE_FRAME
        oc.link_name = EE_LINK
        oc.orientation.x = float(ee_q[0])
        oc.orientation.y = float(ee_q[1])
        oc.orientation.z = float(ee_q[2])
        oc.orientation.w = float(ee_q[3])
        oc.absolute_x_axis_tolerance = tol[0]
        oc.absolute_y_axis_tolerance = tol[1]
        oc.absolute_z_axis_tolerance = tol[2]
        oc.weight = 1.0
        req.path_constraints = Constraints()
        req.path_constraints.orientation_constraints.append(oc)

        # ---- 디버그: roller center 의 tcp TOOL_AXIS 방향 검증 ----
        first_tip = densified[0]
        tip_p = np.array([first_tip.position.x, first_tip.position.y, first_tip.position.z])
        local_axis_in_world = self._local_axis_in_world(
            first_tip.orientation, TOOL_AXIS)
        self.get_logger().info(
            f"[CHECK] roller_center=({tip_p[0]:.3f},{tip_p[1]:.3f},{tip_p[2]:.3f}) | "
            f"tcp local{TOOL_AXIS} in world=({local_axis_in_world[0]:+.2f},"
            f"{local_axis_in_world[1]:+.2f},{local_axis_in_world[2]:+.2f}) "
            f"[기대: -normal=({brush_dir_world[0]:+.2f},"
            f"{brush_dir_world[1]:+.2f},{brush_dir_world[2]:+.2f})]")
        self.get_logger().info(
            f"Cartesian 요청: {len(tcp_wps)} wp, 첫 tcp=("
            f"{tcp_wps[0].position.x:.3f},{tcp_wps[0].position.y:.3f},"
            f"{tcp_wps[0].position.z:.3f}) | "
            f"ori constraint free_axis={['x','y','z'][free_axis]} tol={tol}")

        fut = self.cartesian_client.call_async(req)
        fut.add_done_callback(self._cartesian_done)

    def _cartesian_done(self, future):
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(f"Cartesian 서비스 실패: {e}")
            self.executing = False
            return

        fraction = resp.fraction
        self.get_logger().info(f"Cartesian path: {fraction*100:.1f}% 달성")
        if fraction < MIN_CARTESIAN_FRACTION:
            self.get_logger().error(
                f"Cartesian fraction {fraction*100:.1f}% < "
                f"{MIN_CARTESIAN_FRACTION*100:.1f}% -> 부분 실행 금지, 중단")
            self.executing = False
            return

        traj = self._rescale_trajectory(resp.solution, scale=1.0)
        if not self.execute_trajectory_direct(
                traj, on_complete=self.stage4_retreat,
                force_guard=True, label="STAGE 3 contact path"):
            self.executing = False

    # ---- Stage 4: linear retreat (cartesian) --------------------------------
    def stage4_retreat(self):
        self.get_logger().info("=== STAGE 4: linear retreat (cartesian) ===")
        if not self.cartesian_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/compute_cartesian_path 서비스 없음 (Stage 4)")
            self.executing = False
            return

        target = get_target(self.cfg, self.active_target_name)
        _, n = self._active_surface_plane(target)

        req = GetCartesianPath.Request()
        req.header.frame_id = BASE_FRAME
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = PLANNING_GROUP
        req.link_name = EE_LINK

        rs = RobotState()
        rs.joint_state = self.current_joint_state
        rs.is_diff = False
        req.start_state = rs

        req.waypoints = [self._retreat_tcp_pose]
        req.max_step = 0.005
        req.jump_threshold = 5.0
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = STAGE4_SPEED_SCALE
        req.max_acceleration_scaling_factor = STAGE4_SPEED_SCALE

        ee_q = self._active_ee_quat(target)
        brush_dir_world = -np.asarray(n, dtype=float)
        free_axis = int(np.argmax(np.abs(brush_dir_world)))
        tol = [0.2, 0.2, 0.2]
        tol[free_axis] = 3.14
        oc = OrientationConstraint()
        oc.header.frame_id = BASE_FRAME
        oc.link_name = EE_LINK
        oc.orientation.x = float(ee_q[0])
        oc.orientation.y = float(ee_q[1])
        oc.orientation.z = float(ee_q[2])
        oc.orientation.w = float(ee_q[3])
        oc.absolute_x_axis_tolerance = tol[0]
        oc.absolute_y_axis_tolerance = tol[1]
        oc.absolute_z_axis_tolerance = tol[2]
        oc.weight = 1.0
        req.path_constraints = Constraints()
        req.path_constraints.orientation_constraints.append(oc)

        if not self._check_normal_motion(
                self._stage3_tcp_wps[-1], self._retreat_tcp_pose,
                n, MIN_APPROACH_NORMAL_ALIGN, "STAGE 4"):
            self.executing = False
            return

        future = self.cartesian_client.call_async(req)
        future.add_done_callback(self._stage4_done)

    def _stage4_done(self, future):
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(f"STAGE 4 서비스 실패: {e}")
            self.executing = False
            return
        fraction = resp.fraction
        self.get_logger().info(f"STAGE 4 cartesian: {fraction*100:.1f}%")
        if fraction < MIN_RETREAT_CARTESIAN_FRACTION:
            self.get_logger().error(
                f"STAGE 4 fraction {fraction*100:.1f}% < "
                f"{MIN_RETREAT_CARTESIAN_FRACTION*100:.1f}% -> 중단")
            self.executing = False
            return
        if fraction < MIN_CARTESIAN_FRACTION:
            self.get_logger().warn(
                f"STAGE 4 partial retreat 허용: {fraction*100:.1f}% "
                f"(< {MIN_CARTESIAN_FRACTION*100:.1f}%). "
                "벽에서 멀어지는 안전 방향이므로 계산된 구간만 실행합니다.")
        traj = self._rescale_trajectory(resp.solution, scale=1.0)
        if not self.execute_trajectory_direct(
                traj, on_complete=self._all_stages_done):
            self.executing = False

    def _all_stages_done(self):
        self.get_logger().info("=" * 60)
        self.get_logger().info(">>> Stage 1→2→3→4 완료")
        if RETURN_TO_READY_AFTER_SKETCH:
            self.stage5_return_to_ready()
        else:
            self.get_logger().info(
                ">>> 모든 stage 완료 (1→2→3→4, READY_POSE 복귀 없음)")
            self.get_logger().info("=" * 60)
            self.executing = False

    # ---- Stage 5: return to READY_POSE (joint goal via OMPL) ----------------
    def _is_at_ready_pose(self, tol_rad=0.05):
        """현재 joint state 가 READY_POSE 와 가까운지 (joint 당 tol_rad 이내)."""
        if self.current_joint_state is None:
            return False
        cs = dict(zip(
            self.current_joint_state.name,
            self.current_joint_state.position
        ))
        for jn, target in READY_POSE_JOINTS.items():
            if jn not in cs:
                return False
            if abs(cs[jn] - target) > tol_rad:
                return False
        return True

    def stage5_return_to_ready(self):
        self.get_logger().info(
            f"=== STAGE 5: return to READY_POSE (joint goal, {PLANNER_ID}) ===")
        self._plan_joint_goal(
            "READY_POSE",
            READY_POSE_JOINTS,
            STAGE5_SPEED_SCALE,
            finalize_cb=self._stage5_finalize,
        )

    def _plan_joint_goal(self, label, joints, speed_scale, finalize_cb):
        if not self.move_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn(
                f"MoveGroup action server 없음 — {label} 이동 불가")
            finalize_cb(success=False)
            return
        if self.current_joint_state is None:
            self.get_logger().warn(f"joint_state 미수신 — {label} 이동 불가")
            finalize_cb(success=False)
            return
        if not self._joint_state_within_limits(
                self.current_joint_state, f"{label} start"):
            self.get_logger().warn(
                f"{label} 이동 불가 — 현재 joint_state 가 limit 밖")
            finalize_cb(success=False)
            return

        self._joint_goal_context = {
            "label": label,
            "finalize_cb": finalize_cb,
        }

        goal = MoveGroup.Goal()
        goal.request.group_name = PLANNING_GROUP
        rs = RobotState()
        rs.joint_state = self.current_joint_state
        rs.is_diff = False
        goal.request.start_state = rs

        # Joint goal constraints
        constraints = Constraints()
        for jn, target in joints.items():
            jc = JointConstraint()
            jc.joint_name = jn
            jc.position = float(target)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal.request.goal_constraints = [constraints]

        goal.request.planner_id = PLANNER_ID
        goal.request.allowed_planning_time = ALLOWED_PLANNING_TIME
        goal.request.num_planning_attempts = PLANNING_ATTEMPTS
        goal.request.max_velocity_scaling_factor = speed_scale
        goal.request.max_acceleration_scaling_factor = speed_scale

        goal.planning_options.plan_only = True
        goal.planning_options.planning_scene_diff.is_diff = True

        future = self.move_action_client.send_goal_async(goal)
        future.add_done_callback(self._joint_goal_response)

    def _joint_goal_response(self, future):
        ctx = self._joint_goal_context or {}
        label = ctx.get("label", "joint goal")
        finalize_cb = ctx.get("finalize_cb", lambda success: None)
        try:
            handle = future.result()
        except Exception as e:
            self.get_logger().warn(f"{label} send_goal 실패: {e}")
            finalize_cb(success=False)
            return
        if not handle.accepted:
            self.get_logger().warn(f"{label} goal rejected")
            finalize_cb(success=False)
            return
        self.get_logger().info(f"{label} goal accepted, planning...")
        handle.get_result_async().add_done_callback(self._joint_goal_result)

    def _joint_goal_result(self, future):
        ctx = self._joint_goal_context or {}
        label = ctx.get("label", "joint goal")
        finalize_cb = ctx.get("finalize_cb", lambda success: None)
        try:
            result = future.result().result
        except Exception as e:
            self.get_logger().warn(f"{label} result 실패: {e}")
            finalize_cb(success=False)
            return
        if result.error_code.val != 1:
            self.get_logger().warn(
                f"{label} planning 실패 error_code={result.error_code.val}")
            finalize_cb(success=False)
            return

        traj = result.planned_trajectory
        n_points = len(traj.joint_trajectory.points)
        self.get_logger().info(f"{label} planning OK: {n_points} 포인트")
        traj = self._rescale_trajectory(traj, scale=1.0)
        if not self.execute_trajectory_direct(
                traj, on_complete=lambda: finalize_cb(success=True)):
            finalize_cb(success=False)

    def _stage5_finalize(self, success):
        self._joint_goal_context = None
        if success:
            self.get_logger().info("Stage 5 완료: READY_POSE 복귀")
            self.get_logger().info(">>> 모든 stage 완료 (1→2→3→4→5)")
        else:
            self.get_logger().warn(
                "Stage 5 (READY 복귀) 실패. 다음 Submit 의 시작 위치 부적합 가능.")
        self.get_logger().info("=" * 60)
        self.executing = False

    def _rescale_trajectory(self, traj, scale=0.3):
        if not traj.joint_trajectory.points:
            return traj
        if scale >= 0.999:
            return traj
        for p in traj.joint_trajectory.points:
            total_ns = p.time_from_start.sec * 1_000_000_000 + p.time_from_start.nanosec
            total_ns = int(total_ns / scale)
            p.time_from_start.sec = total_ns // 1_000_000_000
            p.time_from_start.nanosec = total_ns % 1_000_000_000
            if p.velocities:
                p.velocities = [v * scale for v in p.velocities]
            if p.accelerations:
                p.accelerations = [a * scale * scale for a in p.accelerations]
        return traj


def main(args=None):
    rclpy.init(args=args)
    node = MoveItExecutor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
