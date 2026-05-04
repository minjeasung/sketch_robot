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

토치(EoAT)는 tcp 의 TORCH_MOUNT_AXIS 방향으로 뻗음. 마운팅 변경 시 그 상수만 갱신.
"""
import copy
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.action import ActionClient
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener

from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from moveit_msgs.srv import GetCartesianPath, GetPositionIK, ApplyPlanningScene
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    PositionIKRequest, RobotState,
    CollisionObject, AttachedCollisionObject,
    PlanningScene, PlanningSceneWorld,
    Constraints, OrientationConstraint, PositionConstraint, JointConstraint,
)
from shape_msgs.msg import SolidPrimitive
from control_msgs.action import FollowJointTrajectory

from sketch_control.targets import (
    load_objects_config, get_surface_plane, get_target, ee_quat_for_target,
)


# SRDF 의 그룹명이 "mainpulation" 으로 오타. 그대로 유지 (RB10 공식 SRDF 기준).
PLANNING_GROUP = "mainpulation"
EE_LINK = "tcp"
BASE_FRAME = "link0"

BRUSH_PRESS_DEPTH = 0.003  # 3mm 침투 (paint threshold 내)

SAFETY_OFFSET = 0.05   # 5cm 짧은 normal-direction approach
RETREAT_OFFSET = 0.15  # Stage 4 후퇴점: 표면 normal 방향 15cm
PLANNER_ID = "RRTConnect"
ALLOWED_PLANNING_TIME = 5.0
PLANNING_ATTEMPTS = 5

# Stage 5 — READY_POSE 복귀
RETURN_TO_READY = True   # False 면 Stage 5 비활성 (디버깅용)

# RB10 joint 운동학 순서 (URDF 기준).
# 주의: /joint_states 토픽은 알파벳 순으로 발행됨 (base, elbow, shoulder, wrist1, wrist2, wrist3) —
# 이 dict 는 이름 매핑이라 순서 무관, 안전.
# 각 값은 RB10 실로봇에서 teach pendant 로 안전한 자세 만든 후 /joint_states 읽어 실측해야 함.
READY_POSE_JOINTS = {
    "base":     0.0,  # TODO: RB10 실측 필요
    "shoulder": 0.0,  # TODO: RB10 실측 필요
    "elbow":    0.0,  # TODO: RB10 실측 필요
    "wrist1":   0.0,  # TODO: RB10 실측 필요
    "wrist2":   0.0,  # TODO: RB10 실측 필요
    "wrist3":   0.0,  # TODO: RB10 실측 필요
}

# RB10 link0 가 world 원점에 fixed_joint 로 박혀있음. 작업대 위에 따로 옮기면 변경 필요.
ROBOT_ORIGIN = (0.0, 0.0, 0.0)

# 토치 길이 (tcp → torch_tip)
TORCH_LENGTH = 0.25
# 역호환 alias (내부 함수용)
BRUSH_LENGTH = TORCH_LENGTH

# 토치가 EE link (tcp) 의 로컬 어느 축으로 뻗는지.
# 값: "+x", "-x", "+y", "-y", "+z", "-z"
# 기본 +y 는 UR10 시뮬 시절 가정. RB10 실로봇 마운팅 후 실측해서 변경.
TORCH_MOUNT_AXIS = "+y"  # TODO: RB10 + 새 EoAT 마운팅 후 실측 필요


def _torch_attach_quat(axis):
    """SolidPrimitive.CYLINDER (default +z) 를 axis 방향으로 회전시키는 quaternion (x,y,z,w).
    scipy 의 align_vectors 로 동적 계산하여 부호 실수 방지."""
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
    rot, _ = R.align_vectors(tgt[None, :], src[None, :])
    q = rot.as_quat()  # [x, y, z, w]
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _torch_attach_offset(axis):
    """cylinder 중심 위치 (tcp 기준), TORCH_LENGTH/2 만큼 axis 방향."""
    sign = -1.0 if axis.startswith("-") else 1.0
    a = axis[1]
    half = TORCH_LENGTH / 2.0 * sign
    if a == "x":
        return (half, 0.0, 0.0)
    if a == "y":
        return (0.0, half, 0.0)
    if a == "z":
        return (0.0, 0.0, half)
    raise ValueError(f"unknown axis: {axis}")


class MoveItExecutor(Node):
    def __init__(self):
        super().__init__("moveit_executor")

        # I/O
        self.create_subscription(PoseArray, "/sketch_waypoints", self.on_waypoints, 10)
        self.create_subscription(Bool, "/sketch_execute", self.on_execute, 10)
        self.create_subscription(JointState, "/joint_states", self.on_joint_state, 10)
        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)

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
        self._stage1_retried = False

        self.current_waypoints = []
        self.current_joint_state = None
        self.scene_initialized = False
        self.scene_confirmed = False
        self.executing = False

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
        self.create_timer(3.0, self._log_current_tcp)

        self.get_logger().info(
            "MoveIt Executor 노드 시작 (plan + FollowJointTrajectory action)")

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
        # world 프레임 웨이포인트를 link0 프레임으로 변환
        self.current_waypoints = []
        for p in msg.poses:
            bp = copy.deepcopy(p)
            bp.position.x -= ROBOT_ORIGIN[0]
            bp.position.y -= ROBOT_ORIGIN[1]
            bp.position.z -= ROBOT_ORIGIN[2]
            self.current_waypoints.append(bp)
        self.get_logger().info(
            f"{len(self.current_waypoints)}개 웨이포인트 수신 "
            f"(link0 변환: 첫 점 x={self.current_waypoints[0].position.x:.2f} "
            f"y={self.current_waypoints[0].position.y:.2f} z={self.current_waypoints[0].position.z:.2f})")

    def on_joint_state(self, msg: JointState):
        self.current_joint_state = msg

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

        # 첫 Submit 안전성: READY_POSE 가 아니면 사용자에게 경고만.
        # 자동 복귀는 비동기 chain 이라 Stage 1 시작과 race 됨 — 단순 경고로 처리.
        if RETURN_TO_READY and not self._is_at_ready_pose():
            self.get_logger().warn(
                "현재 자세가 READY_POSE 와 다름. Stage 1 path 가 wall 닿을 수 있음. "
                "Stage 5 (자동 복귀) 가 다음 Submit 부터 보장.")

        # Stage 2/3 가 쓸 표면 스냅된 tcp waypoints 미리 계산
        densified_tip, tcp_wps, target, n = self._compute_snapped_tcp_waypoints()
        self._stage3_tcp_wps = tcp_wps

        # Stage 1 의 목표: 첫 점 표면 위치 → normal 방향 SAFETY_OFFSET 후퇴
        fixed_q = ee_quat_for_target(target)
        first_tip = densified_tip[0]
        safety_tip = self._offset_along_normal(first_tip, SAFETY_OFFSET)
        safety_tcp = self._brush_tip_to_tcp(safety_tip)
        safety_tcp.orientation.x = float(fixed_q[0])
        safety_tcp.orientation.y = float(fixed_q[1])
        safety_tcp.orientation.z = float(fixed_q[2])
        safety_tcp.orientation.w = float(fixed_q[3])
        self._safety_tcp_pose = safety_tcp

        # Stage 4 의 목표: 마지막 점 → normal 방향 RETREAT_OFFSET 후퇴
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
        self.stage1_approach_free()

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

    # ---- PlanningScene (물체들 + 붓 AttachedCollisionObject) ------------------
    def publish_scene_periodic(self):
        if self.scene_confirmed:
            return

        ps = PlanningScene()
        ps.is_diff = True
        ps.world = PlanningSceneWorld()

        # --- 활성 물체 전부 ---
        for obj in self.cfg["objects"]:
            if not obj.get("enabled", True):
                continue
            co = CollisionObject()
            co.id = obj["name"]
            co.header.frame_id = BASE_FRAME
            prim = SolidPrimitive()
            prim.type = SolidPrimitive.BOX
            prim.dimensions = list(obj["size"])
            pose = Pose()
            pose.position.x = obj["position"][0] - ROBOT_ORIGIN[0]
            pose.position.y = obj["position"][1] - ROBOT_ORIGIN[1]
            pose.position.z = obj["position"][2] - ROBOT_ORIGIN[2]
            pose.orientation.w = 1.0
            co.primitives.append(prim)
            co.primitive_poses.append(pose)
            co.operation = CollisionObject.ADD
            ps.world.collision_objects.append(co)

        # --- 토치 (tcp 에 attached, cylinder 25cm, radius 2.5cm) ---
        # 실제 torch 는 tcp 의 TORCH_MOUNT_AXIS 방향으로 뻗음.
        # SolidPrimitive.CYLINDER 의 기본 axis 는 +Z 이므로 회전 적용.
        torch_aco = AttachedCollisionObject()
        torch_aco.link_name = EE_LINK  # "tcp"
        torch_aco.object.id = "brush"  # id 는 기존 유지 (scene_confirmed 로직 호환)
        torch_aco.object.header.frame_id = EE_LINK
        torch_prim = SolidPrimitive()
        torch_prim.type = SolidPrimitive.CYLINDER
        torch_prim.dimensions = [TORCH_LENGTH, 0.025]  # [height=0.25, radius=0.025]
        torch_pose = Pose()
        # 중심 위치를 axis 방향 TORCH_LENGTH/2 로 (cylinder 가 0 ~ TORCH_LENGTH 차지)
        ox, oy, oz = _torch_attach_offset(TORCH_MOUNT_AXIS)
        torch_pose.position.x = ox
        torch_pose.position.y = oy
        torch_pose.position.z = oz
        # cylinder 의 default +Z axis 를 TORCH_MOUNT_AXIS 로 회전
        qx, qy, qz, qw = _torch_attach_quat(TORCH_MOUNT_AXIS)
        torch_pose.orientation.x = qx
        torch_pose.orientation.y = qy
        torch_pose.orientation.z = qz
        torch_pose.orientation.w = qw
        torch_aco.object.primitives.append(torch_prim)
        torch_aco.object.primitive_poses.append(torch_pose)
        torch_aco.object.operation = CollisionObject.ADD
        # RB10 link 이름. 손목 마지막 link 들 + tcp 자체.
        torch_aco.touch_links = ["tcp", "link6", "link5"]
        ps.robot_state.attached_collision_objects.append(torch_aco)
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

    # ---- 궤적 실행 (FollowJointTrajectory action) ---------------------------
    def execute_trajectory_direct(self, traj, on_complete=None):
        """RB10 driver 의 FollowJointTrajectory action 으로 trajectory 전송.
        on_complete: action 성공 후 호출할 callback (다음 stage 트리거용).
        함수명은 호출처 호환을 위해 유지."""
        if not traj.joint_trajectory.points:
            self.get_logger().warn("빈 궤적")
            if on_complete is None:
                self.executing = False
            return

        if not self.traj_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("FollowJointTrajectory action server 없음")
            self.executing = False
            return

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

    # ---- helpers for 4-stage approach ---------------------------------------
    def _offset_along_normal(self, pose, distance):
        """pose 를 active target 의 표면 normal 방향으로 distance 만큼 후퇴."""
        target = get_target(self.cfg, self.active_target_name)
        _, n = get_surface_plane(target)
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

    def _compute_snapped_tcp_waypoints(self):
        """current_waypoints 를 표면 스냅 + densify + tcp 변환한 결과 반환.
        Stage 2/3 양쪽이 같은 변환 결과를 써야 일관됨.
        Returns: (snapped_tip_wps, tcp_wps, target, n)
        """
        target = get_target(self.cfg, self.active_target_name)
        sp, n = get_surface_plane(target)
        fixed_q = ee_quat_for_target(target)
        snapped = []
        for wp in self.current_waypoints:
            p = np.array([wp.position.x, wp.position.y, wp.position.z])
            delta = float(np.dot(p - sp, n))
            projected = p - delta * n - n * BRUSH_PRESS_DEPTH
            sw = copy.deepcopy(wp)
            sw.position.x = float(projected[0])
            sw.position.y = float(projected[1])
            sw.position.z = float(projected[2])
            sw.orientation.x = float(fixed_q[0])
            sw.orientation.y = float(fixed_q[1])
            sw.orientation.z = float(fixed_q[2])
            sw.orientation.w = float(fixed_q[3])
            snapped.append(sw)
        densified = self._densify_waypoints(snapped, spacing_m=0.005)
        tcp_wps = [self._brush_tip_to_tcp(wp) for wp in densified]
        return densified, tcp_wps, target, n

    # ---- Stage 1: free-space approach via MoveGroup action -------------------
    def stage1_approach_free(self):
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

        goal = MoveGroup.Goal()
        goal.request.group_name = PLANNING_GROUP
        rs = RobotState()
        rs.joint_state = self.current_joint_state
        rs.is_diff = False
        goal.request.start_state = rs
        goal.request.goal_constraints = [
            self._make_pose_constraints(self._safety_tcp_pose)
        ]
        goal.request.planner_id = PLANNER_ID
        goal.request.allowed_planning_time = ALLOWED_PLANNING_TIME
        goal.request.num_planning_attempts = PLANNING_ATTEMPTS
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3

        goal.planning_options.plan_only = True
        goal.planning_options.planning_scene_diff.is_diff = True

        future = self.move_action_client.send_goal_async(goal)
        future.add_done_callback(self._stage1_goal_response)

    def _retry_stage1_with_default_planner(self):
        """planner_id 를 비워서 MoveGroup 의 기본 planner 로 재시도."""
        goal = MoveGroup.Goal()
        goal.request.group_name = PLANNING_GROUP
        rs = RobotState()
        rs.joint_state = self.current_joint_state
        rs.is_diff = False
        goal.request.start_state = rs
        goal.request.goal_constraints = [
            self._make_pose_constraints(self._safety_tcp_pose)
        ]
        # planner_id 비움 — 서버 기본 사용
        goal.request.planner_id = ""
        goal.request.allowed_planning_time = ALLOWED_PLANNING_TIME
        goal.request.num_planning_attempts = PLANNING_ATTEMPTS
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3
        goal.planning_options.plan_only = True
        goal.planning_options.planning_scene_diff.is_diff = True

        future = self.move_action_client.send_goal_async(goal)
        future.add_done_callback(self._stage1_goal_response)

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

        # 0.3x 스케일링 (cartesian 과 동일)
        traj = self._rescale_trajectory(traj, scale=0.3)
        self.execute_trajectory_direct(
            traj, on_complete=self.stage2_approach_linear)

    # ---- Stage 2: linear approach to first surface point (cartesian) --------
    def stage2_approach_linear(self):
        self.get_logger().info("=== STAGE 2: linear approach (cartesian) ===")
        if not self.cartesian_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/compute_cartesian_path 서비스 없음 (Stage 2)")
            self.executing = False
            return

        target = get_target(self.cfg, self.active_target_name)
        _, n = get_surface_plane(target)

        req = GetCartesianPath.Request()
        req.header.frame_id = BASE_FRAME
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = PLANNING_GROUP
        req.link_name = EE_LINK

        rs = RobotState()
        rs.joint_state = self.current_joint_state
        rs.is_diff = False
        req.start_state = rs

        # 목표: 표면 첫 점 (tcp 좌표)
        req.waypoints = [self._stage3_tcp_wps[0]]
        req.max_step = 0.005
        req.jump_threshold = 5.0
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3

        # Orientation constraint (Stage 3 와 동일)
        ee_q = ee_quat_for_target(target)
        brush_dir_world = -np.asarray(n, dtype=float)
        free_axis = int(np.argmax(np.abs(brush_dir_world)))
        tol = [0.2, 0.2, 0.2]
        tol[free_axis] = 3.14
        oc = OrientationConstraint()
        oc.header.frame_id = "world"
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
        _, n_target = get_surface_plane(target)
        align = float(np.dot(direction, -np.asarray(n_target)))  # +1 이 완벽한 normal 진입
        self.get_logger().info(
            f"[STAGE 2 DEBUG] dist={dist*100:.1f}cm "
            f"direction=({direction[0]:+.2f},{direction[1]:+.2f},{direction[2]:+.2f}) "
            f"normal_align={align:+.3f} (1.0=perfect)")

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
        if fraction < 0.95:
            self.get_logger().error(
                f"STAGE 2 fraction {fraction*100:.0f}% < 95% -> 중단")
            self.executing = False
            return
        traj = self._rescale_trajectory(resp.solution, scale=0.3)
        self.execute_trajectory_direct(
            traj, on_complete=self.plan_cartesian)

    # ---- torch_tip → tcp 오프셋 변환 ------------------------------------------
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
        """torch_tip 기준 좌표를 tcp 기준으로 변환.
        토치는 URDF tcp (RB10) / tool0 (UR10) 의 로컬 TORCH_MOUNT_AXIS 방향으로
        TORCH_LENGTH 만큼 뻗음."""
        axis_world = MoveItExecutor._local_axis_in_world(
            pose.orientation, TORCH_MOUNT_AXIS)
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        new_pos = pos - TORCH_LENGTH * axis_world
        new_pose = Pose()
        new_pose.position.x = float(new_pos[0])
        new_pose.position.y = float(new_pos[1])
        new_pose.position.z = float(new_pos[2])
        new_pose.orientation = copy.deepcopy(pose.orientation)
        return new_pose

    # ---- 3D 웨이포인트 스플라인 밀집화 ------------------------------------------
    @staticmethod
    def _densify_waypoints(waypoints, spacing_m=0.005):
        """3D poses 를 cubic spline 으로 보간 후 등간격 재샘플."""
        if len(waypoints) < 4:
            return waypoints
        positions = np.array([[p.position.x, p.position.y, p.position.z]
                              for p in waypoints])
        # 중복 점 제거 (CubicSpline strictly increasing t 요구)
        diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        keep = np.concatenate([[True], diffs > 1e-6])
        positions = positions[keep]
        if len(positions) < 4:
            return waypoints
        dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        t = np.concatenate([[0], np.cumsum(dists)])
        total = t[-1]
        if total < 1e-6:
            return waypoints
        cs = [CubicSpline(t, positions[:, i], bc_type='natural') for i in range(3)]
        n = max(len(waypoints), int(total / spacing_m))
        t_new = np.linspace(0, total, n)
        new_wps = []
        ref_ori = waypoints[0].orientation
        for ti in t_new:
            p = Pose()
            p.position.x = float(cs[0](ti))
            p.position.y = float(cs[1](ti))
            p.position.z = float(cs[2](ti))
            p.orientation = copy.deepcopy(ref_ori)
            new_wps.append(p)
        return new_wps

    # ---- Stage 3: Cartesian path 계획 + 실행 --------------------------------
    def plan_cartesian(self):
        if not self.cartesian_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/compute_cartesian_path 서비스 없음")
            return

        # 표면 스냅: 모든 waypoint 를 active target 의 평면 위로 투영 후 -depth 만큼 침투
        target = get_target(self.cfg, self.active_target_name)
        sp, n = get_surface_plane(target)
        fixed_q = ee_quat_for_target(target)
        snapped = []
        for wp in self.current_waypoints:
            p = np.array([wp.position.x, wp.position.y, wp.position.z])
            delta = float(np.dot(p - sp, n))
            projected = p - delta * n - n * BRUSH_PRESS_DEPTH
            sw = copy.deepcopy(wp)
            sw.position.x = float(projected[0])
            sw.position.y = float(projected[1])
            sw.position.z = float(projected[2])
            # orientation 을 face 기반으로 강제 (sketch_ui 값과 동일해야 함)
            sw.orientation.x = float(fixed_q[0])
            sw.orientation.y = float(fixed_q[1])
            sw.orientation.z = float(fixed_q[2])
            sw.orientation.w = float(fixed_q[3])
            snapped.append(sw)
        self.get_logger().info(
            f"표면 스냅: target={self.active_target_name} normal={n} "
            f"첫점=({snapped[0].position.x:.3f},{snapped[0].position.y:.3f},"
            f"{snapped[0].position.z:.3f})")

        # 3D 스플라인 밀집화
        densified = self._densify_waypoints(snapped, spacing_m=0.005)
        self.get_logger().info(
            f"웨이포인트 밀집화: {len(snapped)} -> {len(densified)}")

        # torch_tip → tcp 오프셋 적용
        tcp_wps = [self._brush_tip_to_tcp(wp) for wp in densified]

        # [TORCH_CHECK] 첫 waypoint 변환 검증
        _first_tip = densified[0]
        _first_tcp = tcp_wps[0]
        _dist = np.linalg.norm([
            _first_tip.position.x - _first_tcp.position.x,
            _first_tip.position.y - _first_tcp.position.y,
            _first_tip.position.z - _first_tcp.position.z,
        ])
        self.get_logger().info(
            f"[TORCH_CHECK] torch_tip=({_first_tip.position.x:.3f},"
            f"{_first_tip.position.y:.3f},{_first_tip.position.z:.3f}) → "
            f"tcp=({_first_tcp.position.x:.3f},{_first_tcp.position.y:.3f},"
            f"{_first_tcp.position.z:.3f}) 거리={_dist*100:.1f}cm "
            f"(기대={TORCH_LENGTH*100:.0f}cm)"
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
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3

        # ---- OrientationConstraint: tcp 의 EE 자세 유지 ----
        # 붓 축(tcp 의 TORCH_MOUNT_AXIS) 중심 회전만 자유. 나머지는 tight.
        ee_q = ee_quat_for_target(target)
        brush_dir_world = -np.asarray(n, dtype=float)
        free_axis = int(np.argmax(np.abs(brush_dir_world)))
        tol = [0.2, 0.2, 0.2]
        tol[free_axis] = 3.14
        oc = OrientationConstraint()
        oc.header.frame_id = "world"
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

        # ---- 디버그: brush_tip 의 tcp TORCH_MOUNT_AXIS 방향 검증 ----
        first_tip = densified[0]
        tip_p = np.array([first_tip.position.x, first_tip.position.y, first_tip.position.z])
        local_axis_in_world = self._local_axis_in_world(
            first_tip.orientation, TORCH_MOUNT_AXIS)
        self.get_logger().info(
            f"[CHECK] brush_tip=({tip_p[0]:.3f},{tip_p[1]:.3f},{tip_p[2]:.3f}) | "
            f"tcp local{TORCH_MOUNT_AXIS} in world=({local_axis_in_world[0]:+.2f},"
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
        if fraction < 0.5:
            self.get_logger().error(
                "50% 미만 -> 스케치를 재생성하거나 타겟을 변경하세요.")
            self.executing = False
            return
        if fraction < 0.95:
            self.get_logger().warn(f"{fraction*100:.1f}% -> 부분 실행")

        traj = self._rescale_trajectory(resp.solution, scale=0.3)
        self.execute_trajectory_direct(
            traj, on_complete=self.stage4_retreat)

    # ---- Stage 4: linear retreat (cartesian) --------------------------------
    def stage4_retreat(self):
        self.get_logger().info("=== STAGE 4: linear retreat (cartesian) ===")
        if not self.cartesian_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/compute_cartesian_path 서비스 없음 (Stage 4)")
            self.executing = False
            return

        target = get_target(self.cfg, self.active_target_name)
        _, n = get_surface_plane(target)

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
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3

        ee_q = ee_quat_for_target(target)
        brush_dir_world = -np.asarray(n, dtype=float)
        free_axis = int(np.argmax(np.abs(brush_dir_world)))
        tol = [0.2, 0.2, 0.2]
        tol[free_axis] = 3.14
        oc = OrientationConstraint()
        oc.header.frame_id = "world"
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
        if fraction < 0.5:
            self.get_logger().warn(
                f"STAGE 4 fraction {fraction*100:.0f}% 낮음 -> 중단 (토치 표면에 남음)")
            self.executing = False
            return
        traj = self._rescale_trajectory(resp.solution, scale=0.3)
        self.execute_trajectory_direct(traj, on_complete=self._all_stages_done)

    def _all_stages_done(self):
        self.get_logger().info("=" * 60)
        self.get_logger().info(">>> Stage 1→2→3→4 완료")
        if RETURN_TO_READY:
            self.stage5_return_to_ready()
        else:
            self.get_logger().info(">>> 모든 stage 완료 (1→2→3→4)")
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
        if not self.move_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn(
                "MoveGroup action server 없음 — Stage 5 skip (READY 미복귀)")
            self._stage5_finalize(success=False)
            return

        goal = MoveGroup.Goal()
        goal.request.group_name = PLANNING_GROUP
        rs = RobotState()
        rs.joint_state = self.current_joint_state
        rs.is_diff = False
        goal.request.start_state = rs

        # Joint goal constraints
        constraints = Constraints()
        for jn, target in READY_POSE_JOINTS.items():
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
        # Stage 5 는 빈 공간 이동 — Stage 1 보다 빠르게
        goal.request.max_velocity_scaling_factor = 0.5
        goal.request.max_acceleration_scaling_factor = 0.5

        goal.planning_options.plan_only = True
        goal.planning_options.planning_scene_diff.is_diff = True

        future = self.move_action_client.send_goal_async(goal)
        future.add_done_callback(self._stage5_goal_response)

    def _stage5_goal_response(self, future):
        try:
            handle = future.result()
        except Exception as e:
            self.get_logger().warn(f"STAGE 5 send_goal 실패: {e}")
            self._stage5_finalize(success=False)
            return
        if not handle.accepted:
            self.get_logger().warn("STAGE 5 goal rejected")
            self._stage5_finalize(success=False)
            return
        self.get_logger().info("STAGE 5 goal accepted, planning...")
        handle.get_result_async().add_done_callback(self._stage5_result)

    def _stage5_result(self, future):
        try:
            result = future.result().result
        except Exception as e:
            self.get_logger().warn(f"STAGE 5 result 실패: {e}")
            self._stage5_finalize(success=False)
            return
        if result.error_code.val != 1:
            self.get_logger().warn(
                f"STAGE 5 planning 실패 error_code={result.error_code.val}")
            self._stage5_finalize(success=False)
            return

        traj = result.planned_trajectory
        n_points = len(traj.joint_trajectory.points)
        self.get_logger().info(f"STAGE 5 planning OK: {n_points} 포인트")
        traj = self._rescale_trajectory(traj, scale=0.5)
        self.execute_trajectory_direct(
            traj, on_complete=lambda: self._stage5_finalize(success=True))

    def _stage5_finalize(self, success):
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
