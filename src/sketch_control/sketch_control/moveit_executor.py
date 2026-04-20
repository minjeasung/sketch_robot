"""
MoveIt Executor - Cartesian Path 계획 + 직접 Joint Command 실행 (벽에 쓰기)

Isaac Sim 에는 FollowJointTrajectory 액션 서버가 없으므로,
MoveIt 으로 계획만 한 뒤 궤적 포인트를 /joint_command 로 직접 퍼블리시.
붓 TCP (tool0 + 15cm Z) 오프셋 + 3D 스플라인 경로 스무딩.
"""
import copy
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from scipy.interpolate import CubicSpline
from tf2_ros import Buffer, TransformListener

from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from moveit_msgs.msg import (
    PositionIKRequest, RobotState,
    CollisionObject, AttachedCollisionObject,
    PlanningScene, PlanningSceneWorld,
    Constraints, OrientationConstraint,
)
from shape_msgs.msg import SolidPrimitive

from sketch_control.targets import (
    load_objects_config, get_surface_plane, get_target, ee_quat_for_target,
)


PLANNING_GROUP = "ur_manipulator"
EE_LINK = "tool0"
BASE_FRAME = "base_link"

BRUSH_PRESS_DEPTH = 0.003  # 3mm 침투 (paint threshold 내)

# UR10 이 원점에 있으므로 오프셋 없음
ROBOT_ORIGIN = (0.0, 0.0, 0.0)

# 토치 길이 (tool0 → torch_tip)
TORCH_LENGTH = 0.25
# 역호환 alias (내부 함수용)
BRUSH_LENGTH = TORCH_LENGTH


class MoveItExecutor(Node):
    def __init__(self):
        super().__init__("moveit_executor")

        # I/O
        self.create_subscription(PoseArray, "/sketch_waypoints", self.on_waypoints, 10)
        self.create_subscription(Bool, "/sketch_execute", self.on_execute, 10)
        self.create_subscription(JointState, "/joint_states", self.on_joint_state, 10)
        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)
        self.joint_cmd_pub = self.create_publisher(JointState, "/joint_command", 10)

        # MoveIt endpoints (계획만 사용)
        self.cartesian_client = self.create_client(
            GetCartesianPath, "/compute_cartesian_path")
        self.ik_client = self.create_client(
            GetPositionIK, "/compute_ik")

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

        # ---- 진단: 현재 tool0 TF 를 3초마다 출력 (수동 캘리브레이션 용) ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(3.0, self._log_current_tool0)

        self.get_logger().info("MoveIt Executor 노드 시작 (plan + direct joint command)")

    def _log_current_tool0(self):
        """world → tool0 TF 를 주기적으로 로그. 수동 캘리브레이션 시 사용."""
        try:
            tf = self.tf_buffer.lookup_transform(
                "world", "tool0", rclpy.time.Time(),
                timeout=Duration(seconds=0.3),
            )
        except Exception:
            # world 프레임 없으면 World 대문자 시도
            try:
                tf = self.tf_buffer.lookup_transform(
                    "World", "tool0", rclpy.time.Time(),
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
            f"[TOOL0_NOW] pos=({p.x:+.3f},{p.y:+.3f},{p.z:+.3f}) "
            f"quat=({q.x:+.3f},{q.y:+.3f},{q.z:+.3f},{q.w:+.3f})"
        )
        self.get_logger().info(
            f"           local_X_in_world=({local_x[0]:+.2f},{local_x[1]:+.2f},{local_x[2]:+.2f})"
        )
        self.get_logger().info(
            f"           local_Y_in_world=({local_y[0]:+.2f},{local_y[1]:+.2f},{local_y[2]:+.2f})"
        )
        self.get_logger().info(
            f"           local_Z_in_world=({local_z[0]:+.2f},{local_z[1]:+.2f},{local_z[2]:+.2f})"
        )

    # ---- callbacks ----------------------------------------------------------
    def on_waypoints(self, msg: PoseArray):
        # world 프레임 웨이포인트를 base_link 프레임으로 변환
        self.current_waypoints = []
        for p in msg.poses:
            bp = copy.deepcopy(p)
            bp.position.x -= ROBOT_ORIGIN[0]
            bp.position.y -= ROBOT_ORIGIN[1]
            bp.position.z -= ROBOT_ORIGIN[2]
            self.current_waypoints.append(bp)
        self.get_logger().info(
            f"{len(self.current_waypoints)}개 웨이포인트 수신 "
            f"(base_link 변환: 첫 점 x={self.current_waypoints[0].position.x:.2f} "
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
        self.get_logger().info("1단계: 첫 점까지 approach (plan_only)")
        self.approach_to_first()

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

        self.scene_pub.publish(ps)
        if not self.scene_initialized:
            self.get_logger().info(
                "PlanningScene: 물체 + 토치(attached to tool0, 25cm) 퍼블리시")
            self.scene_initialized = True

    # ---- 궤적 직접 실행 (joint_command 퍼블리시) --------------------------------
    def execute_trajectory_direct(self, traj):
        """궤적 포인트를 시간에 맞춰 /joint_command 로 퍼블리시."""
        points = traj.joint_trajectory.points
        names = list(traj.joint_trajectory.joint_names)
        if not points:
            self.get_logger().warn("빈 궤적")
            self.executing = False
            return

        self.executing = True
        self.get_logger().info(f"궤적 실행 시작: {len(points)} 포인트")

        def _run():
            import time
            prev_time = 0.0
            for i, pt in enumerate(points):
                t_sec = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
                dt = t_sec - prev_time
                if dt > 0:
                    time.sleep(dt)
                prev_time = t_sec

                cmd = JointState()
                cmd.name = names
                cmd.position = list(pt.positions)
                if pt.velocities:
                    cmd.velocity = list(pt.velocities)
                self.joint_cmd_pub.publish(cmd)

            self.get_logger().info(">>> 궤적 실행 완료")
            self.executing = False

        threading.Thread(target=_run, daemon=True).start()

    # ---- Stage 1: approach (IK + 직접 이동) -----------------------------------
    def approach_to_first(self):
        """첫 웨이포인트에서 x-5cm 위치로 IK 계산 후 직접 joint command 이동."""
        if not self.ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/compute_ik 서비스 없음")
            return

        first = self.current_waypoints[0]
        approach = copy.deepcopy(first)
        # 표면에서 normal 방향으로 5cm 뒤로
        _, n = get_surface_plane(get_target(self.cfg, self.active_target_name))
        approach.position.x += 0.05 * float(n[0])
        approach.position.y += 0.05 * float(n[1])
        approach.position.z += 0.05 * float(n[2])
        # brush_tip → tool0 오프셋
        approach = self._brush_tip_to_tool0(approach)

        req = GetPositionIK.Request()
        req.ik_request.group_name = PLANNING_GROUP
        ps = PoseStamped()
        ps.header.frame_id = BASE_FRAME
        ps.pose = approach
        req.ik_request.pose_stamped = ps

        if self.current_joint_state:
            rs = RobotState()
            rs.joint_state = self.current_joint_state
            req.ik_request.robot_state = rs

        fut = self.ik_client.call_async(req)
        fut.add_done_callback(self._approach_ik_done)

    def _approach_ik_done(self, future):
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(f"IK 서비스 실패: {e}")
            return

        if resp.error_code.val != 1:
            self.get_logger().error(f"approach IK 실패 error_code={resp.error_code.val}")
            return

        target_positions = list(resp.solution.joint_state.position[:6])
        target_names = list(resp.solution.joint_state.name[:6])
        self.get_logger().info(f"approach IK 성공 -> 직접 이동")

        # 현재 → 목표를 보간해서 부드럽게 이동 (50 스텝, 2초)
        self.executing = True
        def _move():
            import time
            import numpy as np
            if self.current_joint_state is None:
                self.executing = False
                return
            current = list(self.current_joint_state.position[:6])
            steps = 50
            for i in range(1, steps + 1):
                alpha = i / steps
                interp = [c + alpha * (t - c) for c, t in zip(current, target_positions)]
                cmd = JointState()
                cmd.name = target_names
                cmd.position = interp
                self.joint_cmd_pub.publish(cmd)
                time.sleep(2.0 / steps)
            self.get_logger().info("approach 이동 완료 -> joint_state 안정화 대기")
            self.executing = False
            time.sleep(1.0)  # joint_state 업데이트 대기
            js = self.current_joint_state
            if js:
                self.get_logger().info(
                    f"현재 관절: {[f'{p:.3f}' for p in js.position[:6]]}")
            self.plan_cartesian()
        threading.Thread(target=_move, daemon=True).start()

    # ---- torch_tip → tool0 오프셋 변환 ------------------------------------------
    @staticmethod
    def _brush_tip_to_tool0(pose):
        """torch_tip 기준 좌표를 tool0 기준으로 변환.
        토치는 URDF tool0 의 로컬 +Y 축으로 뻗음 (수동 캘리브레이션으로 확정).
        함수명은 내부 호환을 위해 유지."""
        q = pose.orientation
        x, y, z, w = q.x, q.y, q.z, q.w
        # 쿼터니언의 로컬 +Y 축 방향 (world 기준)
        ly = np.array([2*(x*y - z*w), 1 - 2*(x*x + z*z), 2*(y*z + x*w)])
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        new_pos = pos - TORCH_LENGTH * ly
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

    # ---- Stage 2: Cartesian path 계획 + 실행 --------------------------------
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

        # torch_tip → tool0 오프셋 적용
        tool0_wps = [self._brush_tip_to_tool0(wp) for wp in densified]

        # [TORCH_CHECK] 첫 waypoint 변환 검증
        _first_tip = densified[0]
        _first_tool0 = tool0_wps[0]
        _dist = np.linalg.norm([
            _first_tip.position.x - _first_tool0.position.x,
            _first_tip.position.y - _first_tool0.position.y,
            _first_tip.position.z - _first_tool0.position.z,
        ])
        self.get_logger().info(
            f"[TORCH_CHECK] torch_tip=({_first_tip.position.x:.3f},"
            f"{_first_tip.position.y:.3f},{_first_tip.position.z:.3f}) → "
            f"tool0=({_first_tool0.position.x:.3f},{_first_tool0.position.y:.3f},"
            f"{_first_tool0.position.z:.3f}) 거리={_dist*100:.1f}cm "
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

        req.waypoints = tool0_wps
        req.max_step = 0.005  # 5mm
        req.jump_threshold = 0.0  # 점프 제한 없음
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3

        # ---- OrientationConstraint: FIXED_TOOL0_QUAT 자세 유지 ----
        # 붓 축(tool0 +Y) 중심 회전만 자유. 나머지는 tight.
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

        # ---- 디버그: brush_tip 의 tool0 +Y 방향 검증 ----
        first_tip = densified[0]
        q = first_tip.orientation
        tip_p = np.array([first_tip.position.x, first_tip.position.y, first_tip.position.z])
        local_y_in_world = np.array([
            2*(q.x*q.y - q.z*q.w),
            1 - 2*(q.x*q.x + q.z*q.z),
            2*(q.y*q.z + q.x*q.w),
        ])
        self.get_logger().info(
            f"[CHECK] brush_tip=({tip_p[0]:.3f},{tip_p[1]:.3f},{tip_p[2]:.3f}) | "
            f"tool0 local+Y in world=({local_y_in_world[0]:+.2f},"
            f"{local_y_in_world[1]:+.2f},{local_y_in_world[2]:+.2f}) "
            f"[기대: -normal=({brush_dir_world[0]:+.2f},"
            f"{brush_dir_world[1]:+.2f},{brush_dir_world[2]:+.2f})]")
        self.get_logger().info(
            f"Cartesian 요청: {len(tool0_wps)} wp, 첫 tool0=("
            f"{tool0_wps[0].position.x:.3f},{tool0_wps[0].position.y:.3f},"
            f"{tool0_wps[0].position.z:.3f}) | "
            f"ori constraint free_axis={['x','y','z'][free_axis]} tol={tol}")

        fut = self.cartesian_client.call_async(req)
        fut.add_done_callback(self._cartesian_done)

    def _cartesian_done(self, future):
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(f"Cartesian 서비스 실패: {e}")
            return

        fraction = resp.fraction
        self.get_logger().info(f"Cartesian path: {fraction*100:.1f}% 달성")
        if fraction < 0.5:
            self.get_logger().error(
                "50% 미만 -> 스케치를 재생성하거나 타겟을 변경하세요.")
            return
        if fraction < 0.95:
            self.get_logger().warn(f"{fraction*100:.1f}% -> 부분 실행")

        traj = self._rescale_trajectory(resp.solution, scale=0.3)
        self.execute_trajectory_direct(traj)

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
