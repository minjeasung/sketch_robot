"""
관절 캘리브레이션 UI - 6개 슬라이더로 RB10 관절을 실시간 조작.
Isaac Sim 의 /joint_command 를 통해 로봇 이동.
현재 관절 값 + tool0 TF 를 실시간 표시.
"""
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
import tkinter as tk


JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]

# 각 관절의 기본값 (라디안) — Isaac Sim 의 초기 "ready" 자세와 동일
DEFAULT_POS = [0.0005, -0.9343, 2.4246, -1.6293, 1.5675, 0.0]

# 관절 범위
JOINT_MIN = -np.pi
JOINT_MAX = np.pi


class JointCalibrator(Node):
    def __init__(self):
        super().__init__("joint_calibrator")
        self.pub = self.create_publisher(JointState, "/joint_command", 10)
        self.create_subscription(JointState, "/joint_states",
                                 self.on_joint_state, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.current_js = None
        self.get_logger().info("Joint Calibrator 시작")

    def on_joint_state(self, msg):
        self.current_js = msg

    def publish_cmd(self, positions):
        msg = JointState()
        msg.name = JOINT_NAMES
        msg.position = list(positions)
        self.pub.publish(msg)

    def get_tool0_tf(self):
        for parent in ("world", "World"):
            try:
                tf = self.tf_buffer.lookup_transform(
                    parent, "tool0", rclpy.time.Time(),
                    timeout=Duration(seconds=0.2))
                return tf
            except Exception:
                continue
        return None


BRUSH_LENGTH = 0.15
WALL_FRONT_X = 0.80  # 현재 base 좌표계의 벽 앞면


def run_gui(node: JointCalibrator):
    root = tk.Tk()
    root.title("RB10 Joint Calibrator")
    root.geometry("600x820")

    sliders = []
    value_labels = []

    def on_change(_=None):
        pos = [s.get() for s in sliders]
        for i, v in enumerate(pos):
            value_labels[i].config(text=f"{v:+.3f} rad ({np.degrees(v):+.1f}°)")
        node.publish_cmd(pos)

    tk.Label(root, text="RB10 관절 슬라이더 (라디안)",
             font=("Arial", 14, "bold")).pack(pady=8)

    for i, name in enumerate(JOINT_NAMES):
        frame = tk.Frame(root)
        frame.pack(fill="x", padx=10, pady=4)
        tk.Label(frame, text=name, width=20, anchor="w").pack(side=tk.LEFT)
        s = tk.Scale(frame, from_=JOINT_MIN, to=JOINT_MAX, resolution=0.01,
                     orient=tk.HORIZONTAL, length=260,
                     showvalue=False, command=on_change)
        s.set(DEFAULT_POS[i])
        s.pack(side=tk.LEFT, padx=4)
        sliders.append(s)
        lbl = tk.Label(frame, text="", width=18, anchor="w")
        lbl.pack(side=tk.LEFT)
        value_labels.append(lbl)

    # 버튼들
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=8)

    def reset_default():
        for s, v in zip(sliders, DEFAULT_POS):
            s.set(v)
        on_change()

    def print_current():
        pos = [s.get() for s in sliders]
        print("=" * 60)
        print(f"현재 슬라이더 값 (라디안, 복사해서 알려주세요):")
        print(f"  DEFAULT_POS = {pos}")
        if node.current_js:
            js_pos = [node.current_js.position[
                node.current_js.name.index(n)]
                for n in JOINT_NAMES if n in node.current_js.name]
            print(f"  실제 /joint_states 값: {js_pos}")
        tf = node.get_tool0_tf()
        if tf:
            q = tf.transform.rotation
            p = tf.transform.translation
            x, y, z, w = q.x, q.y, q.z, q.w
            lx = (1 - 2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w))
            ly = (2*(x*y-z*w), 1 - 2*(x*x+z*z), 2*(y*z+x*w))
            lz = (2*(x*z+y*w), 2*(y*z-x*w), 1 - 2*(x*x+y*y))
            print(f"  tool0 pos=({p.x:+.3f},{p.y:+.3f},{p.z:+.3f})")
            print(f"  tool0 quat=({q.x:+.3f},{q.y:+.3f},{q.z:+.3f},{q.w:+.3f})")
            print(f"  local_X_in_world=({lx[0]:+.2f},{lx[1]:+.2f},{lx[2]:+.2f})")
            print(f"  local_Y_in_world=({ly[0]:+.2f},{ly[1]:+.2f},{ly[2]:+.2f})")
            print(f"  local_Z_in_world=({lz[0]:+.2f},{lz[1]:+.2f},{lz[2]:+.2f})")
        else:
            print("  tool0 TF 미수신")
        print("=" * 60)

    tk.Button(btn_frame, text="기본 자세로 리셋", command=reset_default,
              bg="#616161", fg="white", width=18).pack(side=tk.LEFT, padx=4)
    tk.Button(btn_frame, text="현재 값 출력", command=print_current,
              bg="#2e7d32", fg="white", width=18).pack(side=tk.LEFT, padx=4)

    # ---- 실시간 상태 표시 ----
    status = tk.Label(root, text="", justify="left", fg="#222",
                      font=("Courier", 10), bg="#f0f0f0", anchor="w")
    status.pack(fill="x", padx=10, pady=8)

    def update_status():
        tf = node.get_tool0_tf()
        if tf is None:
            status.config(text="tool0 TF 미수신 (Isaac Sim play 중인지 확인)")
        else:
            q = tf.transform.rotation
            p = tf.transform.translation
            x, y, z, w = q.x, q.y, q.z, q.w
            # tool0 +Z in world (= 붓 방향)
            lz = np.array([2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)])
            tip = np.array([p.x, p.y, p.z]) + BRUSH_LENGTH * lz
            # 붓 방향이 world +X 와 얼마나 일치하는지 (1.0 = 완벽)
            wall_alignment = float(lz[0])
            # 붓 끝의 벽까지 거리 (x 방향)
            dist_to_wall = float(tip[0] - WALL_FRONT_X)

            # 목표 판정
            ok_align = abs(wall_alignment - 1.0) < 0.1   # +X 와 거의 일치
            ok_pos = abs(dist_to_wall) < 0.05             # 5cm 이내
            status_icon = "✅" if (ok_align and ok_pos) else "⚠️"

            status.config(text=(
                f"{status_icon} 현재 상태\n"
                f"  tool0 pos  : ({p.x:+.3f}, {p.y:+.3f}, {p.z:+.3f})\n"
                f"  붓 방향 (tool0 +Z in world): ({lz[0]:+.2f}, {lz[1]:+.2f}, {lz[2]:+.2f})\n"
                f"  붓 끝 위치 : ({tip[0]:+.3f}, {tip[1]:+.3f}, {tip[2]:+.3f})\n"
                f"  벽 정렬    : {wall_alignment:+.2f}  (목표: +1.00 - 붓이 +X 향)\n"
                f"  벽까지 거리: {dist_to_wall:+.3f}m ({'붓 끝이 벽 안쪽' if dist_to_wall > 0 else '붓 끝이 벽 앞'})"
            ))
        root.after(300, update_status)

    tk.Label(root, text=(
        "목표: 붓 방향 (tool0 +Z) = (+1.00, 0.00, 0.00)\n"
        "      붓 끝이 벽 앞면 (x≈0.80) 근처에 있으면 ✅\n"
        "슬라이더 조정 → 실시간으로 Isaac Sim 로봇이 움직입니다.\n"
        "목표 달성 후 [현재 값 출력] 클릭 → 터미널 값 알려주세요."
    ), justify="left", fg="#555").pack(pady=6)

    on_change()
    update_status()
    root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = JointCalibrator()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    run_gui(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
