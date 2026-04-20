"""
스케치 UI - 벽에 글씨 쓰기
캔버스 = 벽 면, 마우스로 그리면 UR10이 벽에 글씨를 씁니다
"""
import tkinter as tk
import numpy as np
import json
import os

WAYPOINT_FILE = "/tmp/sketch_waypoints.json"

class SketchUI:
    def __init__(self):
        self.points = []
        self.is_drawing = False

        # 벽 매핑: 캔버스 → 벽 좌표
        # 로봇이 테이블(z=0.75) 위에 있고 벽을 향해 팔을 뻗음
        # 벽 실제 범위: z=0.0~1.5, y=-0.5~0.5
        # 안전 마진 적용한 글씨 영역
        self.wall = {
            'y_min': -0.35, 'y_max': 0.35,   # 좌우 (벽 중앙 기준)
            'z_min': 0.15, 'z_max': 1.35,    # 아래~위 (벽 z=0~1.5 내부, 마진 확보)
            'x': 0.63,                         # EE 도달 x
        }
        self.canvas_w = 500
        self.canvas_h = 500

        self.root = tk.Tk()
        self.root.title('벽에 글씨 쓰기 - UR10')
        self.root.configure(bg='#2b2b2b')

        # 제목
        tk.Label(self.root, text='🖊 벽에 글씨를 쓰세요', font=('Arial', 16, 'bold'),
                 bg='#2b2b2b', fg='white').pack(pady=(10, 5))

        # 캔버스 = 벽
        frame = tk.Frame(self.root, bg='#2b2b2b')
        frame.pack(pady=5)
        self.canvas = tk.Canvas(frame, width=self.canvas_w, height=self.canvas_h,
                                bg='#f5f0e8', relief='sunken', bd=3)
        self.canvas.pack()

        # 벽 질감 표현 (벽돌 라인)
        for i in range(0, self.canvas_h, 40):
            self.canvas.create_line(0, i, self.canvas_w, i, fill='#e0d8c8', width=1)
            offset = 0 if (i // 40) % 2 == 0 else 60
            for j in range(offset, self.canvas_w, 120):
                self.canvas.create_line(j, i, j, i + 40, fill='#e0d8c8', width=1)

        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)

        # 버튼
        btn_frame = tk.Frame(self.root, bg='#2b2b2b')
        btn_frame.pack(pady=8)
        tk.Button(btn_frame, text='✏ 쓰기 실행', command=self.on_execute,
                  bg='#4CAF50', fg='white', font=('Arial', 13, 'bold'),
                  width=12, cursor='hand2').pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text='🧹 지우기', command=self.on_clear,
                  bg='#757575', fg='white', font=('Arial', 13),
                  width=12, cursor='hand2').pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self.root, text="벽에 글씨를 그려주세요",
                                     font=('Arial', 12), bg='#2b2b2b', fg='#aaa')
        self.status_label.pack(pady=5)

        self.check_status()
        self.root.mainloop()

    def pixel_to_wall(self, px, py):
        """캔버스 픽셀 → 벽 3D 좌표
        캔버스 가로(px) → 벽 가로(y), 캔버스 세로(py) → 벽 높이(z)"""
        w = self.wall
        # 캔버스 왼쪽→오른쪽 = 벽 y_max→y_min (거울 보는 느낌 방지)
        y = w['y_min'] + (px / self.canvas_w) * (w['y_max'] - w['y_min'])
        # 캔버스 위→아래 = 벽 z_max→z_min
        z = w['z_max'] - (py / self.canvas_h) * (w['z_max'] - w['z_min'])
        return [w['x'], y, z]

    def on_press(self, event):
        self.is_drawing = True
        self.points = []
        self.canvas.delete('sketch')

    def on_drag(self, event):
        if not self.is_drawing:
            return
        self.points.append((event.x, event.y))
        if len(self.points) > 1:
            x1, y1 = self.points[-2]
            x2, y2 = self.points[-1]
            self.canvas.create_line(x1, y1, x2, y2, fill='#333', width=4, tags='sketch',
                                    capstyle=tk.ROUND, joinstyle=tk.ROUND)

    def on_release(self, event):
        self.is_drawing = False

    def on_execute(self):
        if len(self.points) < 2:
            self.status_label.config(text="먼저 글씨를 써주세요!", fg='#ff6666')
            return

        indices = np.linspace(0, len(self.points) - 1, min(30, len(self.points)), dtype=int)
        sampled = [self.points[i] for i in indices]
        waypoints = [self.pixel_to_wall(px, py) for px, py in sampled]

        data = {"waypoints": waypoints, "execute": True}
        with open(WAYPOINT_FILE, 'w') as f:
            json.dump(data, f)

        self.status_label.config(text=f"✏ {len(waypoints)}개 포인트 전송! 로봇이 쓰는 중...",
                                 fg='#4CAF50')

    def on_clear(self):
        self.canvas.delete('sketch')
        # 벽돌 다시 그리기
        for i in range(0, self.canvas_h, 40):
            self.canvas.create_line(0, i, self.canvas_w, i, fill='#e0d8c8', width=1)
            offset = 0 if (i // 40) % 2 == 0 else 60
            for j in range(offset, self.canvas_w, 120):
                self.canvas.create_line(j, i, j, i + 40, fill='#e0d8c8', width=1)
        self.points = []
        self.status_label.config(text="벽에 글씨를 그려주세요", fg='#aaa')
        if os.path.exists(WAYPOINT_FILE):
            os.remove(WAYPOINT_FILE)

    def check_status(self):
        done_file = "/tmp/sketch_done.flag"
        if os.path.exists(done_file):
            self.status_label.config(text="✅ 쓰기 완료!", fg='#4CAF50')
            os.remove(done_file)
        self.root.after(500, self.check_status)


if __name__ == '__main__':
    SketchUI()
