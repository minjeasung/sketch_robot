// sketch_robot UI — rosbridge 연결 + sketch-guided target/work-area/path.
// 현재 구현은 Three.js 가 아니라 native canvas overlay 를 사용한다.

const WS_URL = "ws://localhost:9090";
const WORK_AREA_TOPIC = "/perception/work_area_plane";
const ZED_LEFT_IMAGE_TOPIC = "/zed/zed_node/rgb/color/rect/image";

const $ = (id) => document.getElementById(id);

function setStatus(state, text) {
  const node = $("status");
  node.classList.remove("connecting", "connected", "disconnected", "error");
  node.classList.add(state);
  $("status-text").textContent = text;
}

function logEvent(line) {
  const node = $("events");
  const ts = new Date().toISOString().slice(11, 23); // HH:MM:SS.sss
  node.textContent += `[${ts}] ${line}\n`;
  node.scrollTop = node.scrollHeight;
}

// ---- ROS 연결 ----
const ros = new ROSLIB.Ros({ url: WS_URL });

ros.on("connection", () => {
  setStatus("connected", "connected");
  logEvent("connection opened");
});

ros.on("error", (err) => {
  setStatus("error", "error");
  logEvent(`error: ${err && err.message ? err.message : err}`);
});

ros.on("close", () => {
  setStatus("disconnected", "disconnected");
  logEvent("connection closed");
});

setStatus("connecting", "connecting…");
logEvent(`connecting to ${WS_URL}`);

// ---- /perception/work_area_plane 구독 ----
const workAreaPlane = new ROSLIB.Topic({
  ros: ros,
  name: WORK_AREA_TOPIC,
  messageType: "geometry_msgs/PoseStamped",
});

let msgCount = 0;

workAreaPlane.subscribe((msg) => {
  msgCount += 1;
  const stamp = msg.header.stamp;
  const stampStr = `${stamp.sec}.${String(stamp.nanosec).padStart(9, "0")}`;
  const p = msg.pose.position;
  const o = msg.pose.orientation;

  $("frame-id").textContent = msg.header.frame_id || "—";
  $("stamp").textContent = stampStr;
  $("position").textContent =
    `(${p.x.toFixed(4)}, ${p.y.toFixed(4)}, ${p.z.toFixed(4)})`;
  $("orientation").textContent =
    `(${o.x.toFixed(4)}, ${o.y.toFixed(4)}, ${o.z.toFixed(4)}, ${o.w.toFixed(4)})`;
  $("msg-count").textContent = String(msgCount);

  if (msgCount === 1) {
    logEvent(`first ${WORK_AREA_TOPIC} message received`);
  }
});

logEvent(`subscribed to ${WORK_AREA_TOPIC}`);

// ---- View mode (ZED Raw / Wall Front) + image subscriber ----
// 두 view 의 sketch strokes 는 의미가 다름 (원본 카메라 픽셀 vs 벽 평면 픽셀) → 분리 보관.
const VIEW_TOPICS = {
  zed_raw:    "/zed/zed_node/rgb/color/rect/image",
  wall_front: "/perception/wall_front_view",
};
const VIEW_TITLES = {
  zed_raw:    "ZED LEFT CAMERA",
  wall_front: "WALL FRONT VIEW (벽 정면)",
};

let currentView = "zed_raw";
let currentImageSub = null;

const zedCanvas = $("zed-canvas");
const zedCtx = zedCanvas.getContext("2d");
let zedFrameCount = 0;
let zedFpsFrames = 0;
let zedFpsTimerStart = performance.now();

function decodeImageData(msg) {
  // sensor_msgs/Image, encoding=rgb8 → roslibjs 가 base64 string 으로 data 전달.
  const w = msg.width, h = msg.height;
  const bin = atob(msg.data);
  // step (한 row 의 바이트 수) — rgb8 면 w*3. 다른 encoding 대비 일반화.
  const channels = (msg.encoding === "rgba8" || msg.encoding === "bgra8") ? 4 : 3;
  const buf = new Uint8ClampedArray(w * h * 4);
  let j = 0;
  if (msg.encoding === "rgb8") {
    for (let i = 0; i < bin.length; i += 3) {
      buf[j++] = bin.charCodeAt(i);
      buf[j++] = bin.charCodeAt(i + 1);
      buf[j++] = bin.charCodeAt(i + 2);
      buf[j++] = 255;
    }
  } else if (msg.encoding === "bgr8") {
    for (let i = 0; i < bin.length; i += 3) {
      buf[j++] = bin.charCodeAt(i + 2);
      buf[j++] = bin.charCodeAt(i + 1);
      buf[j++] = bin.charCodeAt(i);
      buf[j++] = 255;
    }
  } else if (msg.encoding === "bgra8") {
    for (let i = 0; i < bin.length; i += 4) {
      buf[j++] = bin.charCodeAt(i + 2);
      buf[j++] = bin.charCodeAt(i + 1);
      buf[j++] = bin.charCodeAt(i);
      buf[j++] = 255;
    }
  } else if (msg.encoding === "rgba8") {
    for (let i = 0; i < bin.length; i += 4) {
      buf[j++] = bin.charCodeAt(i);
      buf[j++] = bin.charCodeAt(i + 1);
      buf[j++] = bin.charCodeAt(i + 2);
      buf[j++] = bin.charCodeAt(i + 3);
    }
  } else {
    // unknown encoding — gray fallback
    for (let i = 0; i < bin.length; i += channels) {
      const v = bin.charCodeAt(i);
      buf[j++] = v; buf[j++] = v; buf[j++] = v; buf[j++] = 255;
    }
  }
  return new ImageData(buf, w, h);
}

function handleImageMsg(msg) {
  if (msg.width !== zedCanvas.width || msg.height !== zedCanvas.height) {
    zedCanvas.width = msg.width;
    zedCanvas.height = msg.height;
    if (sketchCanvas.width !== msg.width || sketchCanvas.height !== msg.height) {
      sketchCanvas.width = msg.width;
      sketchCanvas.height = msg.height;
      redrawSketch();
    }
  }
  try {
    const imgData = decodeImageData(msg);
    zedCtx.putImageData(imgData, 0, 0);
  } catch (e) {
    logEvent(`image decode 실패: ${e.message || e}`);
    return;
  }

  zedFrameCount += 1;
  zedFpsFrames += 1;
  const now = performance.now();
  const elapsed = now - zedFpsTimerStart;
  if (elapsed >= 1000) {
    const fps = (zedFpsFrames * 1000 / elapsed).toFixed(1);
    $("zed-fps").textContent = `${fps} Hz`;
    zedFpsFrames = 0;
    zedFpsTimerStart = now;
  }

  $("zed-res").textContent = `${msg.width} × ${msg.height}`;
  $("zed-encoding").textContent = msg.encoding;
  $("zed-frames").textContent = String(zedFrameCount);

  if (zedFrameCount === 1) {
    logEvent(`first image on ${VIEW_TOPICS[currentView]} (${msg.width}×${msg.height}, ${msg.encoding})`);
  }
}

function subscribeView(viewName) {
  if (currentImageSub) {
    try { currentImageSub.unsubscribe(); } catch (_) {}
    currentImageSub = null;
  }
  const topic = VIEW_TOPICS[viewName];
  const sub = new ROSLIB.Topic({
    ros: ros,
    name: topic,
    messageType: "sensor_msgs/Image",
    throttle_rate: 0,
    queue_size: 1,
  });
  sub.subscribe(handleImageMsg);
  currentImageSub = sub;
  // 새 view 의 첫 frame 도착 전 — stats reset 으로 fps 계산 정확하게.
  zedFrameCount = 0;
  zedFpsFrames = 0;
  zedFpsTimerStart = performance.now();
  $("zed-frames").textContent = "0";
  $("zed-fps").textContent = "—";
  $("zed-res").textContent = "—";
  $("zed-encoding").textContent = "—";
  logEvent(`subscribed to ${topic}`);
}

function switchView(viewName) {
  if (viewName === currentView || !VIEW_TOPICS[viewName]) return;
  // 진행 중 stroke 정리 (모드 무관, 다른 view 로 가면 의미 없음)
  currentStroke = null;
  pendingLine = null;
  currentMouse = null;
  currentView = viewName;
  // 헤더 갱신 — 첫 text node 만 교체, span#view-card-topic 보존
  const titleEl = $("view-card-title");
  titleEl.firstChild.nodeValue = VIEW_TITLES[viewName] + " ";
  $("view-card-topic").textContent =
    `(${VIEW_TOPICS[viewName]}, sensor_msgs/Image)`;
  $("view-mode-text").textContent = viewName;
  // sketch 즉시 redraw (새 view 의 strokes 로)
  redrawSketch();
  // 새 topic subscribe
  subscribeView(viewName);
}

document.querySelectorAll('input[name="view-mode"]').forEach((r) => {
  r.addEventListener("change", () => {
    const v = document.querySelector('input[name="view-mode"]:checked').value;
    switchView(v);
  });
});

// 초기 구독
subscribeView(currentView);


// ============================================================================
// Sketch overlay (Freehand + Line 모드) — canvas native 좌표 (px) 기준 보관.
// 3D 변환은 B3.3 에서. 여기는 시각 + 데이터 저장만.
// ============================================================================
const sketchCanvas = $("sketch-canvas");
const sketchCtx = sketchCanvas.getContext("2d");

// workflow 별 stroke 분리. Target/Work Area/Path 의 의미가 다르므로 합치지 않는다.
const strokesMap = { target: [], work_area: [], path: [] };
let workflowMode = "target";
function currentStrokes() { return strokesMap[workflowMode]; }

let currentStroke = null;     // 진행 중 freehand stroke (mousedown ~ mouseup)
let pendingLine = null;       // Line 모드: 첫 점 찍힌 후 두 번째 클릭 대기 중인 stroke
let currentMouse = null;      // 가장 최근 pointer 위치 (Line preview 용)
let sketchMode = "freehand";

const STROKE_COLOR = "#22d3ee";  // cyan
const STROKE_WIDTH = 3;

function getNativeCoords(ev) {
  // CSS 로 줄어든 canvas 의 client 좌표 → native (canvas.width/height) 좌표
  const rect = sketchCanvas.getBoundingClientRect();
  const scaleX = sketchCanvas.width / rect.width;
  const scaleY = sketchCanvas.height / rect.height;
  return {
    u: (ev.clientX - rect.left) * scaleX,
    v: (ev.clientY - rect.top) * scaleY,
  };
}

function updateSketchStats() {
  const cs = currentStrokes();
  $("sketch-strokes-count").textContent = String(cs.length);
  const total = cs.reduce((acc, s) => acc + s.points.length, 0);
  $("sketch-points-count").textContent = String(total);
  $("btn-set-target").disabled = workflowMode !== "target" || cs.length === 0 || currentView !== "zed_raw";
  $("btn-set-work-area").disabled = workflowMode !== "work_area" || cs.length === 0 || currentView !== "zed_raw";
  $("btn-execute").disabled = workflowMode !== "path" || cs.length === 0 || currentView !== "wall_front";
}

function redrawSketch() {
  sketchCtx.clearRect(0, 0, sketchCanvas.width, sketchCanvas.height);
  sketchCtx.lineCap = "round";
  sketchCtx.lineJoin = "round";
  sketchCtx.lineWidth = STROKE_WIDTH;
  sketchCtx.strokeStyle = STROKE_COLOR;

  for (const s of currentStrokes()) {
    if (s.points.length === 0) continue;
    sketchCtx.beginPath();
    sketchCtx.moveTo(s.points[0].u, s.points[0].v);
    if (s.type === "freehand") {
      for (let i = 1; i < s.points.length; i++) {
        sketchCtx.lineTo(s.points[i].u, s.points[i].v);
      }
    } else if (s.type === "line" && s.points.length >= 2) {
      sketchCtx.lineTo(s.points[1].u, s.points[1].v);
    }
    sketchCtx.stroke();
  }

  // Line 모드 preview (첫 점 찍힌 후, 두 번째 클릭 전)
  if (pendingLine && currentMouse) {
    sketchCtx.save();
    sketchCtx.setLineDash([8, 6]);
    sketchCtx.strokeStyle = "rgba(34, 211, 238, 0.65)";
    sketchCtx.beginPath();
    sketchCtx.moveTo(pendingLine.points[0].u, pendingLine.points[0].v);
    sketchCtx.lineTo(currentMouse.u, currentMouse.v);
    sketchCtx.stroke();
    sketchCtx.restore();
    // 첫 점 marker (원)
    sketchCtx.fillStyle = STROKE_COLOR;
    sketchCtx.beginPath();
    sketchCtx.arc(pendingLine.points[0].u, pendingLine.points[0].v,
                  5, 0, Math.PI * 2);
    sketchCtx.fill();
  }

  updateSketchStats();
}

// ---- Pointer event handlers (mouse + touch 통합) ----
sketchCanvas.addEventListener("pointerdown", (ev) => {
  ev.preventDefault();
  sketchCanvas.setPointerCapture(ev.pointerId);
  const c = getNativeCoords(ev);
  currentMouse = c;
  if (sketchMode === "freehand") {
    currentStroke = { type: "freehand", points: [c] };
    currentStrokes().push(currentStroke);
  } else {
    // line
    if (!pendingLine) {
      pendingLine = { type: "line", points: [c] };
    } else {
      pendingLine.points.push(c);
      currentStrokes().push(pendingLine);
      pendingLine = null;
    }
  }
  redrawSketch();
});

sketchCanvas.addEventListener("pointermove", (ev) => {
  currentMouse = getNativeCoords(ev);
  if (sketchMode === "freehand" && currentStroke) {
    currentStroke.points.push(currentMouse);
    redrawSketch();
  } else if (sketchMode === "line" && pendingLine) {
    redrawSketch();
  }
});

function finishFreehand(ev) {
  if (currentStroke) {
    currentStroke = null;
    redrawSketch();
  }
  try { sketchCanvas.releasePointerCapture(ev.pointerId); } catch (_) {}
}

sketchCanvas.addEventListener("pointerup", finishFreehand);
sketchCanvas.addEventListener("pointercancel", finishFreehand);

sketchCanvas.addEventListener("pointerleave", () => {
  currentMouse = null;
  if (pendingLine) redrawSketch();   // preview 사라짐
});

// ESC: 진행 중 line 취소
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && pendingLine) {
    pendingLine = null;
    redrawSketch();
  }
});

// ---- 모드 라디오 ----
document.querySelectorAll('input[name="sketch-mode"]').forEach((r) => {
  r.addEventListener("change", () => {
    sketchMode = document.querySelector('input[name="sketch-mode"]:checked').value;
    $("sketch-mode-text").textContent = sketchMode;
    // 모드 전환 시 진행 중 stroke 정리
    if (sketchMode !== "line" && pendingLine) {
      pendingLine = null;
    }
    if (sketchMode !== "freehand" && currentStroke) {
      currentStroke = null;
    }
    redrawSketch();
  });
});

function switchWorkflow(mode) {
  if (!strokesMap[mode]) return;
  currentStroke = null;
  pendingLine = null;
  currentMouse = null;
  workflowMode = mode;
  $("workflow-mode-text").textContent = mode;
  const targetView = mode === "path" ? "wall_front" : "zed_raw";
  const radio = document.querySelector(`input[name="view-mode"][value="${targetView}"]`);
  if (radio) radio.checked = true;
  switchView(targetView);
  redrawSketch();
}

document.querySelectorAll('input[name="workflow-mode"]').forEach((r) => {
  r.addEventListener("change", () => {
    const mode = document.querySelector('input[name="workflow-mode"]:checked').value;
    switchWorkflow(mode);
  });
});

// ---- Clear / Undo (현재 view 의 strokes 만) ----
$("btn-clear").addEventListener("click", () => {
  currentStrokes().length = 0;
  pendingLine = null;
  currentStroke = null;
  redrawSketch();
});

$("btn-undo").addEventListener("click", () => {
  const cs = currentStrokes();
  if (cs.length > 0) {
    cs.pop();
    redrawSketch();
  }
});

// ---- Publish workflow strokes as PoseArray ---------------------------------
const TARGET_SELECTION_TOPIC = "/target_selection_pixels";
const WORK_AREA_PIXELS_TOPIC = "/work_area_pixels";
const REFINE_WORK_AREA_TOPIC = "/refine_work_area";
const WORK_AREA_REFINE_STATUS_TOPIC = "/work_area_refine_status";
const SKETCH_PIXELS_TOPIC = "/sketch_pixels";
const SKETCH_EXECUTE_TOPIC = "/sketch_execute";
const targetSelectionPub = new ROSLIB.Topic({
  ros: ros,
  name: TARGET_SELECTION_TOPIC,
  messageType: "geometry_msgs/PoseArray",
});
const workAreaPub = new ROSLIB.Topic({
  ros: ros,
  name: WORK_AREA_PIXELS_TOPIC,
  messageType: "geometry_msgs/PoseArray",
});
const refineWorkAreaPub = new ROSLIB.Topic({
  ros: ros,
  name: REFINE_WORK_AREA_TOPIC,
  messageType: "std_msgs/Bool",
});
const workAreaRefineStatusSub = new ROSLIB.Topic({
  ros: ros,
  name: WORK_AREA_REFINE_STATUS_TOPIC,
  messageType: "std_msgs/String",
});
const sketchPub = new ROSLIB.Topic({
  ros: ros,
  name: SKETCH_PIXELS_TOPIC,
  messageType: "geometry_msgs/PoseArray",
});
const sketchExecutePub = new ROSLIB.Topic({
  ros: ros,
  name: SKETCH_EXECUTE_TOPIC,
  messageType: "std_msgs/Bool",
});
let hasExecuted = false;  // Execute 한 번 이상 → Run Robot 활성화
let waitingWorkAreaRefine = false;

workAreaRefineStatusSub.subscribe((msg) => {
  let payload = {};
  try {
    payload = JSON.parse(msg.data || "{}");
  } catch (_) {
    payload = { state: msg.data || "unknown" };
  }
  const state = payload.state || "unknown";
  logEvent(`work area refine: ${state}`);
  if (!waitingWorkAreaRefine) return;

  if (state === "done") {
    waitingWorkAreaRefine = false;
    const radio = document.querySelector('input[name="workflow-mode"][value="path"]');
    if (radio) radio.checked = true;
    switchWorkflow("path");
  } else if (state === "failed" || state === "timeout" || state === "busy") {
    waitingWorkAreaRefine = false;
    logEvent("D405 refine failed; staying in Work Area view");
  }
});

function nowRosTime() {
  const ms = Date.now();
  return {
    sec: Math.floor(ms / 1000),
    nanosec: (ms % 1000) * 1_000_000,
  };
}

function posesFromStrokes(strokes) {
  const poses = [];
  for (const s of strokes) {
    for (const pt of s.points) {
      poses.push({
        position: { x: pt.u, y: pt.v, z: 0.0 },
        orientation: { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
      });
    }
  }
  return poses;
}

function publishPixels(pub, topicName, frameId, strokes) {
  const poses = posesFromStrokes(strokes);
  if (poses.length === 0) return false;
  const msg = new ROSLIB.Message({
    header: {
      stamp: nowRosTime(),
      frame_id: frameId,
    },
    poses: poses,
  });
  pub.publish(msg);
  logEvent(`published ${poses.length} points to ${topicName} (frame=${frameId})`);
  return true;
}

$("btn-set-target").addEventListener("click", () => {
  if (workflowMode !== "target" || currentView !== "zed_raw") return;
  if (publishPixels(targetSelectionPub, TARGET_SELECTION_TOPIC, "zed_raw", currentStrokes())) {
    const radio = document.querySelector('input[name="workflow-mode"][value="work_area"]');
    if (radio) radio.checked = true;
    switchWorkflow("work_area");
  }
});

$("btn-set-work-area").addEventListener("click", () => {
  if (workflowMode !== "work_area" || currentView !== "zed_raw") return;
  if (publishPixels(workAreaPub, WORK_AREA_PIXELS_TOPIC, "zed_raw", currentStrokes())) {
    waitingWorkAreaRefine = true;
    refineWorkAreaPub.publish(new ROSLIB.Message({ data: true }));
    logEvent(`published refine request to ${REFINE_WORK_AREA_TOPIC}`);
    logEvent("waiting for D405 work-area refinement before Path mode");
  }
});

$("btn-execute").addEventListener("click", () => {
  const cs = currentStrokes();
  if (cs.length === 0) return;
  if (workflowMode !== "path" || currentView !== "wall_front") {
    logEvent("Execute ignored: switch to Wall Front first");
    return;
  }

  if (!publishPixels(sketchPub, SKETCH_PIXELS_TOPIC, "wall_front", cs)) return;

  // 첫 publish 후부터 Run Robot 활성화
  if (!hasExecuted) {
    hasExecuted = true;
    const runBtn = $("btn-run-robot");
    if (runBtn) runBtn.disabled = false;
  }
});

// ---- Run Robot: confirm 후 /sketch_execute Bool(true) publish ----
$("btn-run-robot").addEventListener("click", () => {
  if (!hasExecuted) return;
  const ok = window.confirm("정말 실행? RB10 이 움직입니다.");
  if (!ok) {
    logEvent("Run Robot cancelled by user");
    return;
  }
  sketchExecutePub.publish(new ROSLIB.Message({ data: true }));
  logEvent(`published ${SKETCH_EXECUTE_TOPIC} (RB10 will move)`);
});

updateSketchStats();
logEvent("sketch overlay ready (mode=freehand)");
