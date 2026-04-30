# Phase 3 — OMPL planner 활성화

## 배경

`ur_moveit_config/config/ompl_planning.yaml` 에 `planner_configs` 섹션이 빠져있어서
`query_planner_interface` 호출 시 `planner_ids=['ur_manipulator']` 만 나오고
RRTConnect 등 실제 planner 가 등록 안 됨.

해결: 비어있는 yaml 에 `ompl_defaults.yaml` 의 풀 planner config 와 `ur_manipulator`
group 의 planner 사용 선언을 합쳐서 추가.

## 작업 가이드

이 작업은 시스템 파일 (`/opt/ros/jazzy/share/ur_moveit_config/`) 을 수정합니다.
sudo 권한 필요. 백업 먼저.

### Step 1: 백업

```bash
sudo cp /opt/ros/jazzy/share/ur_moveit_config/config/ompl_planning.yaml \
        /opt/ros/jazzy/share/ur_moveit_config/config/ompl_planning.yaml.bak_phase3
ls -la /opt/ros/jazzy/share/ur_moveit_config/config/ompl_planning.yaml*
```

`.bak_phase3` 파일 보이는지 확인.

### Step 2: 새 yaml 작성

기존 파일을 다음 내용으로 완전히 교체. sudo 가 필요하므로 임시 파일을 작성한 후 sudo mv 로 이동.

```bash
cat > /tmp/ompl_planning_new.yaml << 'EOF'
planning_plugins:
  - ompl_interface/OMPLPlanner

# The order of the elements in the adapter corresponds to the order they are processed by the motion planning pipeline.
request_adapters:
  - default_planning_request_adapters/ResolveConstraintFrames
  - default_planning_request_adapters/ValidateWorkspaceBounds
  - default_planning_request_adapters/CheckStartStateBounds
  - default_planning_request_adapters/CheckStartStateCollision

response_adapters:
  - default_planning_response_adapters/AddTimeOptimalParameterization
  - default_planning_response_adapters/ValidateSolution
  - default_planning_response_adapters/DisplayMotionPath

# OMPL planner_configs - moveit_configs_utils/default_configs/ompl_defaults.yaml 에서 가져옴
planner_configs:
  AnytimePathShortening:
    type: geometric::AnytimePathShortening
    shortcut: 1
    hybridize: 1
    max_hybrid_paths: 24
    num_planners: 4
    planners: ""
  SBL:
    type: geometric::SBL
    range: 0.0
  EST:
    type: geometric::EST
    range: 0.0
    goal_bias: 0.05
  LBKPIECE:
    type: geometric::LBKPIECE
    range: 0.0
    border_fraction: 0.9
    min_valid_path_fraction: 0.5
  BKPIECE:
    type: geometric::BKPIECE
    range: 0.0
    border_fraction: 0.9
    failed_expansion_score_factor: 0.5
    min_valid_path_fraction: 0.5
  KPIECE:
    type: geometric::KPIECE
    range: 0.0
    goal_bias: 0.05
    border_fraction: 0.9
    failed_expansion_score_factor: 0.5
    min_valid_path_fraction: 0.5
  RRT:
    type: geometric::RRT
    range: 0.0
    goal_bias: 0.05
  RRTConnect:
    type: geometric::RRTConnect
    range: 0.0
  RRTstar:
    type: geometric::RRTstar
    range: 0.0
    goal_bias: 0.05
    delay_collision_checking: 1
  TRRT:
    type: geometric::TRRT
    range: 0.0
    goal_bias: 0.05
    max_states_failed: 10
    temp_change_factor: 2.0
    min_temperature: 10e-10
    init_temperature: 10e-6
    frountier_threshold: 0.0
    frountierNodeRatio: 0.1
    k_constant: 0.0
  PRM:
    type: geometric::PRM
    max_nearest_neighbors: 10
  PRMstar:
    type: geometric::PRMstar
  FMT:
    type: geometric::FMT
    num_samples: 1000
    radius_multiplier: 1.1
    nearest_k: 1
    cache_cc: 1
    heuristics: 0
    extended_fmt: 1
  BFMT:
    type: geometric::BFMT
    num_samples: 1000
    radius_multiplier: 1.0
    nearest_k: 1
    balanced: 0
    optimality: 1
    heuristics: 1
    cache_cc: 1
    extended_fmt: 1
  PDST:
    type: geometric::PDST
  STRIDE:
    type: geometric::STRIDE
    range: 0.0
    goal_bias: 0.05
    use_projected_distance: 0
    degree: 16
    max_degree: 18
    min_degree: 12
    max_pts_per_leaf: 6
    estimated_dimension: 0.0
    min_valid_path_fraction: 0.2
  BiTRRT:
    type: geometric::BiTRRT
    range: 0.0
    temp_change_factor: 0.1
    init_temperature: 100
    frountier_threshold: 0.0
    frountier_node_ratio: 0.1
    cost_threshold: 1e300
  LBTRRT:
    type: geometric::LBTRRT
    range: 0.0
    goal_bias: 0.05
    epsilon: 0.4
  BiEST:
    type: geometric::BiEST
    range: 0.0
  ProjEST:
    type: geometric::ProjEST
    range: 0.0
    goal_bias: 0.05
  LazyPRM:
    type: geometric::LazyPRM
    range: 0.0
  LazyPRMstar:
    type: geometric::LazyPRMstar
  SPARS:
    type: geometric::SPARS
    stretch_factor: 3.0
    sparse_delta_fraction: 0.25
    dense_delta_fraction: 0.001
    max_failures: 1000
  SPARStwo:
    type: geometric::SPARStwo
    stretch_factor: 3.0
    sparse_delta_fraction: 0.25
    dense_delta_fraction: 0.001
    max_failures: 5000

# UR group 의 planner 사용 선언
ur_manipulator:
  default_planner_config: RRTConnect
  planner_configs:
    - AnytimePathShortening
    - SBL
    - EST
    - LBKPIECE
    - BKPIECE
    - KPIECE
    - RRT
    - RRTConnect
    - RRTstar
    - TRRT
    - PRM
    - PRMstar
    - FMT
    - BFMT
    - PDST
    - STRIDE
    - BiTRRT
    - LBTRRT
    - BiEST
    - ProjEST
    - LazyPRM
    - LazyPRMstar
    - SPARS
    - SPARStwo
EOF

# 시스템 위치로 이동 (sudo)
sudo mv /tmp/ompl_planning_new.yaml \
        /opt/ros/jazzy/share/ur_moveit_config/config/ompl_planning.yaml

# 권한 확인 (root:root, 644 정도)
ls -la /opt/ros/jazzy/share/ur_moveit_config/config/ompl_planning.yaml
```

### Step 3: 검증

```bash
# 1. yaml 형식 valid 확인
python3 -c "import yaml; yaml.safe_load(open('/opt/ros/jazzy/share/ur_moveit_config/config/ompl_planning.yaml'))" \
  && echo "yaml OK" || echo "yaml ERROR"

# 2. 첫 30 줄 확인
head -30 /opt/ros/jazzy/share/ur_moveit_config/config/ompl_planning.yaml
```

`yaml OK` 나와야 함.

### Step 4: PLANNER_ID 다시 RRTConnect 로

```bash
sed -i 's/PLANNER_ID = "PTP"/PLANNER_ID = "RRTConnect"/' \
   ~/sketch_robot_ws/src/sketch_control/sketch_control/moveit_executor.py

grep -n "PLANNER_ID" ~/sketch_robot_ws/src/sketch_control/sketch_control/moveit_executor.py
```

### Step 5: SAFETY_OFFSET 도 0.15 로 복원

OMPL 은 우회 가능하니 0.25 까지 늘릴 필요 없음. 0.15 충분.

```bash
sed -i 's/SAFETY_OFFSET = 0.25/SAFETY_OFFSET = 0.15/' \
   ~/sketch_robot_ws/src/sketch_control/sketch_control/moveit_executor.py

grep -n "SAFETY_OFFSET" ~/sketch_robot_ws/src/sketch_control/sketch_control/moveit_executor.py
```

### Step 6: Build

```bash
cd ~/sketch_robot_ws
colcon build --packages-select sketch_control --symlink-install
source install/setup.bash
```

### Step 7: 검증 — query_planner_interface 다시

기존 launch 들 정리 후 재시작:

```bash
pkill -9 -f isaac_sim
sleep 5
pkill -9 -f "ros2|moveit|rviz|tcp_endpoint|robot_state|world_to|move_group|weld"
sleep 5

# Isaac Sim
~/sketch_robot_ws/run_isaac_sim.sh
# (별도 터미널에서)
ros2 launch sketch_control phase2_unity.launch.py
```

새 터미널에서 (또 별도):
```bash
source /opt/ros/jazzy/setup.bash
source ~/sketch_robot_ws/install/setup.bash
ros2 service call /query_planner_interface moveit_msgs/srv/QueryPlannerInterfaces "{}" \
  | head -30
```

기대 결과: `planner_ids=['AnytimePathShortening', 'SBL', 'EST', ..., 'RRTConnect', ..., 'SPARStwo']`
RRTConnect 가 목록에 보여야 함.

## 복원 방법 (필요 시)

문제가 생겨서 되돌리고 싶다면:

```bash
sudo cp /opt/ros/jazzy/share/ur_moveit_config/config/ompl_planning.yaml.bak_phase3 \
        /opt/ros/jazzy/share/ur_moveit_config/config/ompl_planning.yaml

sed -i 's/PLANNER_ID = "RRTConnect"/PLANNER_ID = "PTP"/' \
   ~/sketch_robot_ws/src/sketch_control/sketch_control/moveit_executor.py

cd ~/sketch_robot_ws
colcon build --packages-select sketch_control --symlink-install
```
