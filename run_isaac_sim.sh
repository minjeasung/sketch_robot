#!/bin/bash
# Isaac Sim 전용 환경: ROS2 Python 은 제거하되 Isaac 번들 C++ libs 는 유지

# ROS2 Python/빌드 관련만 제거 (시스템 Jazzy 와 충돌 방지)
unset ROS_VERSION ROS_PYTHON_VERSION
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
unset PYTHONPATH

# Isaac Sim 번들 Jazzy libs 경로 자동 탐지 (여러 위치 검색)
ISAAC_JAZZY_LIB=$(find "$HOME/.local/share/ov/data" -path "*/isaacsim.ros2.bridge-*/jazzy/lib" -type d 2>/dev/null | head -1)

if [ -z "$ISAAC_JAZZY_LIB" ]; then
    echo "ERROR: Isaac Sim bridge libs 경로를 못 찾음."
    echo "  검색 패턴: $ISAAC_BRIDGE_GLOB"
    echo "  Isaac Sim 을 한 번 실행해서 캐시 생성 후 다시 시도."
    exit 1
fi

# libament_index_cpp.so 가 실제로 있는지 확인
if [ ! -f "$ISAAC_JAZZY_LIB/libament_index_cpp.so" ]; then
    echo "WARN: Isaac 번들에 libament_index_cpp.so 가 없음."
    echo "  시스템 /opt/ros/jazzy/lib 도 추가 (C++ libs 만)."
    SYS_JAZZY_LIB="/opt/ros/jazzy/lib"
    export LD_LIBRARY_PATH="${ISAAC_JAZZY_LIB}:${SYS_JAZZY_LIB}:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${ISAAC_JAZZY_LIB}:${LD_LIBRARY_PATH}"
fi

export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

echo "=== Isaac Sim 환경 ==="
echo "ISAAC_JAZZY_LIB = $ISAAC_JAZZY_LIB"
echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
echo "======================"

source ~/isaac_env/bin/activate
exec isaacsim --exec ~/sketch_robot_ws/src/sketch_control/sketch_control/isaac_sim_rb10.py
