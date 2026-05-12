#!/bin/bash
# Isaac Sim 용 RB10 URDF + mesh 준비.
# 전제: ~/rb10_ws/src/rbpodo_ros2 가 minjeasung/rbpodo_ros2 fork 로 clone 되어 있음.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ASSETS_DIR="$WS_DIR/isaac_assets"
RBPODO_DESCRIPTION="$HOME/rb10_ws/src/rbpodo_ros2/rbpodo_description"

if [ ! -d "$RBPODO_DESCRIPTION" ]; then
    echo "ERROR: $RBPODO_DESCRIPTION 가 없음."
    echo "  먼저: cd ~/rb10_ws/src && git clone https://github.com/minjeasung/rbpodo_ros2.git"
    exit 1
fi

if [ ! -d "$HOME/rb10_ws/install/rbpodo_description" ]; then
    echo "[1/3] rbpodo_description 빌드..."
    cd "$HOME/rb10_ws"
    colcon build --packages-select rbpodo_description --symlink-install
fi

source "$HOME/rb10_ws/install/setup.bash"

echo "[2/3] xacro -> urdf 변환..."
mkdir -p "$ASSETS_DIR"
ros2 run xacro xacro \
    "$RBPODO_DESCRIPTION/robots/rb10_1300e_u.urdf.xacro" \
    > "$ASSETS_DIR/rb10_1300e_u.urdf"

echo "[3/3] mesh 복사 + URDF path 치환..."
mkdir -p "$ASSETS_DIR/meshes"
cp -r "$RBPODO_DESCRIPTION/meshes/rb10_1300e_u" "$ASSETS_DIR/meshes/"
sed -i "s|package://rbpodo_description/meshes|./meshes|g" \
    "$ASSETS_DIR/rb10_1300e_u.urdf"

echo "완료. isaac_assets:"
ls -la "$ASSETS_DIR"
echo ""
echo "다음: Isaac Sim 띄워서 File > Import 로 $ASSETS_DIR/rb10_1300e_u.urdf import"
