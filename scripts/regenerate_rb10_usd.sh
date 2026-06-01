#!/bin/bash
# Regenerate the Isaac RB10 USD from the same URDF convention used by RViz/MoveIt.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

bash "$SCRIPT_DIR/setup_isaac_assets.sh"

source "$HOME/isaac_env/bin/activate"
cd "$WS_DIR"
exec isaacsim --no-window --exec "$WS_DIR/scripts/import_rb10_urdf_to_usd.py"
