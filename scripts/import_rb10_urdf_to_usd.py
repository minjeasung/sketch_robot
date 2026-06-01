#!/usr/bin/env python3
"""Regenerate the RB10 Isaac USD from the RViz/MoveIt-aligned URDF.

Run with Isaac Sim, not system Python:
    isaacsim --no-window --exec scripts/import_rb10_urdf_to_usd.py
"""
from pathlib import Path
import os
import shutil
import sys

import omni
import omni.kit.app
import omni.kit.commands

try:
    from isaacsim.core.utils.extensions import enable_extension
except ImportError:
    from omni.isaac.core.utils.extensions import enable_extension


WS = Path.home() / "sketch_robot_ws"
URDF_PATH = WS / "isaac_assets" / "rb10_1300e_u.urdf"
USD_PATH = WS / "isaac_assets" / "rb10_1300e_u.usd"


def main():
    if not URDF_PATH.exists():
        raise FileNotFoundError(URDF_PATH)

    text = URDF_PATH.read_text(encoding="utf-8")
    expected = '<origin rpy="0 0 -1.5708" xyz="0 0 0"/>'
    if expected not in text:
        raise RuntimeError(
            "URDF is not RViz/MoveIt aligned: missing base joint "
            f"{expected}"
        )

    enable_extension("isaacsim.asset.importer.urdf")
    app = omni.kit.app.get_app()
    for _ in range(100):
        app.update()

    if USD_PATH.exists():
        backup = USD_PATH.with_suffix(".usd.bak")
        shutil.copy2(USD_PATH, backup)
        print(f"[OK] backup: {backup}")

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("URDFCreateImportConfig failed")

    import_config.merge_fixed_joints = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.create_physics_scene = False
    import_config.import_inertia_tensor = True
    import_config.collision_from_visuals = False

    status, imported_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(URDF_PATH),
        import_config=import_config,
        dest_path=str(USD_PATH),
    )
    if not status:
        raise RuntimeError("URDFParseAndImportFile failed")

    print(f"[OK] imported RB10: {imported_path}")
    print(f"[OK] wrote USD: {USD_PATH}")

    for _ in range(20):
        app.update()
    app.post_quit()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Isaac Kit can keep background threads alive after post_quit() in headless
    # one-shot imports. This script is only used as a regeneration command.
    os._exit(0)
