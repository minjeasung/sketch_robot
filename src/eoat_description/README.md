# EOAT Description

This package contains a ROS 2 URDF description and STL meshes generated from:

`/home/minjea/Downloads/RR-00A_B__EOAT.step`

## Files

- `meshes/rr_00a_b_eoat_visual.stl`: high-detail visual mesh, converted from millimeters to meters.
- `meshes/rr_00a_b_eoat_collision.stl`: lower-detail collision mesh for MoveIt and Isaac Sim physics.
- `meshes/rr_00a_b_eoat_no_camera_visual.stl`: high-detail visual mesh with the D4xx camera body removed.
- `meshes/rr_00a_b_eoat_no_camera_collision.stl`: lower-detail collision mesh with the D4xx camera body removed.
- `urdf/rr_00a_b_eoat.urdf.xacro`: reusable Xacro macro for attaching the EOAT to an existing robot link.
- `urdf/rr_00a_b_eoat_standalone.urdf.xacro`: standalone wrapper with `base_link -> eoat_link`.
- `urdf/rr_00a_b_eoat_standalone.urdf`: generated plain URDF for tools that do not load Xacro.
- `urdf/rr_00a_b_eoat_no_camera.urdf.xacro`: reusable no-camera Xacro macro.
- `urdf/rr_00a_b_eoat_no_camera_standalone.urdf`: generated no-camera plain URDF.
- `urdf/rr_00a_b_eoat_no_camera_d405.urdf.xacro`: reusable no-camera EOAT + Intel RealSense D405 macro.
- `urdf/rr_00a_b_eoat_no_camera_d405_standalone.urdf`: generated no-camera EOAT + D405 plain URDF.

An Isaac Sim-ready copy is also available at:

`isaac_assets/eoat/rr_00a_b_eoat_standalone.urdf`

No-camera Isaac Sim copy:

`isaac_assets/eoat_no_camera/rr_00a_b_eoat_no_camera_standalone.urdf`

That file uses relative mesh paths (`./meshes/...`) so the Isaac Sim URDF importer can load it without resolving ROS package URIs.

## Notes

The STEP file is in millimeters. The generated STLs are already in meters, so the URDF mesh scale is left at `1 1 1`.

The mesh origin is the STL visual mesh bounding-box center. To attach this EOAT to a robot flange in MoveIt or Isaac Sim, include `rr_00a_b_eoat.urdf.xacro` and set the fixed joint `xyz` and `rpy` to the measured flange-to-EOAT-center transform.

Default inertial values use a 1 kg bounding-box approximation. Replace `mass` and inertia values with measured CAD or scale data before running dynamic simulation.

Example Xacro include:

```xml
<xacro:include filename="$(find eoat_description)/urdf/rr_00a_b_eoat.urdf.xacro"/>
<xacro:rr_00a_b_eoat
  parent="tcp"
  prefix="eoat_"
  xyz="0 0 0"
  rpy="0 0 0"
  mass="1.0"/>
```

No-camera variant:

```xml
<xacro:include filename="$(find eoat_description)/urdf/rr_00a_b_eoat_no_camera.urdf.xacro"/>
<xacro:rr_00a_b_eoat_no_camera
  parent="tcp"
  prefix="eoat_"
  xyz="0 0 0"
  rpy="0 0 0"
  mass="1.0"/>
```

No-camera EOAT with D405 mounted:

```xml
<xacro:include filename="$(find eoat_description)/urdf/rr_00a_b_eoat_no_camera_d405.urdf.xacro"/>
<xacro:rr_00a_b_eoat_no_camera_d405
  parent="tcp"
  prefix=""
  xyz="0 -0.16745 0"
  rpy="1.57079632679 0 0"
  mass="1.072"/>
```

The D405 macro comes from Intel RealSense `realsense2_description`. Its origin
is the bottom screw frame, so this package offsets that screw frame to place the
actual D405 camera link at TCP `xyz="0.00905 -0.07640 0.04375"`, matching the
Isaac Sim EOAT pose. The default no-camera EOAT-local mount is
`xyz="0.00005 0.02275 -0.10190"` and
`rpy="0 -1.57079632679 -1.57079632679"`.
