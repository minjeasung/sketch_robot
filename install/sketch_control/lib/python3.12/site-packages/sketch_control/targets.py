"""작업 대상 물체 관리 + 표면 평면 계산.

sketch_ui / moveit_executor 가 공유하는 유틸.
YAML 로부터 objects 를 로드하고, 각 box 물체의 sketch_face 에 해당하는
표면 평면(point, normal) 과 EE 방향(tool0 +Z 가 향할 world 방향)을 반환.
"""
import os
import yaml
import numpy as np
from ament_index_python.packages import get_package_share_directory


def load_objects_config():
    pkg_share = get_package_share_directory("sketch_control")
    cfg_path = os.path.join(pkg_share, "config", "objects.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


_FACE_MAP = {
    "+x": (np.array([1.0, 0.0, 0.0]), 0),
    "-x": (np.array([-1.0, 0.0, 0.0]), 0),
    "+y": (np.array([0.0, 1.0, 0.0]), 1),
    "-y": (np.array([0.0, -1.0, 0.0]), 1),
    "+z": (np.array([0.0, 0.0, 1.0]), 2),
    "-z": (np.array([0.0, 0.0, -1.0]), 2),
}


def get_surface_plane(obj):
    """(point_on_plane, normal) 반환. world 좌표계.
    point_on_plane: 면 중심. normal: 면 바깥 방향 단위벡터."""
    face = obj["sketch_face"]
    if face not in _FACE_MAP:
        raise ValueError(f"unknown sketch_face: {face}")
    normal, axis = _FACE_MAP[face]
    pos = np.array(obj["position"], dtype=float)
    half = np.array(obj["size"], dtype=float) / 2.0
    point_on_plane = pos + normal * half[axis]
    return point_on_plane, normal


def get_target(cfg, name):
    for obj in cfg["objects"]:
        if obj["name"] == name and obj.get("enabled", True):
            return obj
    raise KeyError(f"target '{name}' not found or disabled")


def list_enabled_targets(cfg):
    return [obj["name"] for obj in cfg["objects"] if obj.get("enabled", True)]


def ee_quat_for_target(obj):
    """타겟의 tool0 orientation 쿼터니언 [x, y, z, w] 반환.
    objects.yaml 의 ee_orientation 필드가 우선 (수동 캘리브레이션 값).
    없으면 face normal 로부터 계산 (tool0 +Y 가 -normal 향하도록)."""
    if "ee_orientation" in obj:
        return np.asarray(obj["ee_orientation"], dtype=float)
    # Fallback: face normal 기반 계산 (tool0 +Y = brush axis)
    _, normal = get_surface_plane(obj)
    d = -np.asarray(normal, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-12)
    y = np.array([0.0, 1.0, 0.0])
    dot = float(np.clip(np.dot(y, d), -1.0, 1.0))
    if dot > 0.9999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot < -0.9999:
        return np.array([0.0, 0.0, 1.0, 0.0])
    axis = np.cross(y, d)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    angle = np.arccos(dot)
    s = np.sin(angle / 2.0)
    c = np.cos(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])


# 하위 호환 별칭 (기존 import 지원)
def ee_quat_for_face(normal):
    """Deprecated: ee_quat_for_target(obj) 권장. face normal 만으로 계산."""
    dummy = {"sketch_face": None, "position": [0, 0, 0], "size": [0, 0, 0]}
    # normal 을 직접 받아 계산 (기존 로직, tool0 +Y 기준)
    d = -np.asarray(normal, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-12)
    y = np.array([0.0, 1.0, 0.0])
    dot = float(np.clip(np.dot(y, d), -1.0, 1.0))
    if dot > 0.9999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot < -0.9999:
        return np.array([0.0, 0.0, 1.0, 0.0])
    axis = np.cross(y, d)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    angle = np.arccos(dot)
    s = np.sin(angle / 2.0)
    c = np.cos(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])
