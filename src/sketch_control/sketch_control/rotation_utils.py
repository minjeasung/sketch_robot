"""Small quaternion helpers used by runtime ROS nodes.

The workstation currently has an apt SciPy build that is binary-incompatible
with the active NumPy.  Keep runtime nodes independent from SciPy so motion
control can still start reliably.

Quaternion convention is ROS/scipy style: [x, y, z, w].
"""
import math
import numpy as np


def normalize_quat(q):
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def quat_to_matrix(q):
    x, y, z, w = normalize_quat(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=float)


def quat_apply(q, points):
    pts = np.asarray(points, dtype=float)
    return pts @ quat_to_matrix(q).T


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = normalize_quat(q1)
    x2, y2, z2, w2 = normalize_quat(q2)
    return normalize_quat(np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=float))


def quat_from_two_vectors(src, dst):
    a = np.asarray(src, dtype=float)
    b = np.asarray(dst, dtype=float)
    a /= np.linalg.norm(a) + 1e-12
    b /= np.linalg.norm(b) + 1e-12
    dot = float(np.dot(a, b))
    if dot > 1.0 - 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    if dot < -1.0 + 1e-10:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis) + 1e-12
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=float)
    axis = np.cross(a, b)
    q = np.array([axis[0], axis[1], axis[2], 1.0 + dot], dtype=float)
    return normalize_quat(q)


def quat_from_matrix(m):
    m = np.asarray(m, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return normalize_quat(np.array([x, y, z, w], dtype=float))
