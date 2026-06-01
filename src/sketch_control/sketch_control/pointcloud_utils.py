"""Numpy-only point cloud helpers for runtime perception nodes."""
import numpy as np


def voxel_downsample(points, voxel_size):
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        return pts.reshape(0, 3)
    idx = np.floor(pts / float(voxel_size)).astype(np.int64)
    _uniq, inverse = np.unique(idx, axis=0, return_inverse=True)
    sums = np.zeros((_uniq.shape[0], 3), dtype=np.float64)
    counts = np.bincount(inverse).astype(np.float64)
    np.add.at(sums, inverse, pts)
    return (sums / counts[:, None]).astype(np.float32)


def ransac_plane(points, distance_threshold, num_iterations, sample_limit=50000,
                 seed=7):
    """Return (a,b,c,d), inlier_indices for a normalized plane."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 3:
        raise ValueError("at least 3 points are required")

    rng = np.random.default_rng(seed)
    if pts.shape[0] > sample_limit:
        sample_idx = rng.choice(pts.shape[0], size=sample_limit, replace=False)
        sample = pts[sample_idx]
    else:
        sample_idx = None
        sample = pts

    best_model = None
    best_count = -1
    best_error = float("inf")
    n_sample = sample.shape[0]

    for _ in range(int(num_iterations)):
        i0, i1, i2 = rng.choice(n_sample, size=3, replace=False)
        p0, p1, p2 = sample[i0], sample[i1], sample[i2]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -float(np.dot(normal, p0))
        distances = np.abs(sample @ normal + d)
        inlier_mask = distances < float(distance_threshold)
        count = int(np.count_nonzero(inlier_mask))
        if count <= best_count:
            continue
        error = float(np.mean(distances[inlier_mask])) if count else float("inf")
        best_model = (normal, d)
        best_count = count
        best_error = error

    if best_model is None:
        raise RuntimeError("RANSAC failed to find a plane")

    normal, d = best_model
    full_distances = np.abs(pts @ normal + d)
    inliers = np.flatnonzero(full_distances < float(distance_threshold))

    # One least-squares refinement on full inliers.
    if inliers.shape[0] >= 3:
        inlier_pts = pts[inliers].astype(np.float64)
        centroid = inlier_pts.mean(axis=0)
        _, _, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
        normal = vh[-1]
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        d = -float(np.dot(normal, centroid))
        full_distances = np.abs(pts @ normal + d)
        inliers = np.flatnonzero(full_distances < float(distance_threshold))

    return (
        [float(normal[0]), float(normal[1]), float(normal[2]), float(d)],
        inliers.tolist(),
    )
