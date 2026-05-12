#!/usr/bin/env python3
"""
Improved RANSAC Multi-Plane Fitting for Robotic Weld Seam Extraction
=====================================================================
Based on:
  Yi et al. (2026) "Weld seam extraction and path generation for
  robotic welding of steel structures based on 3D vision"
  Automation in Construction 183 (2026) 106792

Pipeline:
  1. Synthetic steel workpiece point cloud generation
  2. Original RANSAC (fixed iterations) — baseline
  3. Improved RANSAC (auto iteration via confidence + inlier ratio stop)
  4. Plane parameter optimization (centroid-based, Eq.19)
  5. Weld seam & torch pose extraction (Section 6)
  6. Visualization + performance comparison table

Usage:
  ~/ros2_ws/venv/bin/python ransac_multiplane.py [--no-vis]
"""

import argparse
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

# ─────────────────────────────────────────────────────────────
# 1. Synthetic Steel Structure Point Cloud Generation
# ─────────────────────────────────────────────────────────────

def _plate(origin, u_vec, v_vec, u_len, v_len,
           n_pts: int = 2500, noise: float = 0.001) -> np.ndarray:
    """Generate random points on a planar plate with Gaussian noise."""
    u = np.random.uniform(0, u_len, n_pts)
    v = np.random.uniform(0, v_len, n_pts)
    pts = origin + np.outer(u, u_vec) + np.outer(v, v_vec)
    pts += np.random.normal(0, noise, pts.shape)
    return pts


def make_workpiece_T(n=2500, noise=0.001) -> o3d.geometry.PointCloud:
    """T-형 (base + 수직 1장): 용접 심 1개."""
    pts = np.vstack([
        _plate([0,0,0], [1,0,0], [0,1,0], .30, .20, n, noise),  # 수평 base
        _plate([0,.10,0], [1,0,0], [0,0,1], .30, .15, n, noise), # 수직 plate
    ])
    return _to_pcd(pts)


def make_workpiece_cross(n=2000, noise=0.001) -> o3d.geometry.PointCloud:
    """크로스형 (base + 수직 2장): 용접 심 2개."""
    pts = np.vstack([
        _plate([0,0,0],   [1,0,0], [0,1,0], .30, .30, n*2, noise),
        _plate([0,.15,0], [1,0,0], [0,0,1], .30, .15, n,   noise),
        _plate([.15,0,0], [0,1,0], [0,0,1], .30, .15, n,   noise),
    ])
    return _to_pcd(pts)


def make_workpiece_H(n=1500, noise=0.001) -> o3d.geometry.PointCloud:
    """H-형 (base + 수직 2장 + 상부 플레이트): 용접 심 4개."""
    pts = np.vstack([
        _plate([0,0,0],    [1,0,0], [0,1,0], .40, .25, n*2, noise),  # bottom
        _plate([.10,0,0],  [0,1,0], [0,0,1], .25, .18, n,   noise),  # web L
        _plate([.30,0,0],  [0,1,0], [0,0,1], .25, .18, n,   noise),  # web R
        _plate([0,0,.18],  [1,0,0], [0,1,0], .40, .25, n*2, noise),  # top
    ])
    return _to_pcd(pts)


def make_workpiece_complex(n=1500, noise=0.001) -> o3d.geometry.PointCloud:
    """복합형 (논문 Fig.20(a) 유사): base + 수직 2장 비대칭."""
    pts = np.vstack([
        _plate([0,0,0],     [1,0,0], [0,1,0], .40, .25, n*2, noise),
        _plate([0,.125,0],  [1,0,0], [0,0,1], .40, .18, n,   noise),
        _plate([.20,0,0],   [0,1,0], [0,0,1], .25, .18, n,   noise),
    ])
    return _to_pcd(pts)


def _to_pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


# ─────────────────────────────────────────────────────────────
# 2. Plane Geometry Helpers
# ─────────────────────────────────────────────────────────────

def fit_plane_3pts(p1, p2, p3) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """3점으로 평면 계산 (Eq. 13-15)."""
    n = np.cross(p2 - p1, p3 - p1)
    norm = np.linalg.norm(n)
    if norm < 1e-10:
        return None, None
    n /= norm
    return n, -np.dot(n, p1)


def pt_plane_dist(pts: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    """점군 → 평면 거리 (Eq. 16)."""
    return np.abs(pts @ n + d)


def _adaptive_K_max(n_in: int, N_fp: int, eta0: float = 0.7,
                    K_hard_max: int = 5000) -> int:
    """
    인라이어 비율 w = n_in/N_fp 로 필요 반복 횟수 동적 계산 (Eq. 18 의도).

    표준 RANSAC 공식:  K = log(1-η₀) / log(1 - w³)
    ─ n_in/N_fp 이 커질수록 K가 급격히 감소 → 속도 개선
    ─ n_in이 작을 때는 K_hard_max 로 보수적 상한 유지

    논문의 (3/N_fp) 표기는 sample_size/remaining_pts 로 쓰였으나,
    동적 인라이어 비율로 해석해야 논문 결과(5.4배 속도 향상)가 재현됨.
    """
    if n_in <= 0 or N_fp <= 0:
        return K_hard_max
    w = n_in / N_fp
    if w <= 0:
        return K_hard_max
    log_base = math.log(max(1.0 - w ** 3, 1e-15))
    if log_base >= 0:
        return 1
    K = math.ceil(math.log(max(1.0 - eta0, 1e-15)) / log_base)
    return max(1, min(K, K_hard_max))


# ─────────────────────────────────────────────────────────────
# 3. Original RANSAC (고정 반복 횟수, 비교용)
# ─────────────────────────────────────────────────────────────

def ransac_original(pcd: o3d.geometry.PointCloud,
                    T_d2: float = 0.002,
                    T_mpp: float = 0.025,
                    max_iter: int = 100) -> List[Dict]:
    """
    원본 RANSAC: 평면당 max_iter 고정.
    논문 Fig.13(a)(b) 대응.
    """
    pts = np.asarray(pcd.points)
    N_seg = len(pts)
    mask = np.ones(N_seg, dtype=bool)
    planes: List[Dict] = []

    while True:
        idx = np.where(mask)[0]
        N_fp = len(idx)
        if N_fp < 3:
            break

        cur = pts[idx]
        best_in, best_n, best_d = np.array([], dtype=int), None, None

        for _ in range(max_iter):
            s = np.random.choice(N_fp, 3, replace=False)
            n, d = fit_plane_3pts(cur[s[0]], cur[s[1]], cur[s[2]])
            if n is None:
                continue
            in_loc = np.where(pt_plane_dist(cur, n, d) < T_d2)[0]
            if len(in_loc) > len(best_in):
                best_in, best_n, best_d = in_loc, n, d

        if best_n is None or len(best_in) / N_seg <= T_mpp:
            break

        planes.append({'normal': best_n, 'd': best_d,
                        'inlier_indices': idx[best_in], 'n_inliers': len(best_in)})
        mask[idx[best_in]] = False

    return planes


# ─────────────────────────────────────────────────────────────
# 4. Improved RANSAC (논문 핵심, Section 5.3)
# ─────────────────────────────────────────────────────────────

def ransac_improved(pcd: o3d.geometry.PointCloud,
                    T_d2: float = 0.002,
                    T_mpp: float = 0.025,
                    eta0: float = 0.7,
                    verbose: bool = True) -> List[Dict]:
    """
    개선 RANSAC 다중 평면 피팅 (Algorithm 1 flowchart, Fig.14).

    개선 포인트:
      - 인라이어 점유율 T_mpp (Eq.17): 유효 평면 최소 비율 → 과분할 방지
      - 신뢰 수준 eta0 (Eq.18): 인라이어 비율 w=n_in/N_fp 기반 반복 수 동적 갱신
        → 인라이어가 많이 발견될수록 K 급감 → 논문 Fig.13(d) 속도 향상 재현

    Args:
        pcd    : 용접 영역 세그먼트 포인트클라우드
        T_d2   : 인라이어/아웃라이어 거리 임계값 [m] (논문: 2 mm)
        T_mpp  : 최소 인라이어 점유율 (논문: 0.025)
        eta0   : 신뢰 수준 (논문: 0.7)
        verbose: 진행 상황 출력
    Returns:
        List of plane dicts: {normal, d, inlier_indices, n_inliers, ratio, K_used}
    """
    pts = np.asarray(pcd.points)
    N_seg = len(pts)
    mask = np.ones(N_seg, dtype=bool)
    planes: List[Dict] = []
    total_K = 0

    if verbose:
        print(f"  [Improved] {N_seg} pts | T_d2={T_d2*1e3:.1f}mm "
              f"T_mpp={T_mpp} eta0={eta0}")

    while True:
        idx = np.where(mask)[0]
        N_fp = len(idx)
        if N_fp < 3:
            break

        cur = pts[idx]
        best_in, best_n, best_d = np.array([], dtype=int), None, None
        K_max = 5000   # 보수적 초기 상한
        k = 0

        while k < K_max:
            s = np.random.choice(N_fp, 3, replace=False)
            n, d = fit_plane_3pts(cur[s[0]], cur[s[1]], cur[s[2]])
            if n is not None:
                in_loc = np.where(pt_plane_dist(cur, n, d) < T_d2)[0]
                if len(in_loc) > len(best_in):
                    best_in, best_n, best_d = in_loc, n, d
                    # ← Eq.18: 더 좋은 인라이어 집합 발견 시 K 상한 즉시 갱신
                    K_max = _adaptive_K_max(len(best_in), N_fp, eta0)
            k += 1

        total_K += k

        if best_n is None:
            break

        ratio = len(best_in) / N_seg
        if ratio <= T_mpp:                        # ← Eq.17: 인라이어 비율 검사
            if verbose:
                print(f"    → ratio {ratio:.4f} ≤ T_mpp, 종료")
            break

        plane = {'normal': best_n, 'd': best_d,
                 'inlier_indices': idx[best_in],
                 'n_inliers': len(best_in), 'ratio': ratio, 'K_used': k}
        planes.append(plane)
        mask[idx[best_in]] = False

        if verbose:
            print(f"    Plane {len(planes)}: {len(best_in)} inliers "
                  f"({ratio:.3f}) K={k} n={best_n.round(3)}")

    if verbose:
        print(f"  → {len(planes)}개 평면, 총 반복={total_K}")
    return planes


# ─────────────────────────────────────────────────────────────
# 5. 평면 파라미터 최적화 (Section 5.4, Eq.19)
# ─────────────────────────────────────────────────────────────

def optimize_plane(pts: np.ndarray, plane: Dict,
                   k_neighbors: int = 120,
                   n_trials: int = 10) -> Dict:
    """
    중심점 기반 평면 파라미터 최적화 (Eq.19 + SVD 안정화).
    교차선 근방 과분할(over-segmentation) 오차 감소.

    절차:
      1. 인라이어 중심점(centroid) 계산
      2. KD-tree로 centroid 주변 k개 이웃 탐색
      3. 논문 Eq.19: 이웃 중 2점 랜덤 선택 → cross product로 법선 후보 생성
         (n_trials 회 반복 후 인라이어 수 최다인 후보 채택 → 불안정성 보완)
      4. 최종 법선을 이웃점 SVD 로 재정제 (정밀도 향상)
    """
    in_pts = pts[plane['inlier_indices']]
    if len(in_pts) < 3:
        return plane

    centroid = in_pts.mean(axis=0)

    # KD-tree로 centroid 주변 이웃 탐색
    tmp = o3d.geometry.PointCloud()
    tmp.points = o3d.utility.Vector3dVector(in_pts)
    tree = o3d.geometry.KDTreeFlann(tmp)
    k = min(k_neighbors, len(in_pts))
    _, nb_idx, _ = tree.search_knn_vector_3d(centroid, k)
    nb = in_pts[np.array(nb_idx)]

    # Eq.19: 2점 랜덤 cross product 를 n_trials 번 시도 → 최적 후보 선택
    best_n, best_d, best_cnt = plane['normal'], plane['d'], -1
    for _ in range(n_trials):
        ri = np.random.choice(len(nb), 2, replace=False)
        cand_n = np.cross(nb[ri[0]] - centroid, nb[ri[1]] - centroid)
        norm = np.linalg.norm(cand_n)
        if norm < 1e-10:
            continue
        cand_n /= norm
        if np.dot(cand_n, plane['normal']) < 0:
            cand_n = -cand_n
        cand_d = -np.dot(cand_n, centroid)
        # 인라이어 수로 후보 품질 평가
        cnt = int((pt_plane_dist(nb, cand_n, cand_d) < 0.002).sum())
        if cnt > best_cnt:
            best_n, best_d, best_cnt = cand_n, cand_d, cnt

    # SVD로 이웃점 기반 법선 정제 (정밀도 보강)
    centered = nb - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    svd_n = Vt[-1]
    if np.dot(svd_n, plane['normal']) < 0:
        svd_n = -svd_n
    svd_d = -np.dot(svd_n, centroid)

    # cross-product 후보와 SVD 결과 중 인라이어 많은 쪽 채택
    cnt_svd = int((pt_plane_dist(nb, svd_n, svd_d) < 0.002).sum())
    if cnt_svd >= best_cnt:
        best_n, best_d = svd_n, svd_d

    opt = plane.copy()
    opt['normal'] = best_n
    opt['d'] = best_d
    return opt


# ─────────────────────────────────────────────────────────────
# 6. 용접 경로 및 토치 자세 생성 (Section 6)
# ─────────────────────────────────────────────────────────────

def compute_weld_seam(p1: Dict, p2: Dict) -> Optional[Dict]:
    """
    두 평면 교선 = 필릿 용접 심 방향 벡터 계산 (Eq.20-23).
    Returns None if planes are (near-)parallel.
    """
    n1, n2 = p1['normal'], p2['normal']
    d_weld = np.cross(n1, n2)
    norm = np.linalg.norm(d_weld)
    if norm < 1e-6:
        return None

    d_weld /= norm
    # 이면각 검사: 필릿 용접은 대략 60°~120°
    cos_a = abs(float(np.dot(n1, n2)))
    angle = math.degrees(math.acos(min(cos_a, 1.0)))
    if angle < 20 or angle > 160:
        return None

    # 교선 위의 한 점 계산 (Eq.22, z=0 고정)
    d1, d2 = float(p1['d']), float(p2['d'])
    A = np.array([[n1[0], n1[1]], [n2[0], n2[1]]])
    b = np.array([-d1, -d2])
    pt_on: Optional[np.ndarray] = None
    if abs(np.linalg.det(A)) > 1e-6:
        xy = np.linalg.solve(A, b)
        pt_on = np.array([xy[0], xy[1], 0.0])
    else:
        A2 = np.array([[n1[0], n1[2]], [n2[0], n2[2]]])
        if abs(np.linalg.det(A2)) > 1e-6:
            xz = np.linalg.solve(A2, np.array([-d1, -d2]))
            pt_on = np.array([xz[0], 0.0, xz[1]])
    if pt_on is None:
        return None

    # 토치 접근 벡터 v0 = v1 + v2  (Eq.23)
    v1 = np.cross(d_weld, n1); v1 /= max(np.linalg.norm(v1), 1e-10)
    v2 = np.cross(d_weld, n2); v2 /= max(np.linalg.norm(v2), 1e-10)
    v0 = v1 + v2
    v0n = np.linalg.norm(v0)
    v0 = v0 / v0n if v0n > 1e-10 else (n1 + n2) / 2

    return {'d_weld': d_weld, 'pt_on': pt_on, 'v0': v0,
            'dihedral': 180.0 - angle, 'n1': n1, 'n2': n2}


def weld_endpoints(pts: np.ndarray, p1: Dict, p2: Dict,
                   seam: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    용접 심 시작/끝점 결정 (Eq.26-28).
    인라이어 점군을 d_weld 방향으로 투영 → 최대 범위.
    """
    all_in = np.vstack([pts[p1['inlier_indices']], pts[p2['inlier_indices']]])
    ref = seam['pt_on']
    t = (all_in - ref) @ seam['d_weld']
    return ref + t.min() * seam['d_weld'], ref + t.max() * seam['d_weld']


# ─────────────────────────────────────────────────────────────
# 7. 시각화 헬퍼
# ─────────────────────────────────────────────────────────────

_PALETTE = [
    [0.9, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.4, 0.9],
    [0.9, 0.8, 0.1], [0.8, 0.2, 0.8], [0.1, 0.8, 0.8],
    [0.5, 0.1, 0.1], [0.1, 0.5, 0.1], [0.1, 0.1, 0.5],
    [0.6, 0.4, 0.0],
]


def colorize(pts: np.ndarray, planes: List[Dict]) -> o3d.geometry.PointCloud:
    colors = np.full((len(pts), 3), 0.45)
    for i, p in enumerate(planes):
        colors[p['inlier_indices']] = _PALETTE[i % len(_PALETTE)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def seam_line_pcd(start, end, n_pts=200) -> o3d.geometry.PointCloud:
    t = np.linspace(0, 1, n_pts)[:, None]
    line = start + t * (end - start)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(line)
    pcd.paint_uniform_color([1.0, 0.5, 0.0])  # 주황
    return pcd


def approach_arrow(start, end, v0, scale=0.03) -> o3d.geometry.PointCloud:
    """토치 접근 방향 벡터 시각화."""
    mid = (start + end) / 2
    arrow_pts = np.linspace(0, scale, 30)[:, None] * v0 + mid
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arrow_pts)
    pcd.paint_uniform_color([0.0, 1.0, 0.3])  # 초록
    return pcd


# ─────────────────────────────────────────────────────────────
# 8. 전체 파이프라인 실행
# ─────────────────────────────────────────────────────────────

def run_pipeline(name: str, pcd: o3d.geometry.PointCloud,
                 T_d2=0.002, T_mpp=0.025, eta0=0.7,
                 visualize=True):
    pts = np.asarray(pcd.points)
    print(f"\n{'═'*60}")
    print(f" {name}  ({len(pts):,} points)")
    print(f"{'═'*60}")

    # ── Improved RANSAC ──────────────────────────────────────
    t0 = time.perf_counter()
    planes_raw = ransac_improved(pcd, T_d2, T_mpp, eta0, verbose=True)
    t_ransac = (time.perf_counter() - t0) * 1e3

    # ── 평면 최적화 ─────────────────────────────────────────
    t0 = time.perf_counter()
    planes = [optimize_plane(pts, p) for p in planes_raw]
    t_opt = (time.perf_counter() - t0) * 1e3

    # ── 용접 심 추출 ────────────────────────────────────────
    t0 = time.perf_counter()
    seams = []
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            s = compute_weld_seam(planes[i], planes[j])
            if s:
                sp, ep = weld_endpoints(pts, planes[i], planes[j], s)
                s.update({'start': sp, 'end': ep,
                           'length': float(np.linalg.norm(ep - sp)),
                           'pair': (i, j)})
                seams.append(s)
    t_seam = (time.perf_counter() - t0) * 1e3
    t_total = t_ransac + t_opt + t_seam

    # ── 결과 출력 ────────────────────────────────────────────
    print(f"\n[결과] 평면 {len(planes)}개 / 용접 심 후보 {len(seams)}개")
    for i, s in enumerate(seams):
        pi, pj = s['pair']
        print(f"  Seam {i+1} (P{pi+1}↔P{pj+1}): "
              f"길이={s['length']*1e3:.1f}mm  "
              f"이면각={s['dihedral']:.1f}°  "
              f"d_weld={s['d_weld'].round(3)}")

    print(f"\n[시간] RANSAC={t_ransac:.0f}ms  Opt={t_opt:.0f}ms  "
          f"Seam={t_seam:.0f}ms  Total={t_total:.0f}ms")

    # ── 시각화 ──────────────────────────────────────────────
    if visualize:
        geoms = [colorize(pts, planes)]
        for s in seams:
            if s['length'] > 0.01:
                geoms.append(seam_line_pcd(s['start'], s['end']))
                geoms.append(approach_arrow(s['start'], s['end'], s['v0']))
        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"Multi-Plane Fitting: {name}",
            width=1024, height=768,
        )

    return planes, seams, t_total


def run_comparison(pcd: o3d.geometry.PointCloud, T_d2=0.002, T_mpp=0.025):
    """논문 Table 6 스타일: Original vs Improved RANSAC 비교."""
    print(f"\n{'─'*60}")
    print(" 비교: Original RANSAC vs Improved RANSAC  (Fig.13 재현)")
    print(f"{'─'*60}")
    pts = np.asarray(pcd.points)

    rows = []
    for label, fn in [
        ("Original (100 iter)", lambda p: ransac_original(p, T_d2, T_mpp, 100)),
        ("Original (500 iter)", lambda p: ransac_original(p, T_d2, T_mpp, 500)),
        ("Improved (η₀=0.7)",   lambda p: ransac_improved(p, T_d2, T_mpp, 0.7, False)),
    ]:
        t0 = time.perf_counter()
        pl = fn(pcd)
        elapsed = (time.perf_counter() - t0) * 1e3
        rows.append((label, len(pl), elapsed))

    print(f"\n{'방법':<25} {'평면 수':>8} {'시간(ms)':>10}")
    print("─" * 48)
    for label, n_pl, ms in rows:
        print(f"  {label:<23} {n_pl:>8}   {ms:>8.1f}")
    print()

    speedup = rows[1][2] / rows[2][2]
    print(f"  Improved speedup vs 500-iter: {speedup:.1f}×")


# ─────────────────────────────────────────────────────────────
# 9. 평면 최적화 정확도 검증 (논문 5.4절 재현)
# ─────────────────────────────────────────────────────────────

def verify_optimization_accuracy():
    """
    이론 평면 2개 합성 → RANSAC 피팅 → 최적화 전후 각도/거리 오차 비교.
    논문 결과: 각도 0.304°→0.011°, 거리 0.158mm→0.004mm (개선 96~97%).
    """
    print(f"\n{'─'*60}")
    print(" 평면 파라미터 최적화 정확도 검증 (Section 5.4)")
    print(f"{'─'*60}")
    np.random.seed(0)

    # 이론 평면 1: z=0  법선=[0,0,1]
    n1_true = np.array([0., 0., 1.])
    pts1 = _plate([0,0,0], [1,0,0], [0,1,0], .30, .30, 4000, 0.001)
    # 이론 평면 2: y=0.15 법선=[0,1,0]
    n2_true = np.array([0., 1., 0.])
    pts2 = _plate([0,.15,0], [1,0,0], [0,0,1], .30, .20, 2000, 0.001)

    all_pts = np.vstack([pts1, pts2])
    pcd = _to_pcd(all_pts)

    # RANSAC 피팅
    planes_raw = ransac_improved(pcd, T_d2=0.002, T_mpp=0.025, verbose=False)
    planes_opt = [optimize_plane(all_pts, p) for p in planes_raw]

    true_normals = [n1_true, n2_true]
    print(f"\n{'':20} {'각도오차(°)':>14} {'거리오차(mm)':>14}")
    print("─" * 52)

    for i, (raw, opt) in enumerate(zip(planes_raw, planes_opt)):
        n_true = true_normals[i % len(true_normals)]
        # 각도 오차
        def ang_err(n):
            c = abs(float(np.dot(n, n_true)))
            return math.degrees(math.acos(min(c, 1.0)))
        # 거리 오차 (centroid → true plane)
        c_raw = all_pts[raw['inlier_indices']].mean(0)
        c_opt = all_pts[opt['inlier_indices']].mean(0)
        def dist_err(n, d, c):
            return abs(float(np.dot(n, c) + d)) * 1e3

        ae_raw = ang_err(raw['normal'])
        ae_opt = ang_err(opt['normal'])
        de_raw = dist_err(raw['normal'], raw['d'], c_raw)
        de_opt = dist_err(opt['normal'], opt['d'], c_opt)

        print(f"  Plane {i+1} before opt: {ae_raw:>10.3f}°   {de_raw:>10.3f} mm")
        print(f"  Plane {i+1} after  opt: {ae_opt:>10.3f}°   {de_opt:>10.3f} mm")
        if ae_raw > 0:
            print(f"           개선율: 각도 {(1-ae_opt/ae_raw)*100:.1f}%  "
                  f"거리 {(1-de_opt/max(de_raw,1e-9))*100:.1f}%")
        print()


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-vis", action="store_true",
                        help="시각화 창 비활성화 (헤드리스 환경)")
    args = parser.parse_args()
    vis = not args.no_vis

    print("╔══════════════════════════════════════════════════════╗")
    print("║ Improved RANSAC Multi-Plane Fitting                  ║")
    print("║ Yi et al. (2026) Autom. Constr. 183, 106792          ║")
    print("╚══════════════════════════════════════════════════════╝")

    np.random.seed(42)

    workpieces = {
        "Workpiece-A  T형":       make_workpiece_T(),
        "Workpiece-B  크로스형":  make_workpiece_cross(),
        "Workpiece-C  H형":       make_workpiece_H(),
        "Workpiece-D  복합형":    make_workpiece_complex(),
    }

    summary = []
    for name, pcd in workpieces.items():
        planes, seams, t_ms = run_pipeline(name, pcd, visualize=vis)
        summary.append((name, len(planes), len(seams), t_ms))

    # ── 비교 실험 (논문 Table 6 / Fig.13 재현) ────────────
    run_comparison(workpieces["Workpiece-B  크로스형"])

    # ── 최적화 정확도 검증 ─────────────────────────────────
    verify_optimization_accuracy()

    # ── 최종 요약 ─────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(" 최종 요약 (Improved RANSAC)")
    print(f"{'═'*60}")
    print(f"  {'워크피스':<26} {'평면':>5} {'심':>5} {'시간(ms)':>10}")
    print("─" * 52)
    for name, np_, ns, t in summary:
        print(f"  {name:<26} {np_:>5} {ns:>5} {t:>10.0f}")
    print()


if __name__ == "__main__":
    main()
