from __future__ import annotations

import numpy as np


def svd_solve(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form least-squares rigid transform from source onto target.

    Given N corresponding 3D points, returns R (3,3) and t (3,) minimizing
    sum_i || R @ source_pts[i] + t - target_pts[i] ||^2 (Kabsch / Umeyama,
    rigid-only). No Open3D, no iteration — one SVD.
    """
    if source_pts.shape != target_pts.shape:
        raise ValueError(
            f"shape mismatch: source {source_pts.shape} vs target {target_pts.shape}"
        )
    if source_pts.ndim != 2 or source_pts.shape[1] != 3:
        raise ValueError(f"expected (N, 3) arrays, got {source_pts.shape}")
    if source_pts.shape[0] < 3:
        raise ValueError(f"need at least 3 correspondences, got {source_pts.shape[0]}")

    # 1. Centroids. The optimal translation aligns the centroids, so once
    #    we subtract them off we are left with a pure-rotation problem.
    centroid_src = source_pts.mean(axis=0)
    centroid_tgt = target_pts.mean(axis=0)

    # 2. Center both point sets around their own centroids. Now we want the
    #    rotation R that best maps the centered source onto the centered target.
    src_centered = source_pts - centroid_src
    tgt_centered = target_pts - centroid_tgt

    # 3. Cross-covariance matrix H (3x3). Each entry H[i, j] is the sum over
    #    correspondences of src_centered[:, i] * tgt_centered[:, j] — a measure
    #    of how the two clouds co-vary along each pair of axes.
    H = src_centered.T @ tgt_centered

    # 4. SVD of H. The singular values themselves are irrelevant for the
    #    rotation; only the orthonormal frames U and V matter. The optimal
    #    rotation lives in the product V @ U.T.
    U, _, Vt = np.linalg.svd(H)

    # 5. Reflection check. Naively R = Vt.T @ U.T can land on a reflection
    #    (det = -1) when the data is noisy or nearly coplanar. Flipping the
    #    sign of the last column of V (equivalently, last row of Vt) before
    #    forming the product guarantees det(R) = +1 — a proper rotation.
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    # 6. Translation drops out of the centroid equation: R sends centroid_src
    #    to (R @ centroid_src), so t is whatever offset takes that to centroid_tgt.
    t = centroid_tgt - R @ centroid_src

    return R, t
