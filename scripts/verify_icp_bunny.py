"""Apply a known rigid offset to the Stanford bunny and check the wrapper recovers its inverse."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from icp.align import align_point_to_plane
from icp.loader import load_point_cloud
from icp.solver import svd_solve


def known_offset() -> np.ndarray:
    theta = np.deg2rad(5.0)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    t = np.array([0.01, 0.005, 0.0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def main() -> int:
    bunny_path = o3d.data.BunnyMesh().path
    target = load_point_cloud(bunny_path)

    T_offset = known_offset()
    source = copy.deepcopy(target).transform(T_offset)

    result = align_point_to_plane(
        source,
        target,
        max_correspondence_distance=0.02,
    )

    T_recovered = np.asarray(result.transformation)
    residual = T_recovered @ T_offset
    residual_err = float(np.linalg.norm(residual - np.eye(4)))

    print(f"fitness:        {result.fitness:.6f}")
    print(f"inlier rmse:    {result.inlier_rmse:.6e}")
    print(f"||T_recovered @ T_offset - I||_F: {residual_err:.6e}")
    print("T_offset:")
    print(T_offset)
    print("T_recovered:")
    print(T_recovered)

    corr = np.asarray(result.correspondence_set)
    src_xyz = np.asarray(source.points)[corr[:, 0]]
    tgt_xyz = np.asarray(target.points)[corr[:, 1]]

    R_svd, t_svd = svd_solve(src_xyz, tgt_xyz)
    R_o3d = T_recovered[:3, :3]
    t_o3d = T_recovered[:3, 3]

    R_diff = float(np.linalg.norm(R_svd - R_o3d))
    t_diff = float(np.linalg.norm(t_svd - t_o3d))
    print(f"||R_svd - R_o3d||_F: {R_diff:.6e}")
    print(f"||t_svd - t_o3d||:   {t_diff:.6e}")

    assert np.allclose(R_svd, R_o3d, atol=1e-4), "svd R disagrees with open3d R"
    assert np.allclose(t_svd, t_o3d, atol=1e-4), "svd t disagrees with open3d t"

    return 0 if residual_err < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
