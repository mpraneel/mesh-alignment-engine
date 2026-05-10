from __future__ import annotations

import numpy as np
import open3d as o3d


def align_point_to_plane(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance: float,
    init: np.ndarray | None = None,
    max_iterations: int = 50,
) -> o3d.pipelines.registration.RegistrationResult:
    """Run point-to-plane ICP aligning source onto target."""
    if not target.has_normals():
        raise ValueError("target point cloud must have normals for point-to-plane ICP")

    if init is None:
        init = np.eye(4)

    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )
