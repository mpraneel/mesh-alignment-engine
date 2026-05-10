"""Open3D visualizations: tri-color alignment overlay and per-point deviation heatmap."""
from __future__ import annotations

import copy

import matplotlib as mpl
import numpy as np
import open3d as o3d


def show_alignment(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformed: o3d.geometry.PointCloud,
    window_name: str = "alignment (red=source, blue=target, green=transformed)",
) -> None:
    """Render source / target / transformed overlaid in one viewer for visual comparison."""
    src = copy.deepcopy(source).paint_uniform_color([1.0, 0.0, 0.0])
    tgt = copy.deepcopy(target).paint_uniform_color([0.0, 0.0, 1.0])
    out = copy.deepcopy(transformed).paint_uniform_color([0.0, 1.0, 0.0])
    o3d.visualization.draw_geometries([src, tgt, out], window_name=window_name)


def show_heatmap(
    transformed: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    window_name: str = "deviation heatmap (blue=0, red=max)",
) -> None:
    """Color each transformed point by its distance to the nearest target point."""
    distances = np.asarray(transformed.compute_point_cloud_distance(target))

    d_max = float(distances.max()) if distances.size else 0.0
    normed = distances / d_max if d_max > 0 else np.zeros_like(distances)

    rgba = mpl.colormaps["RdYlBu_r"](normed)
    colored = copy.deepcopy(transformed)
    colored.colors = o3d.utility.Vector3dVector(rgba[:, :3])

    o3d.visualization.draw_geometries([colored], window_name=window_name)
