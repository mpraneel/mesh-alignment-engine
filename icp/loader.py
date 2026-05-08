from pathlib import Path

import numpy as np
import open3d as o3d


def load_point_cloud(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix not in {".ply", ".stl"}:
        raise ValueError(f"unsupported extension {suffix!r}; expected .ply or .stl")

    if suffix == ".ply":
        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points)
        if points.size > 0:
            return points

    mesh = o3d.io.read_triangle_mesh(str(path))
    points = np.asarray(mesh.vertices)
    if points.size == 0:
        raise ValueError(f"no points or vertices found in {path}")
    return points
