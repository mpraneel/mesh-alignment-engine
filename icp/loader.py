from pathlib import Path

import open3d as o3d


def load_point_cloud(path: str | Path) -> o3d.geometry.PointCloud:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix not in {".ply", ".stl"}:
        raise ValueError(f"unsupported extension {suffix!r}; expected .ply or .stl")

    if suffix == ".ply":
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) > 0:
            if not pcd.has_normals():
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
                )
            return pcd

    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"no points or vertices found in {path}")

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals
    return pcd
