# mesh-alignment-engine

ICP-based 3D mesh alignment pipeline in Python. Given two surface meshes
(e.g., two scans of the same object taken at different times), computes the
rigid transformation that best aligns them and visualizes residual surface
deviation as a heatmap.

Run `python align.py scan_a.ply scan_b.ply` and get back a 4x4 transform
matrix, RMSE score, and a colored deviation heatmap in under 10 seconds
on a standard mesh.

<img width="1920" height="1061" alt="ScreenCapture_2026-05-10-19-18-27" src="https://github.com/user-attachments/assets/de909647-1357-4164-bc88-0e4d19d05860" />



---

## Algorithm: Iterative Closest Point (ICP)


<img width="550" height="400" alt="icp_mesh_alignment_pipeline_static" src="https://github.com/user-attachments/assets/872fb73b-4102-4ef9-bfca-37e7c8206baf" />


ICP estimates a rigid transformation (rotation + translation) between two
point clouds by iterating three steps until convergence:

1. For each point in the source mesh, find its nearest neighbor in the
   target mesh
2. Compute the optimal rotation R and translation t minimizing mean squared
   distance between corresponding pairs via SVD:
   - Construct covariance matrix H from centered point clouds
   - Decompose: H = U * S * Vt
   - Optimal rotation: R = V * Ut
3. Apply the transform to the source mesh and repeat

Open3D's point-to-plane variant is used over naive point-to-point because
it leverages surface normals to constrain correction direction, converging
faster and more accurately on smooth surfaces.

---

## Stack

| Library | Role |
|---------|------|
| Open3D | Mesh I/O, ICP implementation, 3D visualization |
| NumPy | Transform math, SVD, covariance matrix construction |
| trimesh | Mesh loading and preprocessing fallback |
| matplotlib | Deviation heatmap colormap |

---

## Usage

```bash
pip install open3d numpy trimesh matplotlib

python align.py mesh_a.ply mesh_b.ply
python align.py mesh_a.ply mesh_b.ply --threshold 0.001
python align.py mesh_a.ply mesh_b.ply --export
```

Output:
```
RMSE:     3.135359e-17
fitness:  1.000000
transform (4x4):
      0.996195      0.087156      0.000000     -0.010398
     -0.087156      0.996195      0.000000     -0.004109
     -0.000000     -0.000000      1.000000      0.000000
      0.000000      0.000000      0.000000      1.000000
```

---

## MVP

- Load two `.ply` or `.stl` files
- Run Open3D point-to-plane ICP
- Output RMSE alignment error and 4x4 transform matrix
- Render side-by-side before/after view and colored deviation heatmap
- CLI with `--threshold` and `--export` flags

## Stretch goals

- Multi-resolution ICP (coarse to fine, faster convergence)
- Export aligned mesh as new `.ply`

---

## Test Meshes

- [Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/)
  (bunny, dragon) — standard benchmarks
- [Thingiverse](https://www.thingiverse.com/) — free STLs
- Programmatically generated offset meshes for controlled unit testing

---

## Status

In progress, Summer 2026. Part of a broader computational geometry and
spatial systems portfolio alongside
[cpp-convex-hull](https://github.com/mpraneel/cpp-convex-hull) and
[rrt-motion-planner](https://github.com/mpraneel/rrt-motion-planner).
