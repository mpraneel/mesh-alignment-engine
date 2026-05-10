# Plan: SVD-based rigid transform solver

## Goal
Implement `icp/solver.py::svd_solve(source_pts, target_pts) -> (R, t)` in
pure NumPy (Kabsch / Umeyama) and verify it recovers the same transform
Open3D's point-to-plane ICP converges to, using the correspondences
Open3D itself produced.

This is the linear-algebra learning step in the build order — the SVD
math should become obvious rather than magical, so the function body
will be heavily commented step-by-step.

## Files touched
1. **New:** `icp/solver.py` — contains `svd_solve`.
2. **Edit:** `scripts/verify_icp_bunny.py` — after Open3D ICP returns,
   extract `result.correspondence_set`, slice source/target point
   arrays, call `svd_solve`, assert agreement with
   `result.transformation` within `1e-4`.

No new dependencies; numpy is already required.

## `svd_solve` design

Signature:
```python
def svd_solve(
    source_pts: np.ndarray,  # (N, 3)
    target_pts: np.ndarray,  # (N, 3)
) -> tuple[np.ndarray, np.ndarray]:  # R (3,3), t (3,)
```

Returns `R`, `t` such that `R @ source_pts.T + t[:, None] ≈ target_pts.T`
in the least-squares sense.

Validation: both arrays must be `(N, 3)` with matching `N >= 3`.
Raise `ValueError` otherwise — this is a real boundary (caller could
pass anything), so it gets a real check, not a fallback.

Steps (each gets a comment block in the source):

1. **Centroids.** `c_s = source_pts.mean(axis=0)`, same for target.
   Translating to the centroid removes the translation component so we
   can solve for rotation in isolation.
2. **Center both point sets.** `S = source_pts - c_s`, `T = target_pts - c_t`.
   Now we want the rotation that best maps `S` onto `T`.
3. **Covariance matrix.** `H = S.T @ T` — a 3×3 cross-covariance.
   This is the matrix whose SVD encodes the optimal rotation; the
   intuition is that `H` captures how the two centered clouds correlate
   along each pair of axes.
4. **SVD.** `U, _, Vt = np.linalg.svd(H)`. The singular values
   themselves are discarded — only the orthonormal frames matter for
   the rotation.
5. **Reflection check.** Naively `R = Vt.T @ U.T` can come out as a
   reflection (det = -1) when the point sets are noisy or nearly
   coplanar. Compute `d = sign(det(Vt.T @ U.T))` and inject
   `diag(1, 1, d)` between `Vt.T` and `U.T` so the result is guaranteed
   to be a proper rotation (det = +1).
6. **Recover translation.** `t = c_t - R @ c_s`. Falls straight out of
   the centroid equation: the rotation maps the source centroid to the
   target centroid up to a translation.

No early returns, no try/except — this is internal code on validated
inputs, so a bad SVD result should surface as the numpy error.

## Smoke-test extension

Inside `scripts/verify_icp_bunny.py`, after the existing
`align_point_to_plane` call:

1. Pull `corr = np.asarray(result.correspondence_set)` — shape `(M, 2)`
   of `(source_idx, target_idx)` pairs.
2. Slice point arrays:
   ```python
   src_xyz = np.asarray(source.points)[corr[:, 0]]
   tgt_xyz = np.asarray(target.points)[corr[:, 1]]
   ```
3. Call `R_svd, t_svd = svd_solve(src_xyz, tgt_xyz)`.
4. Compare against Open3D's transform:
   ```python
   R_o3d = T_recovered[:3, :3]
   t_o3d = T_recovered[:3, 3]
   assert np.allclose(R_svd, R_o3d, atol=1e-4)
   assert np.allclose(t_svd, t_o3d, atol=1e-4)
   ```
5. Print `||R_svd - R_o3d||_F` and `||t_svd - t_o3d||` so a future
   reader sees how close the two solvers actually land.

### Why this comparison is meaningful (and the one wrinkle)

Open3D ran *point-to-plane* ICP; `svd_solve` is the *point-to-point*
closed-form solution. These minimize different cost functions, so in
general they would not produce identical transforms even on identical
correspondences.

For this test it still works because the bunny is being aligned against
a rigidly-transformed copy of itself: at convergence the
correspondences are essentially noise-free, so both cost functions are
minimized by the same rigid transform. `1e-4` is loose enough to absorb
the small disagreement from any leftover residual, tight enough to fail
loudly if the SVD math is wrong.

If we later test against a noisier real-world pair, this assertion
would need to be relaxed — noting it here so future-me doesn't copy the
pattern blindly.

## Out of scope
- Weighted Umeyama (per-correspondence weights). Not needed for the
  current build step; add only when the heatmap / outlier-rejection
  work calls for it.
- A scale factor (the full Umeyama). Rigid-only is what ICP expects.
- A standalone unit test under `tests/`. The bunny smoke test is the
  verification for now; a `tests/` file can come in a later step if/when
  we have more solver code to cover.

## Acceptance
- `python scripts/verify_icp_bunny.py` exits 0.
- Both new `np.allclose` assertions pass.
- The existing residual-vs-known-offset check still passes.
