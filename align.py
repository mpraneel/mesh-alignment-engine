"""CLI entry point: rigidly align mesh_a onto mesh_b with point-to-plane ICP."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from icp.loader import load_point_cloud
from icp.registration import align_point_to_plane


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="align",
        description="Rigidly align mesh_a onto mesh_b using point-to-plane ICP.",
    )
    p.add_argument("mesh_a", type=Path, help="source mesh (.ply or .stl)")
    p.add_argument("mesh_b", type=Path, help="target mesh (.ply or .stl)")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="max correspondence distance (default: 0.02)",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="max ICP iterations (default: 50)",
    )
    return p.parse_args(argv)


def format_matrix(T: np.ndarray) -> str:
    return "\n".join("  " + "  ".join(f"{v: 12.6f}" for v in row) for row in T)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.threshold <= 0:
        print(f"error: --threshold must be positive, got {args.threshold}", file=sys.stderr)
        return 1
    if args.max_iterations <= 0:
        print(
            f"error: --max-iterations must be positive, got {args.max_iterations}",
            file=sys.stderr,
        )
        return 1

    try:
        source = load_point_cloud(args.mesh_a)
        target = load_point_cloud(args.mesh_b)
        result = align_point_to_plane(
            source,
            target,
            max_correspondence_distance=args.threshold,
            max_iterations=args.max_iterations,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    T = np.asarray(result.transformation)
    print(f"source:   {args.mesh_a}")
    print(f"target:   {args.mesh_b}")
    print(f"RMSE:     {result.inlier_rmse:.6e}")
    print(f"fitness:  {result.fitness:.6f}")
    print("transform (4x4):")
    print(format_matrix(T))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
