#!/usr/bin/env python3
"""Generate a small synthetic colored point cloud (a cube) and run it through
the real conversion pipeline, producing a known-good .ply to sanity-check in
SuperSplat (https://superspl.at/editor) before trying a real scan.

Each face of the cube gets a distinct flat color so it's easy to tell, at a
glance in the viewer, whether orientation/colors survived the round trip.

Usage:
    python scripts/make_test_cube.py [--out test_data/test_cube.splat.ply] [--points-per-edge 25]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splatconv.config import SplatParams  # noqa: E402
from splatconv.io.readers import PointCloudData  # noqa: E402
from splatconv.pipeline import convert_point_cloud  # noqa: E402

FACE_COLORS = {
    "+x": (230, 60, 60),
    "-x": (60, 230, 230),
    "+y": (60, 230, 60),
    "-y": (230, 60, 230),
    "+z": (60, 60, 230),
    "-z": (230, 230, 60),
}


def make_cube_points(points_per_edge: int, half_extent: float) -> tuple[np.ndarray, np.ndarray]:
    lin = np.linspace(-half_extent, half_extent, points_per_edge)
    u, v = np.meshgrid(lin, lin)
    u, v = u.ravel(), v.ravel()
    ones = np.ones_like(u)

    faces = {
        "+x": np.column_stack([half_extent * ones, u, v]),
        "-x": np.column_stack([-half_extent * ones, u, v]),
        "+y": np.column_stack([u, half_extent * ones, v]),
        "-y": np.column_stack([u, -half_extent * ones, v]),
        "+z": np.column_stack([u, v, half_extent * ones]),
        "-z": np.column_stack([u, v, -half_extent * ones]),
    }

    all_points, all_colors = [], []
    for name, pts in faces.items():
        all_points.append(pts)
        color = np.array(FACE_COLORS[name], dtype=np.float64) / 255.0
        all_colors.append(np.tile(color, (pts.shape[0], 1)))

    return np.concatenate(all_points), np.concatenate(all_colors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="test_data/test_cube.splat.ply")
    parser.add_argument("--points-per-edge", type=int, default=25)
    parser.add_argument("--half-extent", type=float, default=1.0)
    parser.add_argument("--swiss-offset", action="store_true",
                         help="Shift the cube by a large Swiss-coordinate-like offset, to test recentring")
    args = parser.parse_args()

    points, colors = make_cube_points(args.points_per_edge, args.half_extent)
    if args.swiss_offset:
        points += np.array([2_600_000.0, 1_200_000.0, 500.0])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    xyz_path = os.path.splitext(args.out)[0] + ".source.xyz"
    rgb255 = np.round(colors * 255).astype(int)
    with open(xyz_path, "w", encoding="utf-8") as f:
        for p, c in zip(points, rgb255):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
    print(f"Nuage synthétique écrit : {xyz_path} ({points.shape[0]} points)")

    params = SplatParams(k_neighbors=8, scale_factor=1.5, voxel_size=None)

    def on_progress(label: str, fraction: float) -> None:
        print(f"\r[{fraction * 100:5.1f}%] {label:<35}", end="", flush=True)

    result = convert_point_cloud(xyz_path, args.out, params, progress_cb=on_progress)
    print()
    print(f"OK : {result.n_splats} splats -> {result.output_path} ({result.file_size_bytes} octets)")
    print("Chargez ce fichier dans https://superspl.at/editor pour vérifier l'affichage.")
    for w in result.warnings:
        print(f"Avertissement : {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
