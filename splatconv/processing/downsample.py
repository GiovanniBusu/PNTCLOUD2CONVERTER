"""Optional voxel downsampling, used to cap point count / memory before
splat generation on very large clouds.

Uses Open3D's `voxel_down_sample` when available (averages points and
colors per occupied voxel); falls back to an equivalent numpy
implementation (group by voxel index, average position and color) so the
feature works without the Open3D dependency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def voxel_downsample(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    voxel_size: float,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if voxel_size is None or voxel_size <= 0:
        return points, colors

    try:
        import open3d as o3d
    except ImportError:
        return _voxel_downsample_numpy(points, colors, voxel_size)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    down = pcd.voxel_down_sample(voxel_size)
    out_points = np.asarray(down.points, dtype=np.float64)
    out_colors = np.asarray(down.colors, dtype=np.float32) if colors is not None else None
    return out_points, out_colors


def _voxel_downsample_numpy(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    voxel_size: float,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    voxel_idx = np.floor(points / voxel_size).astype(np.int64)
    # Collapse (i, j, k) triples into a single key for np.unique.
    _, inverse, counts = np.unique(voxel_idx, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)

    n_voxels = counts.shape[0]
    sum_points = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(sum_points, inverse, points)
    out_points = sum_points / counts[:, None]

    out_colors = None
    if colors is not None:
        sum_colors = np.zeros((n_voxels, 3), dtype=np.float64)
        np.add.at(sum_colors, inverse, colors)
        out_colors = (sum_colors / counts[:, None]).astype(np.float32)

    return out_points, out_colors


def suggest_voxel_size(points: np.ndarray, target_point_count: int) -> float:
    """Rough voxel size suggestion so the downsampled cloud has roughly
    `target_point_count` points, assuming a roughly uniform point density.
    """
    if points.shape[0] <= target_point_count:
        return 0.0
    bbox = points.max(axis=0) - points.min(axis=0)
    volume = max(float(np.prod(np.clip(bbox, 1e-6, None))), 1e-6)
    voxel_volume = volume / target_point_count
    return float(voxel_volume ** (1.0 / 3.0))
