"""Per-point -> per-splat parameter computation.

Turns (position, color, normal) triples into the full 3DGS parameter set:
local anisotropic scale (flattened disk aligned to the normal), rotation
quaternion, opacity (denser where safe, softer in sparse areas), and the
degree-0 spherical harmonic color coefficient.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

SH_C0 = 0.28209479177387814  # Y_0^0, the standard 3DGS degree-0 SH constant


def local_mean_neighbor_distance(points: np.ndarray, k: int) -> np.ndarray:
    """Mean distance to the k nearest neighbors of each point."""
    n = points.shape[0]
    k = max(1, min(k, n - 1))
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1, workers=-1)
    return dists[:, 1:].mean(axis=1)


def compute_scales(mean_nn_dist: np.ndarray, scale_factor: float, anisotropy_ratio: float) -> np.ndarray:
    """Anisotropic log-scale per splat: (N, 3) = [tangent, tangent, normal-axis].

    The normal axis (scale_2, paired with the rotation below) is shrunk by
    `anisotropy_ratio` to flatten each splat into a disk-like surfel instead
    of a sphere.
    """
    tangent_radius = np.clip(mean_nn_dist * scale_factor, 1e-6, None)
    normal_radius = np.clip(tangent_radius * anisotropy_ratio, 1e-6, None)
    scales = np.column_stack([tangent_radius, tangent_radius, normal_radius])
    return np.log(scales).astype(np.float32)


def compute_rotations(normals: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) rotating local +Z onto each point's normal.

    scale_2 (the shrunk axis, see compute_scales) is expressed in the
    splat's *local* frame along local Z, so this rotation is what actually
    aligns the flattened axis with the surface normal in world space.
    """
    n = normals.shape[0]
    a = np.array([0.0, 0.0, 1.0])
    normals = normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)

    dot = normals @ a
    cross = np.cross(np.tile(a, (n, 1)), normals)
    w = 1.0 + dot
    quat = np.column_stack([w, cross])  # (N, 4) as (w, x, y, z)

    norm = np.linalg.norm(quat, axis=1)
    degenerate = norm < 1e-6  # normal ~= -Z: 180 degree rotation, axis choice is arbitrary
    quat[degenerate] = np.array([0.0, 1.0, 0.0, 0.0])
    norm[degenerate] = 1.0

    quat /= norm[:, None]
    return quat.astype(np.float32)


def compute_opacity_logit(
    mean_nn_dist: np.ndarray,
    opacity_dense: float,
    opacity_sparse: float,
) -> np.ndarray:
    """Opacity in [opacity_sparse, opacity_dense], reduced where points are
    locally sparse (large mean NN distance relative to the cloud median),
    stored in logit space (ln(a / (1 - a))) as required by the 3DGS format.
    """
    median = np.median(mean_nn_dist)
    if median < 1e-12:
        ratio = np.ones_like(mean_nn_dist)
    else:
        ratio = mean_nn_dist / median
    sparsity = np.clip(ratio - 1.0, 0.0, None)
    sparsity = sparsity / (1.0 + sparsity)  # smooth compression into [0, 1)

    opacity = opacity_dense - sparsity * (opacity_dense - opacity_sparse)
    opacity = np.clip(opacity, opacity_sparse, opacity_dense)
    return _logit(opacity).astype(np.float32)


def _logit(a: np.ndarray) -> np.ndarray:
    a = np.clip(a, 1e-6, 1 - 1e-6)
    return np.log(a / (1.0 - a))


def colors_to_sh_dc(colors: np.ndarray) -> np.ndarray:
    """RGB in [0, 1] -> degree-0 spherical harmonic DC coefficient."""
    return ((colors - 0.5) / SH_C0).astype(np.float32)
