"""Binary little-endian PLY writer for 3D Gaussian Splatting point clouds.

Property order matches the reference Inria 3DGS implementation (and the
viewers built on it, including SuperSplat):

    x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2,
    [f_rest_0 .. f_rest_44,]           <- only when include_f_rest=True
    opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3

Without photography there is no view-dependent information to encode, so by
default we write SH degree 0 only (no f_rest_*), which keeps files small.
Some older/stricter loaders expect the full 45 f_rest_* columns to be
present (zeroed) -- pass include_f_rest=True to fall back to that layout.
"""

from __future__ import annotations

import numpy as np
from plyfile import PlyData, PlyElement

N_SH_REST = 45  # (deg 1-3) - 3 channels x 15 coeffs, written as zeros in fallback mode


def build_vertex_array(
    positions: np.ndarray,
    normals: np.ndarray,
    f_dc: np.ndarray,
    opacity_logit: np.ndarray,
    log_scales: np.ndarray,
    rotations: np.ndarray,
    include_f_rest: bool = False,
) -> np.ndarray:
    """Assemble the structured numpy array written to the PLY 'vertex' element.

    All inputs are (N, k) float arrays (k depends on the field); everything
    is cast to float32, matching `property float ...` in the header.
    """
    n = positions.shape[0]
    for name, arr, k in (
        ("positions", positions, 3),
        ("normals", normals, 3),
        ("f_dc", f_dc, 3),
        ("log_scales", log_scales, 3),
        ("rotations", rotations, 4),
    ):
        if arr.shape != (n, k):
            raise ValueError(f"{name} doit avoir la forme ({n}, {k}), reçu {arr.shape}")
    if opacity_logit.shape != (n,):
        raise ValueError(f"opacity_logit doit avoir la forme ({n},), reçu {opacity_logit.shape}")

    fields = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    if include_f_rest:
        fields += [(f"f_rest_{i}", "f4") for i in range(N_SH_REST)]
    fields += [
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]

    vertex = np.zeros(n, dtype=fields)
    vertex["x"], vertex["y"], vertex["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
    vertex["nx"], vertex["ny"], vertex["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    # f_rest_* fields default to zero from np.zeros(); nothing more to do.
    vertex["opacity"] = opacity_logit
    vertex["scale_0"], vertex["scale_1"], vertex["scale_2"] = (
        log_scales[:, 0], log_scales[:, 1], log_scales[:, 2],
    )
    vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"] = (
        rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3],
    )
    return vertex


def write_ply(path: str, vertex_array: np.ndarray) -> None:
    """Write a binary little-endian PLY with a single 'vertex' element."""
    element = PlyElement.describe(vertex_array, "vertex")
    PlyData([element], text=False, byte_order="<").write(path)
