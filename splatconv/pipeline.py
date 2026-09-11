"""Top-level orchestration: source file -> 3DGS PLY.

Shared by both the CLI (cli/convert.py) and the web backend
(backend/jobs.py) so the two entry points can never drift apart.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .config import SplatParams
from .errors import InsufficientMemoryError
from .io.readers import PointCloudData, load_point_cloud
from .ply_writer import build_vertex_array, write_ply
from .processing.downsample import voxel_downsample
from .processing.normals import estimate_normals
from .processing.recenter import convert_z_up_to_y_up, recenter, write_offset_sidecar
from .processing.splat import (
    colors_to_sh_dc,
    compute_opacity_logit,
    compute_rotations,
    compute_scales,
    local_mean_neighbor_distance,
)

ProgressCallback = Optional[Callable[[str, float], None]]

_STAGE_WEIGHTS = {
    "read": 0.30,
    "downsample": 0.05,
    "normals": 0.25,
    "scale": 0.15,
    "splats": 0.15,
    "write": 0.10,
}


@dataclass
class ConversionResult:
    output_path: str
    offset_sidecar_path: str
    n_points_source: int
    n_splats: int
    file_size_bytes: int
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)


def _progress(cb: ProgressCallback, stage: str, stage_fraction: float, label: str) -> None:
    if cb is None:
        return
    base = sum(v for k, v in _STAGE_WEIGHTS.items() if list(_STAGE_WEIGHTS).index(k) < list(_STAGE_WEIGHTS).index(stage))
    overall = base + _STAGE_WEIGHTS[stage] * stage_fraction
    cb(label, min(overall, 1.0))


def convert_point_cloud(
    input_path: str,
    output_path: str,
    params: SplatParams,
    progress_cb: ProgressCallback = None,
) -> ConversionResult:
    params.validate()
    start = time.time()
    warnings: list[str] = []

    def read_progress(label: str, frac: float) -> None:
        _progress(progress_cb, "read", frac, label)

    data: PointCloudData = load_point_cloud(input_path, progress_cb=read_progress)
    warnings.extend(data.warnings)
    n_points_source = data.n_points

    if params.voxel_size in (None, 0) and n_points_source > params.max_points_without_voxel:
        raise InsufficientMemoryError(
            f"Le nuage contient {n_points_source:,} points, au-dessus du seuil de sécurité "
            f"({params.max_points_without_voxel:,}) pour un traitement sans sous-échantillonnage. "
            "Activez le sous-échantillonnage voxel (--voxel-size) pour continuer."
        )

    points = data.points
    colors = data.colors
    if colors is None:
        if data.intensity is not None:
            gray = np.repeat(data.intensity[:, None], 3, axis=1).astype(np.float32)
            colors = gray
            warnings.append("Pas de couleur RGB dans la source : intensité utilisée en niveaux de gris.")
        else:
            colors = np.full((n_points_source, 3), 0.5, dtype=np.float32)
            warnings.append("Pas de couleur ni d'intensité dans la source : gris neutre utilisé par défaut.")

    _progress(progress_cb, "downsample", 0.0, "Sous-échantillonnage")
    points, colors = voxel_downsample(points, colors, params.voxel_size or 0.0)
    _progress(progress_cb, "downsample", 1.0, "Sous-échantillonnage terminé")

    n_after_downsample = points.shape[0]
    if n_after_downsample == 0:
        raise InsufficientMemoryError("Le sous-échantillonnage a supprimé tous les points (voxel trop grand ?).")

    points, offset = recenter(points, mode=params.center_mode)
    if params.convert_z_up_to_y_up:
        points = convert_z_up_to_y_up(points)

    _progress(progress_cb, "normals", 0.0, "Estimation des normales")
    if data.normals is not None and data.normals.shape[0] == n_after_downsample:
        normals = data.normals
        if params.convert_z_up_to_y_up:
            normals = convert_z_up_to_y_up(normals)
    else:
        # points are already in the final (possibly Y-up) frame, so normals
        # estimated from them come out correctly oriented with no extra step.
        normals = estimate_normals(points, k=params.k_neighbors)
    _progress(progress_cb, "normals", 1.0, "Normales calculées")

    _progress(progress_cb, "scale", 0.0, "Calcul de l'échelle locale")
    mean_nn_dist = local_mean_neighbor_distance(points, k=params.k_neighbors)
    _progress(progress_cb, "scale", 1.0, "Échelle locale calculée")

    _progress(progress_cb, "splats", 0.0, "Génération des splats")
    log_scales = compute_scales(mean_nn_dist, params.scale_factor, params.anisotropy_ratio)
    rotations = compute_rotations(normals)
    opacity_logit = compute_opacity_logit(mean_nn_dist, params.opacity_dense, params.opacity_sparse)
    f_dc = colors_to_sh_dc(colors)
    _progress(progress_cb, "splats", 1.0, "Splats générés")

    n_splats = points.shape[0]
    if n_splats > params.splat_count_warning_threshold:
        warnings.append(
            f"{n_splats:,} splats générés : au-delà de ~{params.splat_count_warning_threshold:,}, "
            "le rendu WebGL dans SuperSplat peut devenir peu fluide sur un GPU standard. "
            "Envisagez un sous-échantillonnage voxel plus agressif."
        )

    _progress(progress_cb, "write", 0.0, "Écriture du fichier PLY")
    vertex_array = build_vertex_array(
        positions=points.astype(np.float32),
        normals=normals.astype(np.float32),
        f_dc=f_dc,
        opacity_logit=opacity_logit,
        log_scales=log_scales,
        rotations=rotations,
        include_f_rest=params.include_f_rest,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    write_ply(output_path, vertex_array)
    sidecar_path = write_offset_sidecar(
        output_path, offset, params.center_mode, axis_converted=params.convert_z_up_to_y_up
    )
    _progress(progress_cb, "write", 1.0, "Terminé")

    file_size = os.path.getsize(output_path)
    return ConversionResult(
        output_path=output_path,
        offset_sidecar_path=sidecar_path,
        n_points_source=n_points_source,
        n_splats=n_splats,
        file_size_bytes=file_size,
        elapsed_seconds=time.time() - start,
        warnings=warnings,
    )
