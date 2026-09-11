"""Tunable parameters for the surface-splat generation pipeline.

Defaults follow the values discussed in the project brief. They are meant to
be adjusted per-scan from the CLI (`--scale-factor`, ...) or the web UI, not
hard-coded truths.
"""

from dataclasses import dataclass


@dataclass
class SplatParams:
    # --- Neighborhood / scale ---
    k_neighbors: int = 10  # k used both for mean-NN-distance and normal estimation
    scale_factor: float = 1.5  # multiplier applied to mean NN distance (hole filling)
    anisotropy_ratio: float = 0.3  # normal-axis radius = tangent radius * this ratio

    # --- Opacity ---
    opacity_dense: float = 0.97  # opacity used in average/dense areas
    opacity_sparse: float = 0.5  # opacity floor used in the sparsest areas

    # --- Recentring ---
    center_mode: str = "centroid"  # "centroid" or "bbox_min"

    # --- Axis convention ---
    # Geomatics/BIM/ReCap sources are almost always Z-up; SuperSplat/PlayCanvas
    # is Y-up. True by default so a level scan doesn't render tipped on its side.
    convert_z_up_to_y_up: bool = True

    # --- Downsampling ---
    voxel_size: float | None = None  # None/0 disables voxel downsampling

    # --- Color model ---
    include_f_rest: bool = False  # write 45 zeroed f_rest_* (SH deg 1-3) for compat
    # SuperSplat/most PBR renderers treat the DC color term as linear light and
    # apply their own linear->sRGB display conversion; source RGB is sRGB, so
    # convert it to linear first or colors render oversaturated/blown out.
    srgb_to_linear: bool = True

    # --- Safety ---
    max_points_without_voxel: int = 30_000_000
    splat_count_warning_threshold: int = 8_000_000

    def validate(self) -> None:
        if self.k_neighbors < 3:
            raise ValueError("k_neighbors doit être >= 3")
        if self.scale_factor <= 0:
            raise ValueError("scale_factor doit être > 0")
        if not (0 < self.anisotropy_ratio <= 1):
            raise ValueError("anisotropy_ratio doit être dans ]0, 1]")
        if not (0 < self.opacity_sparse <= self.opacity_dense < 1):
            raise ValueError("il faut 0 < opacity_sparse <= opacity_dense < 1")
        if self.center_mode not in ("centroid", "bbox_min"):
            raise ValueError("center_mode doit être 'centroid' ou 'bbox_min'")
        if self.voxel_size is not None and self.voxel_size < 0:
            raise ValueError("voxel_size doit être >= 0")
