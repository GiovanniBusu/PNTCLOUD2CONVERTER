"""Recentring: Swiss coordinates (MN95/LV95) are several million meters from
the origin. float32 at that magnitude has ~0.1-1m precision, which shows up
as visible jitter/z-fighting in SuperSplat's WebGL renderer. We subtract a
constant offset before export and stash it in a sidecar JSON so the splat
cloud can be geo-referenced again later if needed.
"""

from __future__ import annotations

import json
import os

import numpy as np


def compute_offset(points: np.ndarray, mode: str = "centroid") -> np.ndarray:
    if mode == "centroid":
        return points.mean(axis=0)
    if mode == "bbox_min":
        return points.min(axis=0)
    raise ValueError(f"center_mode inconnu : {mode}")


def recenter(points: np.ndarray, mode: str = "centroid") -> tuple[np.ndarray, np.ndarray]:
    offset = compute_offset(points, mode)
    return points - offset, offset


def convert_z_up_to_y_up(vectors: np.ndarray) -> np.ndarray:
    """Right-handed Z-up -> Y-up remap: (x, y, z) -> (x, z, -y).

    Geomatics/BIM/ReCap sources are almost always Z-up (Z = elevation);
    PlayCanvas/SuperSplat, like most realtime engines, is Y-up. Without
    this, a level scan renders tipped onto its side. Works for both
    positions and normals -- it's a pure rotation, no translation.
    """
    out = np.empty_like(vectors)
    out[:, 0] = vectors[:, 0]
    out[:, 1] = vectors[:, 2]
    out[:, 2] = -vectors[:, 1]
    return out


def write_offset_sidecar(
    ply_output_path: str, offset: np.ndarray, mode: str, axis_converted: bool = False
) -> str:
    sidecar_path = os.path.splitext(ply_output_path)[0] + ".offset.json"
    if axis_converted:
        note = (
            "Le PLY est en Y-up (pour SuperSplat/PlayCanvas) alors que la source etait Z-up. "
            "Pour retrouver les coordonnees monde d'origine (Z-up) a partir d'un point "
            "(px,py,pz) du PLY : x = px + offset.x ; y = -pz + offset.y ; z = py + offset.z."
        )
    else:
        note = (
            "Coordonnees locales du PLY = coordonnees monde - offset. "
            "Ajouter cet offset aux positions x,y,z du PLY pour retrouver "
            "les coordonnees monde d'origine (ex. MN95/LV95)."
        )
    payload = {
        "x": float(offset[0]),
        "y": float(offset[1]),
        "z": float(offset[2]),
        "mode": mode,
        "axis_convention": "z_up_source_converted_to_y_up_ply" if axis_converted else "unchanged",
        "note": note,
    }
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return sidecar_path
