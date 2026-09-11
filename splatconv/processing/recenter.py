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


def write_offset_sidecar(ply_output_path: str, offset: np.ndarray, mode: str) -> str:
    sidecar_path = os.path.splitext(ply_output_path)[0] + ".offset.json"
    payload = {
        "x": float(offset[0]),
        "y": float(offset[1]),
        "z": float(offset[2]),
        "mode": mode,
        "note": (
            "Coordonnees locales du PLY = coordonnees monde - offset. "
            "Ajouter cet offset aux positions x,y,z du PLY pour retrouver "
            "les coordonnees monde d'origine (ex. MN95/LV95)."
        ),
    }
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return sidecar_path
