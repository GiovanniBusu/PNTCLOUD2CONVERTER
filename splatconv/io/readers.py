"""Point cloud readers for the supported input formats.

All readers converge on a single `PointCloudData` container so the rest of
the pipeline never has to know which format the data came from.

Supported natively:
    - .e57  (via pye57 / libE57Format)  -- primary target format
    - .pts, .xyz, .txt                  (via numpy, whitespace/comma separated)
    - .las, .laz                        (via laspy)

Explicitly rejected with guidance:
    - .rcp / .rcs (Autodesk ReCap) -- proprietary, undocumented binary format.
      See errors.RcpNotSupportedError for the exact message shown to the user.

Memory note: KNN-based normal estimation and local scale both require the
full point set to be resident for correctness, so these readers load an
entire scan into memory. For multi-scan .e57 files we do process scan-by-scan
(a natural chunk boundary), which keeps peak memory bounded by the largest
single scan rather than the whole file. There is no true out-of-core /
streaming reader here -- see README "Limites connues".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import numpy as np

from ..errors import CorruptFileError, EmptyPointCloudError, RcpNotSupportedError, UnsupportedFormatError

ProgressCallback = Optional[Callable[[str, float], None]]

SUPPORTED_EXTENSIONS = {".e57", ".pts", ".xyz", ".txt", ".las", ".laz"}
REJECTED_EXTENSIONS = {".rcp", ".rcs"}


@dataclass
class PointCloudData:
    """Raw point cloud extracted from a source file, before any processing."""

    points: np.ndarray  # (N, 3) float64
    colors: Optional[np.ndarray] = None  # (N, 3) float32 in [0, 1]
    intensity: Optional[np.ndarray] = None  # (N,) float32, normalized [0, 1]
    normals: Optional[np.ndarray] = None  # (N, 3) float32, unit vectors (rare on input)
    source_scans: int = 1
    warnings: list[str] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])


def detect_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in REJECTED_EXTENSIONS:
        raise RcpNotSupportedError(path)
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Format de fichier non supporté : '{ext}'.\n"
            f"Formats acceptés : {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    return ext


def load_point_cloud(path: str, progress_cb: ProgressCallback = None) -> PointCloudData:
    """Dispatch to the right reader based on file extension.

    Raises RcpNotSupportedError, UnsupportedFormatError, CorruptFileError,
    or EmptyPointCloudError -- all subclasses of SplatConvError with a
    ready-to-display `.message`.
    """
    if not os.path.isfile(path):
        raise CorruptFileError(f"Fichier introuvable : {path}")
    if os.path.getsize(path) == 0:
        raise CorruptFileError(f"Le fichier est vide : {path}")

    ext = detect_format(path)
    try:
        if ext == ".e57":
            data = _read_e57(path, progress_cb)
        elif ext in (".pts", ".xyz", ".txt"):
            data = _read_pts_xyz(path, progress_cb)
        elif ext in (".las", ".laz"):
            data = _read_las(path, progress_cb)
        else:  # pragma: no cover - guarded by detect_format above
            raise UnsupportedFormatError(f"Format non supporté : {ext}")
    except (RcpNotSupportedError, UnsupportedFormatError, CorruptFileError, EmptyPointCloudError):
        raise
    except Exception as exc:  # noqa: BLE001 - convert any parser crash into a clean error
        raise CorruptFileError(
            f"Impossible de lire le fichier '{os.path.basename(path)}' ({ext}) : "
            f"le fichier semble corrompu ou dans un format inattendu.\nDétail technique : {exc}"
        ) from exc

    if data.n_points == 0:
        raise EmptyPointCloudError(
            f"Aucun point exploitable trouvé dans '{os.path.basename(path)}'."
        )
    return data


def _report(progress_cb: ProgressCallback, label: str, fraction: float) -> None:
    if progress_cb is not None:
        progress_cb(label, fraction)


# ---------------------------------------------------------------------------
# .e57
# ---------------------------------------------------------------------------

def _read_e57(path: str, progress_cb: ProgressCallback) -> PointCloudData:
    try:
        import pye57
    except ImportError as exc:
        raise CorruptFileError(
            "Le module 'pye57' n'est pas installé (ou libE57Format est "
            "manquant sur le système). Voir README.md pour l'installation."
        ) from exc

    e57 = pye57.E57(path)
    scan_count = e57.scan_count
    all_points, all_colors, all_intensity = [], [], []
    warnings: list[str] = []
    have_colors = have_intensity = False

    for i in range(scan_count):
        _report(progress_cb, f"Lecture du scan {i + 1}/{scan_count}", i / max(scan_count, 1))
        try:
            scan = e57.read_scan(i, colors=True, intensity=True, ignore_missing_fields=True, transform=True)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Scan {i} illisible, ignoré ({exc})")
            continue

        if "cartesianX" not in scan:
            continue
        x = np.asarray(scan["cartesianX"], dtype=np.float64)
        y = np.asarray(scan["cartesianY"], dtype=np.float64)
        z = np.asarray(scan["cartesianZ"], dtype=np.float64)
        pts = np.column_stack([x, y, z])

        # Drop cartesianInvalidState==0 points if that field is present.
        if "cartesianInvalidState" in scan:
            valid = np.asarray(scan["cartesianInvalidState"]) == 0
            pts = pts[valid]
        else:
            valid = None

        all_points.append(pts)

        if "colorRed" in scan and "colorGreen" in scan and "colorBlue" in scan:
            have_colors = True
            r = np.asarray(scan["colorRed"], dtype=np.float64)
            g = np.asarray(scan["colorGreen"], dtype=np.float64)
            b = np.asarray(scan["colorBlue"], dtype=np.float64)
            rgb = np.column_stack([r, g, b])
            if valid is not None:
                rgb = rgb[valid]
            rgb = _normalize_color_range(rgb)
            all_colors.append(rgb)
        elif have_colors:
            # a previous scan had colors but this one doesn't: pad with gray
            all_colors.append(np.full((pts.shape[0], 3), 0.5, dtype=np.float32))

        if "intensity" in scan:
            have_intensity = True
            inten = np.asarray(scan["intensity"], dtype=np.float64)
            if valid is not None:
                inten = inten[valid]
            all_intensity.append(inten)
        elif have_intensity:
            all_intensity.append(np.zeros(pts.shape[0], dtype=np.float32))

    _report(progress_cb, "Lecture terminée", 1.0)

    if not all_points:
        raise EmptyPointCloudError("Aucun scan valide dans ce fichier .e57.")

    points = np.concatenate(all_points, axis=0)
    colors = _normalize_intensity(np.concatenate(all_intensity, axis=0)) if all_intensity else None
    intensity = colors
    colors = np.concatenate(all_colors, axis=0).astype(np.float32) if all_colors else None

    return PointCloudData(
        points=points,
        colors=colors,
        intensity=intensity,
        source_scans=scan_count,
        warnings=warnings,
    )


def _normalize_color_range(rgb: np.ndarray) -> np.ndarray:
    """E57 color range is per-file (often 0-255, sometimes 0-1 or 0-65535)."""
    max_val = float(rgb.max()) if rgb.size else 1.0
    if max_val <= 1.0 + 1e-6:
        scale = 1.0
    elif max_val <= 255.0 + 1e-6:
        scale = 255.0
    else:
        scale = 65535.0
    return np.clip(rgb / scale, 0.0, 1.0).astype(np.float32)


def _normalize_intensity(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    lo, hi = np.percentile(values, [1, 99])
    if hi - lo < 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# .pts / .xyz / .txt
# ---------------------------------------------------------------------------

def _read_pts_xyz(path: str, progress_cb: ProgressCallback) -> PointCloudData:
    _report(progress_cb, "Lecture du fichier texte", 0.05)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()

    skip_header = 0
    # Classic .pts format: first line is a single integer point count.
    if first_line.replace(" ", "").isdigit():
        skip_header = 1

    try:
        arr = np.loadtxt(path, skiprows=skip_header, ndmin=2)
    except ValueError:
        # Fall back to comma-separated, or a file with a text header line.
        try:
            arr = np.loadtxt(path, skiprows=skip_header, delimiter=",", ndmin=2)
        except ValueError:
            arr = np.genfromtxt(path, skip_header=max(skip_header, 1), ndmin=2)

    _report(progress_cb, "Fichier chargé", 0.8)

    if arr.ndim != 2 or arr.shape[1] < 3:
        raise CorruptFileError(
            "Le fichier ne contient pas au moins 3 colonnes (X Y Z)."
        )

    n_cols = arr.shape[1]
    points = arr[:, 0:3].astype(np.float64)
    colors = None
    intensity = None

    # Common layouts:
    #   X Y Z                              (3 cols)
    #   X Y Z I                            (4 cols, .pts)
    #   X Y Z R G B                        (6 cols)
    #   X Y Z R G B I                      (7 cols -- Autodesk ReCap's documented
    #                                        .pts export order: RGB right after XYZ,
    #                                        intensity *last*. Do not swap this back
    #                                        to "X Y Z I R G B": that reads R/G/B one
    #                                        column short and renders as solid red.)
    #   X Y Z NX NY NZ R G B               (9 cols)
    normals = None
    if n_cols == 4:
        intensity = _normalize_intensity(arr[:, 3])
    elif n_cols == 6:
        colors = _normalize_color_range(arr[:, 3:6])
    elif n_cols == 7:
        colors = _normalize_color_range(arr[:, 3:6])
        intensity = _normalize_intensity(arr[:, 6])
    elif n_cols >= 9:
        normals = arr[:, 3:6].astype(np.float32)
        colors = _normalize_color_range(arr[:, 6:9])

    _report(progress_cb, "Lecture terminée", 1.0)
    return PointCloudData(points=points, colors=colors, intensity=intensity, normals=normals)


# ---------------------------------------------------------------------------
# .las / .laz
# ---------------------------------------------------------------------------

def _read_las(path: str, progress_cb: ProgressCallback) -> PointCloudData:
    try:
        import laspy
    except ImportError as exc:
        raise CorruptFileError(
            "Le module 'laspy' n'est pas installé. Voir README.md pour l'installation."
        ) from exc

    _report(progress_cb, "Lecture du fichier LAS/LAZ", 0.05)
    with laspy.open(path) as f:
        las = f.read()

    points = np.column_stack([las.x, las.y, las.z]).astype(np.float64)

    colors = None
    dims = set(las.point_format.dimension_names)
    if {"red", "green", "blue"}.issubset(dims):
        rgb = np.column_stack([las.red, las.green, las.blue]).astype(np.float64)
        colors = _normalize_color_range(rgb)

    intensity = None
    if "intensity" in dims:
        intensity = _normalize_intensity(np.asarray(las.intensity, dtype=np.float64))

    _report(progress_cb, "Lecture terminée", 1.0)
    return PointCloudData(points=points, colors=colors, intensity=intensity)
