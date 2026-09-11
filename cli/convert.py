#!/usr/bin/env python3
"""Standalone CLI: convert a point cloud file into a 3DGS PLY for SuperSplat.

Usage:
    python cli/convert.py fichier.e57 --output splat.ply --voxel-size 0.02 --scale-factor 1.5

Run `python cli/convert.py --help` for the full parameter list.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splatconv.config import SplatParams  # noqa: E402
from splatconv.errors import SplatConvError  # noqa: E402
from splatconv.pipeline import convert_point_cloud  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convertit un nuage de points (.e57/.pts/.xyz/.las) en splat gaussien .ply pour SuperSplat.",
    )
    parser.add_argument("input", help="Fichier source (.e57, .pts, .xyz, .txt, .las, .laz)")
    parser.add_argument("-o", "--output", help="Fichier .ply de sortie (défaut : <input>.splat.ply)")

    neighborhood = parser.add_argument_group("Voisinage / échelle")
    neighborhood.add_argument("--k-neighbors", type=int, default=10, help="k pour normales + échelle locale (défaut : 10)")
    neighborhood.add_argument("--scale-factor", type=float, default=1.5, help="Multiplicateur de comblement de trous (défaut : 1.5)")
    neighborhood.add_argument("--anisotropy", type=float, default=0.3, help="Ratio rayon normal / rayon tangent, aplatissement en disque (défaut : 0.3)")

    opacity = parser.add_argument_group("Opacité")
    opacity.add_argument("--opacity-dense", type=float, default=0.97, help="Opacité en zone dense (défaut : 0.97)")
    opacity.add_argument("--opacity-sparse", type=float, default=0.5, help="Opacité plancher en zone clairsemée (défaut : 0.5)")

    misc = parser.add_argument_group("Divers")
    misc.add_argument("--voxel-size", type=float, default=None, help="Taille de voxel pour sous-échantillonnage (défaut : désactivé)")
    misc.add_argument("--center-mode", choices=["centroid", "bbox_min"], default="centroid", help="Mode de recentrage (défaut : centroid)")
    misc.add_argument("--include-f-rest", action="store_true", help="Écrire 45 coefficients f_rest_* à zéro (compat. lecteurs stricts SH deg 1-3)")
    misc.add_argument("--no-zup-convert", dest="zup_convert", action="store_false", default=True,
                       help="Ne pas convertir Z-up (géomatique/BIM/ReCap) vers Y-up (SuperSplat) — actif par défaut")
    misc.add_argument("--quiet", action="store_true", help="Ne pas afficher la barre de progression")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"Erreur : fichier introuvable : {args.input}", file=sys.stderr)
        return 1

    output = args.output or os.path.splitext(args.input)[0] + ".splat.ply"

    params = SplatParams(
        k_neighbors=args.k_neighbors,
        scale_factor=args.scale_factor,
        anisotropy_ratio=args.anisotropy,
        opacity_dense=args.opacity_dense,
        opacity_sparse=args.opacity_sparse,
        center_mode=args.center_mode,
        voxel_size=args.voxel_size,
        include_f_rest=args.include_f_rest,
        convert_z_up_to_y_up=args.zup_convert,
    )

    try:
        params.validate()
    except ValueError as exc:
        print(f"Erreur de paramètres : {exc}", file=sys.stderr)
        return 1

    def on_progress(label: str, fraction: float) -> None:
        if args.quiet:
            return
        bar_len = 30
        filled = int(bar_len * fraction)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r[{bar}] {fraction * 100:5.1f}%  {label:<40}", end="", flush=True)

    try:
        result = convert_point_cloud(args.input, output, params, progress_cb=on_progress)
    except SplatConvError as exc:
        if not args.quiet:
            print()
        print(f"\nErreur : {exc.message}", file=sys.stderr)
        return 2
    except MemoryError:
        if not args.quiet:
            print()
        print(
            "\nErreur : mémoire insuffisante pour traiter ce nuage. "
            "Essayez --voxel-size pour sous-échantillonner avant traitement.",
            file=sys.stderr,
        )
        return 3

    if not args.quiet:
        print()

    print(f"OK : {result.n_points_source:,} points source -> {result.n_splats:,} splats")
    print(f"Fichier PLY   : {result.output_path} ({_human_size(result.file_size_bytes)})")
    print(f"Offset sidecar: {result.offset_sidecar_path}")
    print(f"Durée         : {result.elapsed_seconds:.1f} s")
    for warning in result.warnings:
        print(f"Avertissement : {warning}")

    return 0


def _human_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("o", "Ko", "Mo", "Go"):
        if size < 1024 or unit == "Go":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} Go"


if __name__ == "__main__":
    sys.exit(main())
