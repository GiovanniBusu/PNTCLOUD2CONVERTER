import os

import numpy as np
import pytest
from plyfile import PlyData

from splatconv.config import SplatParams
from splatconv.errors import EmptyPointCloudError, RcpNotSupportedError, UnsupportedFormatError
from splatconv.io.readers import load_point_cloud
from splatconv.pipeline import convert_point_cloud


def _write_xyz_cube(path: str, points_per_edge: int = 10, offset=(0.0, 0.0, 0.0)) -> int:
    lin = np.linspace(-1, 1, points_per_edge)
    u, v = np.meshgrid(lin, lin)
    u, v = u.ravel(), v.ravel()
    ones = np.ones_like(u)
    faces = [
        np.column_stack([ones, u, v]),
        np.column_stack([-ones, u, v]),
        np.column_stack([u, ones, v]),
        np.column_stack([u, -ones, v]),
        np.column_stack([u, v, ones]),
        np.column_stack([u, v, -ones]),
    ]
    points = np.concatenate(faces) + np.array(offset)
    colors = np.tile([200, 50, 50], (points.shape[0], 1))
    with open(path, "w", encoding="utf-8") as f:
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
    return points.shape[0]


def test_full_pipeline_on_synthetic_cube(tmp_path):
    xyz_path = tmp_path / "cube.xyz"
    n_source = _write_xyz_cube(str(xyz_path))

    out_path = tmp_path / "cube.splat.ply"
    params = SplatParams(k_neighbors=8)
    result = convert_point_cloud(str(xyz_path), str(out_path), params)

    assert result.n_points_source == n_source
    assert result.n_splats == n_source
    assert os.path.isfile(result.output_path)
    assert os.path.isfile(result.offset_sidecar_path)
    assert result.file_size_bytes > 0

    ply = PlyData.read(str(out_path))
    v = ply["vertex"]
    assert v.count == n_source
    normals = np.stack([v["nx"], v["ny"], v["nz"]], axis=1)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-3)
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    np.testing.assert_allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-5)


def test_recentring_keeps_coordinates_small(tmp_path):
    xyz_path = tmp_path / "swiss_cube.xyz"
    swiss_offset = (2_600_000.0, 1_200_000.0, 500.0)
    _write_xyz_cube(str(xyz_path), offset=swiss_offset)

    out_path = tmp_path / "swiss.splat.ply"
    result = convert_point_cloud(str(xyz_path), str(out_path), SplatParams(k_neighbors=8))

    v = PlyData.read(str(out_path))["vertex"]
    assert np.abs(v["x"]).max() < 10.0
    assert np.abs(v["y"]).max() < 10.0

    import json
    with open(result.offset_sidecar_path) as f:
        offset = json.load(f)
    assert offset["x"] == pytest.approx(swiss_offset[0], abs=1.0)


def test_pts_10col_with_normals_confirmed_recap_layout(tmp_path):
    # Confirmed against a real Autodesk ReCap export: X Y Z R G B I NX NY NZ.
    # The last 3 columns are genuine unit normals (verified: sum of squares ~= 1).
    pts_path = tmp_path / "cas.pts"
    pts_path.write_text(
        "2555142.910 1188513.712 467.116 154 163 123 124 -0.550 -0.824 -0.138\n"
        "2555148.879 1188504.469 467.040 161 35 27 64 -0.366 -0.922 0.126\n"
    )
    data = load_point_cloud(str(pts_path))
    assert data.colors is not None
    assert data.intensity is not None
    assert data.normals is not None
    np.testing.assert_allclose(data.colors[0], [154 / 255, 163 / 255, 123 / 255], atol=1e-3)
    np.testing.assert_allclose(data.normals[0], [-0.550, -0.824, -0.138], atol=1e-3)
    assert np.linalg.norm(data.normals[0]) == pytest.approx(1.0, abs=0.01)


def test_pts_9col_without_intensity(tmp_path):
    pts_path = tmp_path / "nine.xyz"
    pts_path.write_text("0 0 0 200 100 50 0 0 1\n" * 2)
    data = load_point_cloud(str(pts_path))
    assert data.intensity is None
    assert data.colors is not None
    assert data.normals is not None
    np.testing.assert_allclose(data.normals[0], [0, 0, 1], atol=1e-6)


def test_z_up_to_y_up_conversion_flattens_correct_axis(tmp_path):
    # A level "floor" in Z-up source coords (X,Y spread wide, Z barely varies).
    xyz_path = tmp_path / "floor.xyz"
    n_side = 10
    lines = []
    for i in range(n_side):
        for j in range(n_side):
            lines.append(f"{i * 0.5} {j * 0.5} {(i + j) * 0.001} 200 150 100")
    xyz_path.write_text("\n".join(lines) + "\n")

    out_path = tmp_path / "floor.splat.ply"
    result = convert_point_cloud(
        str(xyz_path), str(out_path), SplatParams(k_neighbors=8, convert_z_up_to_y_up=True)
    )
    v = PlyData.read(str(out_path))["vertex"]
    # the near-constant source Z should land in output Y (small spread there)
    assert (v["y"].max() - v["y"].min()) < 0.05
    assert (v["z"].max() - v["z"].min()) > 1.0

    import json
    with open(result.offset_sidecar_path) as f:
        offset = json.load(f)
    assert offset["axis_convention"] == "z_up_source_converted_to_y_up_ply"


def test_rcp_file_raises_clear_guidance(tmp_path):
    rcp_path = tmp_path / "scan.rcp"
    rcp_path.write_bytes(b"not a real rcp file")

    with pytest.raises(RcpNotSupportedError) as excinfo:
        convert_point_cloud(str(rcp_path), str(tmp_path / "out.ply"), SplatParams())
    assert "ReCap" in excinfo.value.message


def test_pts_7col_uses_recap_column_order(tmp_path):
    # Autodesk ReCap's documented .pts export order is X Y Z R G B I (RGB
    # right after XYZ, intensity last) -- not "X Y Z I R G B". Reading it
    # with the wrong offset shifts every channel by one column and renders
    # as a solid, wrong-hued tint instead of the real per-point color.
    pts_path = tmp_path / "recap.pts"
    pts_path.write_text(
        "2\n"
        "0.0 0.0 0.0 200 10 5 180\n"
        "1.0 0.0 0.0 5 200 10 90\n"
    )
    data = load_point_cloud(str(pts_path))
    assert data.colors is not None
    assert data.intensity is not None
    np.testing.assert_allclose(data.colors[0], [200 / 255, 10 / 255, 5 / 255], atol=1e-3)
    np.testing.assert_allclose(data.colors[1], [5 / 255, 200 / 255, 10 / 255], atol=1e-3)
    # intensity is the last column (180, 90) -> min-max normalized to [0, 1]
    assert data.intensity[0] > data.intensity[1]


def test_unsupported_extension_raises(tmp_path):
    bad_path = tmp_path / "scan.foobar"
    bad_path.write_text("nonsense")
    with pytest.raises(UnsupportedFormatError):
        convert_point_cloud(str(bad_path), str(tmp_path / "out.ply"), SplatParams())


def test_empty_file_raises(tmp_path):
    empty_path = tmp_path / "empty.xyz"
    empty_path.write_text("")
    with pytest.raises(Exception):
        convert_point_cloud(str(empty_path), str(tmp_path / "out.ply"), SplatParams())


def test_voxel_downsample_reduces_point_count(tmp_path):
    xyz_path = tmp_path / "dense_cube.xyz"
    n_source = _write_xyz_cube(str(xyz_path), points_per_edge=20)

    out_path = tmp_path / "down.splat.ply"
    params = SplatParams(k_neighbors=8, voxel_size=0.3)
    result = convert_point_cloud(str(xyz_path), str(out_path), params)

    assert result.n_points_source == n_source
    assert result.n_splats < n_source
