import numpy as np
from plyfile import PlyData

from splatconv.ply_writer import build_vertex_array, write_ply

EXPECTED_PROPERTIES = [
    "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
]

EXPECTED_PROPERTIES_WITH_F_REST = (
    EXPECTED_PROPERTIES[:9]
    + [f"f_rest_{i}" for i in range(45)]
    + EXPECTED_PROPERTIES[9:]
)


def _dummy_splats(n: int = 5):
    rng = np.random.default_rng(0)
    positions = rng.normal(size=(n, 3)).astype(np.float32)
    normals = rng.normal(size=(n, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    f_dc = rng.normal(size=(n, 3)).astype(np.float32)
    opacity = rng.normal(size=n).astype(np.float32)
    log_scales = rng.normal(size=(n, 3)).astype(np.float32)
    rotations = rng.normal(size=(n, 4)).astype(np.float32)
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    return positions, normals, f_dc, opacity, log_scales, rotations


def test_property_order_matches_3dgs_spec(tmp_path):
    positions, normals, f_dc, opacity, log_scales, rotations = _dummy_splats()
    vertex_array = build_vertex_array(positions, normals, f_dc, opacity, log_scales, rotations)

    out_path = tmp_path / "out.ply"
    write_ply(str(out_path), vertex_array)

    ply = PlyData.read(str(out_path))
    assert ply.text is False, "PLY must be written binary, not ASCII"
    props = [p.name for p in ply["vertex"].properties]
    assert props == EXPECTED_PROPERTIES


def test_include_f_rest_inserts_45_zeroed_columns(tmp_path):
    positions, normals, f_dc, opacity, log_scales, rotations = _dummy_splats()
    vertex_array = build_vertex_array(
        positions, normals, f_dc, opacity, log_scales, rotations, include_f_rest=True
    )
    out_path = tmp_path / "out.ply"
    write_ply(str(out_path), vertex_array)

    ply = PlyData.read(str(out_path))
    props = [p.name for p in ply["vertex"].properties]
    assert props == EXPECTED_PROPERTIES_WITH_F_REST
    for i in range(45):
        assert np.all(ply["vertex"][f"f_rest_{i}"] == 0.0)


def test_values_round_trip(tmp_path):
    positions, normals, f_dc, opacity, log_scales, rotations = _dummy_splats()
    vertex_array = build_vertex_array(positions, normals, f_dc, opacity, log_scales, rotations)
    out_path = tmp_path / "out.ply"
    write_ply(str(out_path), vertex_array)

    v = PlyData.read(str(out_path))["vertex"]
    np.testing.assert_allclose(v["x"], positions[:, 0], atol=1e-6)
    np.testing.assert_allclose(v["rot_3"], rotations[:, 3], atol=1e-6)
    np.testing.assert_allclose(v["opacity"], opacity, atol=1e-6)


def test_shape_mismatch_raises():
    import pytest

    positions = np.zeros((5, 3), dtype=np.float32)
    bad_normals = np.zeros((4, 3), dtype=np.float32)
    f_dc = np.zeros((5, 3), dtype=np.float32)
    opacity = np.zeros(5, dtype=np.float32)
    log_scales = np.zeros((5, 3), dtype=np.float32)
    rotations = np.zeros((5, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        build_vertex_array(positions, bad_normals, f_dc, opacity, log_scales, rotations)
