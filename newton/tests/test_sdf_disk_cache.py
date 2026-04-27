# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the on-disk cooked-SDF cache.

The temp-directory convention here mirrors Warp's
``tests/cuda/test_conditional_captures.test_graph_debug_dot_print`` —
files are placed directly under ``tempfile.gettempdir()`` rather than a
``TemporaryDirectory`` context. CI environments are reliable about the
system temp dir but can be flaky about other locations, so this keeps
behaviour identical to the upstream pattern.
"""

import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import warp as wp

from newton import Mesh
from newton._src.geometry import _sdf_cache
from newton._src.geometry.sdf_texture import (
    QuantizationMode,
    TextureSDFData,
    texture_sample_sdf,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

_cuda_available = wp.is_cuda_available()


def _make_cache_dir(tag: str) -> Path:
    """Create a fresh, uniquely-named cache directory under ``$TMPDIR``.

    Using ``tempfile.gettempdir()`` matches the Warp test convention for
    artifacts that need a writable, well-known temp location on CI.
    A short uuid suffix isolates parallel test workers.
    """

    base = Path(tempfile.gettempdir()) / f"newton_sdf_cache_test_{tag}_{uuid.uuid4().hex[:8]}"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _remove_cache_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _make_box_mesh() -> Mesh:
    hx = hy = hz = 0.5
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            2, 3, 7, 2, 7, 6,
            0, 4, 7, 0, 7, 3,
            1, 2, 6, 1, 6, 5,
        ],
        dtype=np.int32,
    )
    return Mesh(vertices, indices)


def _common_hash_kwargs(vertices: np.ndarray, indices: np.ndarray) -> dict:
    return {
        "vertices": vertices,
        "indices": indices,
        "is_solid": True,
        "narrow_band_range": (-0.1, 0.1),
        "target_voxel_size": None,
        "max_resolution": 64,
        "margin": 0.05,
        "texture_format": "uint16",
        "sign_method_resolved": "parity",
        "winding_threshold": 0.5,
        "scale": None,
    }


@wp.kernel
def _sample_sdf_kernel(
    sdf: TextureSDFData,
    points: wp.array[wp.vec3],
    out: wp.array[float],
) -> None:
    tid = wp.tid()
    out[tid] = texture_sample_sdf(sdf, points[tid])


def _sample(sdf: TextureSDFData, points_np: np.ndarray, device: str) -> np.ndarray:
    pts = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    out = wp.zeros(points_np.shape[0], dtype=float, device=device)
    wp.launch(_sample_sdf_kernel, dim=points_np.shape[0], inputs=[sdf, pts, out], device=device)
    return out.numpy()


# -----------------------------------------------------------------------------
# Hash + serialization tests (no GPU required)
# -----------------------------------------------------------------------------


class TestSDFDiskCachePure(unittest.TestCase):
    """Tests that exercise hashing and on-disk format only."""

    def setUp(self) -> None:
        self.mesh = _make_box_mesh()
        self.vertices = np.asarray(self.mesh.vertices, dtype=np.float32)
        self.indices = np.asarray(self.mesh.indices, dtype=np.int32).reshape(-1)
        self.cache_dir = _make_cache_dir(self._testMethodName)

    def tearDown(self) -> None:
        _remove_cache_dir(self.cache_dir)

    def test_hash_is_stable(self) -> None:
        kwargs = _common_hash_kwargs(self.vertices, self.indices)
        h1, _ = _sdf_cache.hash_inputs(**kwargs)
        h2, _ = _sdf_cache.hash_inputs(**kwargs)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 32)

    def test_hash_changes_with_params(self) -> None:
        base = _sdf_cache.hash_inputs(**_common_hash_kwargs(self.vertices, self.indices))[0]

        sensitive = [
            ("narrow_band_range", (-0.2, 0.2)),
            ("target_voxel_size", 0.05),
            ("max_resolution", 32),
            ("margin", 0.1),
            ("texture_format", "float32"),
            ("sign_method_resolved", "winding"),
            ("winding_threshold", -0.5),
            ("scale", (2.0, 1.0, 1.0)),
            ("is_solid", False),
        ]
        for name, value in sensitive:
            kwargs = _common_hash_kwargs(self.vertices, self.indices)
            kwargs[name] = value
            h, _ = _sdf_cache.hash_inputs(**kwargs)
            self.assertNotEqual(h, base, f"hash should differ when {name} changes")

    def test_hash_changes_with_mesh(self) -> None:
        base = _sdf_cache.hash_inputs(**_common_hash_kwargs(self.vertices, self.indices))[0]

        moved = self.vertices.copy()
        moved[0, 0] += 0.1
        kwargs = _common_hash_kwargs(moved, self.indices)
        h, _ = _sdf_cache.hash_inputs(**kwargs)
        self.assertNotEqual(h, base)

    def test_hash_winding_threshold_sign_only(self) -> None:
        kwargs = _common_hash_kwargs(self.vertices, self.indices)
        kwargs["winding_threshold"] = 0.5
        h_pos_a, _ = _sdf_cache.hash_inputs(**kwargs)
        kwargs["winding_threshold"] = 0.7
        h_pos_b, _ = _sdf_cache.hash_inputs(**kwargs)
        self.assertEqual(h_pos_a, h_pos_b, "hash must be insensitive to winding_threshold magnitude")

        kwargs["winding_threshold"] = -0.5
        h_neg, _ = _sdf_cache.hash_inputs(**kwargs)
        self.assertNotEqual(h_pos_a, h_neg, "hash must reflect winding_threshold sign")

    def _fake_sparse_data(self) -> dict:
        """A minimal but schema-correct sparse_data dict for round-trip tests."""

        return {
            "coarse_sdf": np.zeros((4, 4, 4), dtype=np.float32),
            "subgrid_data": np.zeros((1, 1, 1), dtype=np.float32),
            "subgrid_start_slots": np.zeros((2, 2, 2), dtype=np.uint32),
            "subgrid_required": np.zeros(8, dtype=np.int32),
            "subgrid_occupied": np.zeros(8, dtype=np.int32),
            "coarse_dims": (2, 2, 2),
            "subgrid_tex_size": 1,
            "num_subgrids": 0,
            "min_extents": np.array([-0.5, -0.5, -0.5], dtype=np.float64),
            "max_extents": np.array([0.5, 0.5, 0.5], dtype=np.float64),
            "cell_size": np.array([0.25, 0.25, 0.25], dtype=np.float64),
            "subgrid_size": 8,
            "quantization_mode": int(QuantizationMode.UINT16),
            "subgrids_min_sdf_value": 0.0,
            "subgrids_sdf_value_range": 1.0,
        }

    def test_round_trip_save_and_load(self) -> None:
        sparse_data = self._fake_sparse_data()
        tmp = self.cache_dir
        kwargs = _common_hash_kwargs(self.vertices, self.indices)
        h, key_inputs = _sdf_cache.hash_inputs(**kwargs)
        _sdf_cache.save_sparse_data(tmp, h, sparse_data, key_inputs=key_inputs, newton_version="test")
        npz_path, json_path = _sdf_cache.cache_paths(tmp, h)
        self.assertTrue(npz_path.exists())
        self.assertTrue(json_path.exists())

        loaded = _sdf_cache.try_load_sparse_data(tmp, h)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded["coarse_sdf"], sparse_data["coarse_sdf"])
        np.testing.assert_array_equal(loaded["subgrid_start_slots"], sparse_data["subgrid_start_slots"])
        self.assertEqual(loaded["coarse_dims"], sparse_data["coarse_dims"])
        self.assertEqual(loaded["subgrid_size"], sparse_data["subgrid_size"])
        self.assertEqual(loaded["quantization_mode"], sparse_data["quantization_mode"])

    def test_sidecar_documents_arrays(self) -> None:
        sparse_data = self._fake_sparse_data()
        tmp = self.cache_dir
        kwargs = _common_hash_kwargs(self.vertices, self.indices)
        h, key_inputs = _sdf_cache.hash_inputs(**kwargs)
        _sdf_cache.save_sparse_data(tmp, h, sparse_data, key_inputs=key_inputs, newton_version="test")
        _, json_path = _sdf_cache.cache_paths(tmp, h)
        with open(json_path) as f:
            manifest = json.load(f)

        self.assertEqual(manifest["cache_format_version"], _sdf_cache.CACHE_FORMAT_VERSION)
        self.assertEqual(manifest["kind"], "newton.texture_sdf")
        present = {entry["name"] for entry in manifest["arrays_present"]}
        for required in (
            "__cache_format_version__",
            "coarse_sdf",
            "subgrid_data",
            "subgrid_start_slots",
            "subgrid_required",
            "subgrid_occupied",
        ):
            self.assertIn(required, present)
        self.assertIn("key_inputs", manifest)
        self.assertIn("hash", manifest)
        self.assertIn("newton_version", manifest)

    def test_missing_files_is_miss(self) -> None:
        self.assertIsNone(_sdf_cache.try_load_sparse_data(self.cache_dir, "deadbeef"))

    def test_corrupt_npz_is_miss(self) -> None:
        sparse_data = self._fake_sparse_data()
        tmp = self.cache_dir
        kwargs = _common_hash_kwargs(self.vertices, self.indices)
        h, key_inputs = _sdf_cache.hash_inputs(**kwargs)
        _sdf_cache.save_sparse_data(tmp, h, sparse_data, key_inputs=key_inputs, newton_version="test")
        npz_path, _ = _sdf_cache.cache_paths(tmp, h)
        npz_path.write_bytes(b"not an npz")
        self.assertIsNone(_sdf_cache.try_load_sparse_data(tmp, h))

    def test_missing_sidecar_is_miss(self) -> None:
        sparse_data = self._fake_sparse_data()
        tmp = self.cache_dir
        kwargs = _common_hash_kwargs(self.vertices, self.indices)
        h, key_inputs = _sdf_cache.hash_inputs(**kwargs)
        _sdf_cache.save_sparse_data(tmp, h, sparse_data, key_inputs=key_inputs, newton_version="test")
        _, json_path = _sdf_cache.cache_paths(tmp, h)
        json_path.unlink()
        self.assertIsNone(_sdf_cache.try_load_sparse_data(tmp, h))

    def test_embedded_version_mismatch_is_miss(self) -> None:
        sparse_data = self._fake_sparse_data()
        tmp = self.cache_dir
        kwargs = _common_hash_kwargs(self.vertices, self.indices)
        h, key_inputs = _sdf_cache.hash_inputs(**kwargs)
        _sdf_cache.save_sparse_data(tmp, h, sparse_data, key_inputs=key_inputs, newton_version="test")
        npz_path, _ = _sdf_cache.cache_paths(tmp, h)
        with np.load(npz_path) as npz:
            contents = {k: npz[k] for k in npz.files if k != "__cache_format_version__"}
        contents["__cache_format_version__"] = np.asarray(
            _sdf_cache.CACHE_FORMAT_VERSION + 999, dtype=np.int32
        )
        np.savez(npz_path, **contents)
        self.assertIsNone(_sdf_cache.try_load_sparse_data(tmp, h))


# -----------------------------------------------------------------------------
# End-to-end hit/miss test (CUDA required)
# -----------------------------------------------------------------------------


def test_disk_cache_hit_matches_live(test, device) -> None:
    """A cache hit must produce SDF samples matching a fresh cook."""

    mesh = _make_box_mesh()
    cache_path = _make_cache_dir("hit_matches_live")
    try:
        sdf_live = mesh.build_sdf(device=device, cache_dir=cache_path)
        cache_files = list(cache_path.glob("*.sdf.npz"))
        test.assertEqual(
            len(cache_files), 1, f"expected exactly one cache file, found {cache_files}"
        )
        sidecar_files = list(cache_path.glob("*.sdf.json"))
        test.assertEqual(len(sidecar_files), 1)

        rng = np.random.default_rng(seed=0)
        points = rng.uniform(-0.6, 0.6, size=(64, 3)).astype(np.float32)
        live_values = _sample(sdf_live.texture_data, points, device)

        mesh2 = _make_box_mesh()
        sdf_cached = mesh2.build_sdf(device=device, cache_dir=cache_path)
        cached_values = _sample(sdf_cached.texture_data, points, device)

        np.testing.assert_allclose(
            cached_values,
            live_values,
            rtol=1e-5,
            atol=1e-5,
            err_msg="cached SDF samples must match the freshly cooked SDF",
        )

        test.assertIsNotNone(sdf_cached.texture_block_coords)
        test.assertGreater(len(sdf_cached.texture_block_coords), 0)
    finally:
        _remove_cache_dir(cache_path)


def test_disk_cache_param_change_invalidates(test, device) -> None:
    """Different build parameters must produce different cache entries."""

    cache_path = _make_cache_dir("param_change")
    try:
        mesh = _make_box_mesh()
        mesh.build_sdf(device=device, cache_dir=cache_path, max_resolution=32)

        mesh2 = _make_box_mesh()
        mesh2.build_sdf(device=device, cache_dir=cache_path, max_resolution=64)

        files = sorted(cache_path.glob("*.sdf.npz"))
        test.assertEqual(
            len(files), 2,
            f"expected two distinct cache entries, found {[p.name for p in files]}",
        )
    finally:
        _remove_cache_dir(cache_path)


# -----------------------------------------------------------------------------
# Test class wiring
# -----------------------------------------------------------------------------


class TestSDFDiskCacheCuda(unittest.TestCase):
    pass


_cuda_devices = get_cuda_test_devices()
add_function_test(
    TestSDFDiskCacheCuda,
    "test_disk_cache_hit_matches_live",
    test_disk_cache_hit_matches_live,
    devices=_cuda_devices,
)
add_function_test(
    TestSDFDiskCacheCuda,
    "test_disk_cache_param_change_invalidates",
    test_disk_cache_param_change_invalidates,
    devices=_cuda_devices,
)


if __name__ == "__main__":
    unittest.main()
