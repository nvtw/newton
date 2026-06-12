# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`newton._src.solvers.phoenx.helpers.array_access`.

The raw-pointer ``read2d_f32`` / ``write2d_f32`` helpers must be
bit-equivalent to Warp's bounds-checked ``arr[i, j]`` indexing for any
contiguous, row-major ``wp.array2d[wp.float32]``. We verify that with:

* a copy kernel that mirrors a host-initialised buffer via the
  helpers and compares against the source;
* a kernel that reads via the helper and writes via Warp's regular
  indexing (and vice versa) -- mismatched layout would surface as a
  diff against the NumPy reference;
* a kernel parameterised over dword offsets, mimicking the
  ``contact_container`` access pattern (column-major dword layout with
  inner k).

GPU-only -- inline-C ``wp.func_native`` snippets have no CPU fallback.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.helpers.array_access import read1d_i32, read2d_f32, write2d_f32


@wp.kernel(enable_backward=False)
def _copy_via_helpers_kernel(
    src: wp.array2d[wp.float32],
    dst: wp.array2d[wp.float32],
    rows: wp.int32,
    cols: wp.int32,
):
    """``dst[i, j] = src[i, j]`` for every in-range ``(i, j)``,
    routed through :func:`read2d_f32` / :func:`write2d_f32`."""
    tid = wp.tid()
    if tid >= rows * cols:
        return
    i = tid // cols
    j = tid - i * cols
    v = read2d_f32(src, i, j)
    write2d_f32(dst, i, j, v)


@wp.kernel(enable_backward=False)
def _cross_check_kernel(
    src: wp.array2d[wp.float32],
    dst_helper: wp.array2d[wp.float32],
    dst_indexed: wp.array2d[wp.float32],
    rows: wp.int32,
    cols: wp.int32,
):
    """Mirror ``src`` two ways. ``dst_helper`` is filled via the
    helpers; ``dst_indexed`` is filled via Warp's indexing. If the
    helpers respect the row stride, the buffers must be identical."""
    tid = wp.tid()
    if tid >= rows * cols:
        return
    i = tid // cols
    j = tid - i * cols

    write2d_f32(dst_helper, i, j, read2d_f32(src, i, j))
    dst_indexed[i, j] = src[i, j]


@wp.kernel(enable_backward=False)
def _dword_layout_kernel(
    buf: wp.array2d[wp.float32],
    dword_offset: wp.int32,
    n: wp.int32,
):
    """Write ``v = dword * 1000.0 + k`` to ``buf[dword_offset, k]``
    for every ``k < n``. Mimics the ``cc_set_*`` family which keys a
    fixed dword row by inner index ``k``."""
    k = wp.tid()
    if k >= n:
        return
    write2d_f32(buf, dword_offset, k, wp.float32(dword_offset) * 1000.0 + wp.float32(k))


@wp.kernel(enable_backward=False)
def _copy_i32_via_helper_kernel(
    src: wp.array[wp.int32],
    dst: wp.array[wp.int32],
    n: wp.int32,
):
    tid = wp.tid()
    if tid >= n:
        return
    dst[tid] = read1d_i32(src, tid)


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "array_access tests require CUDA (wp.func_native has no CPU fallback).",
)
class TestArrayHelpers(unittest.TestCase):
    """Bit-equivalence + layout checks for ``read2d_f32`` / ``write2d_f32``."""

    def setUp(self) -> None:
        self.device = wp.get_preferred_device()

    def _make_indexed_buffer(self, rows: int, cols: int) -> tuple[np.ndarray, wp.array]:
        """Return ``(host, device)`` for an ``[rows, cols]`` matrix
        whose ``[i, j]`` entry is ``i * 1000 + j`` -- a value that
        encodes both indices and so makes a stride bug visible."""
        host = np.fromfunction(
            lambda i, j: i.astype(np.float32) * 1000.0 + j.astype(np.float32),
            (rows, cols),
            dtype=np.float32,
        )
        dev = wp.array(host, dtype=wp.float32, device=self.device)
        return host, dev

    def test_copy_via_helpers_matches_source(self) -> None:
        """``read`` + ``write`` round-trip reproduces the source
        verbatim. Catches a wrong stride, off-by-one, or sign flip."""
        rows, cols = 9, 17  # non-power-of-2 to expose stride padding
        host, src = self._make_indexed_buffer(rows, cols)
        dst = wp.zeros((rows, cols), dtype=wp.float32, device=self.device)

        wp.launch(
            _copy_via_helpers_kernel,
            dim=rows * cols,
            inputs=[src, dst, rows, cols],
            device=self.device,
        )

        np.testing.assert_array_equal(dst.numpy(), host)

    def test_copy_i32_via_helper_matches_source(self) -> None:
        n = 257
        host = (np.arange(n, dtype=np.int32) * np.int32(17)) - np.int32(123)
        src = wp.array(host, dtype=wp.int32, device=self.device)
        dst = wp.zeros(n, dtype=wp.int32, device=self.device)

        wp.launch(
            _copy_i32_via_helper_kernel,
            dim=n,
            inputs=[src, dst, n],
            device=self.device,
        )

        np.testing.assert_array_equal(dst.numpy(), host)

    def test_helper_and_indexed_writes_agree(self) -> None:
        """Helper-written buffer must equal the buffer written via
        Warp's regular ``arr[i, j] = ...``."""
        rows, cols = 7, 33
        _, src = self._make_indexed_buffer(rows, cols)
        dst_helper = wp.zeros((rows, cols), dtype=wp.float32, device=self.device)
        dst_indexed = wp.zeros((rows, cols), dtype=wp.float32, device=self.device)

        wp.launch(
            _cross_check_kernel,
            dim=rows * cols,
            inputs=[src, dst_helper, dst_indexed, rows, cols],
            device=self.device,
        )

        np.testing.assert_array_equal(dst_helper.numpy(), dst_indexed.numpy())

    def test_dword_layout_matches_contact_container(self) -> None:
        """``contact_container`` keys writes by ``[dword, k]`` with k
        inner. Walk every dword row of a 21-row buffer and confirm
        the helper writes land in the expected slot."""
        dwords, n = 21, 64
        buf = wp.zeros((dwords, n), dtype=wp.float32, device=self.device)
        for off in range(dwords):
            wp.launch(
                _dword_layout_kernel,
                dim=n,
                inputs=[buf, off, n],
                device=self.device,
            )

        out = buf.numpy()
        expected = np.fromfunction(
            lambda d, k: d.astype(np.float32) * 1000.0 + k.astype(np.float32),
            (dwords, n),
            dtype=np.float32,
        )
        np.testing.assert_array_equal(out, expected)

    def test_partial_row_write_leaves_neighbours_untouched(self) -> None:
        """Writing every ``k`` for one ``dword`` row must NOT touch
        adjacent rows. Verifies the ``i * row_stride`` term, not just
        ``j``."""
        dwords, n = 21, 32
        buf = wp.zeros((dwords, n), dtype=wp.float32, device=self.device)
        target_dword = 7

        wp.launch(
            _dword_layout_kernel,
            dim=n,
            inputs=[buf, target_dword, n],
            device=self.device,
        )

        out = buf.numpy()
        for d in range(dwords):
            row = out[d]
            if d == target_dword:
                expected = np.arange(n, dtype=np.float32) + target_dword * 1000.0
                np.testing.assert_array_equal(row, expected)
            else:
                self.assertTrue(
                    np.all(row == 0.0),
                    msg=f"row {d} was mutated by writes targeted at row {target_dword}",
                )


if __name__ == "__main__":
    unittest.main()
