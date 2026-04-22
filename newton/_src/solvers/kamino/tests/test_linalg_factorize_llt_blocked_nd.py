# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the LLTBlockedNDSolver.

These tests compare the ND-reordered semi-sparse blocked LLT solver to the
existing dense blocked LLT solver (:class:`LLTBlockedSolver`) on identical
inputs. Because the ND reordering and the tile-pattern-aware kernels are
supposed to be a transparent optimization (same result, different execution
plan), the two solvers must produce identical numerical outputs up to
round-off. The tests check:

- solution vector ``x`` matches the dense reference;
- ``||A x - b||`` residual is small;
- factorization accuracy: ``P A P^T == L L^T`` for the permuted system;
- the internal tile-pattern mask respects the symbolic-Cholesky inclusion
  rule (diagonal always set, and the pattern is at least as filled as the
  raw thresholded sparsity of the permuted matrix).

Test matrices cover both dense (``RandomProblemLLT``) and genuinely sparse
(banded, random-sparse, multi-block) cases so the tile mask is actually
exercised.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import float32
from newton._src.solvers.kamino._src.linalg.core import (
    DenseLinearOperatorData,
    DenseSquareMultiLinearInfo,
)
from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked_nd_solver import (
    LLTBlockedNDSolver,
)
from newton._src.solvers.kamino._src.linalg.linear import LLTBlockedSolver
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.extract import get_vector_block
from newton._src.solvers.kamino.tests.utils.rand import RandomProblemLLT


###
# Helpers: sparse SPD generators that match the flat-batched layout
###


def _make_banded_spd(n: int, bandwidth: int, seed: int, np_dtype=np.float32) -> np.ndarray:
    """Random banded symmetric SPD matrix with given bandwidth."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=np_dtype)
    for i in range(n):
        lo = max(0, i - bandwidth)
        hi = min(n, i + bandwidth + 1)
        for j in range(lo, hi):
            if i == j:
                A[i, j] = np_dtype(1.0 + bandwidth)
            elif i < j:
                v = np_dtype(0.1) * np_dtype(rng.standard_normal())
                A[i, j] = v
                A[j, i] = v
    return A


def _make_random_sparse_spd(n: int, density: float, seed: int, np_dtype=np.float32) -> np.ndarray:
    """Random sparse SPD of the form ``D + alpha * sum_k v_k v_k^T`` where the
    ``v_k`` are sparse - keeps the final pattern sparse-ish while guaranteeing
    SPD without a dense ``S S^T`` stage.
    """
    rng = np.random.default_rng(seed)
    num_rank1 = max(1, int(n * density * 2))  # a few sparse rank-1 updates
    A = np.eye(n, dtype=np_dtype) * np_dtype(2.0)
    for _ in range(num_rank1):
        v = np.zeros(n, dtype=np_dtype)
        support = rng.choice(n, size=max(2, int(n * density)), replace=False)
        v[support] = rng.standard_normal(len(support)).astype(np_dtype)
        A = A + np_dtype(0.5) * np.outer(v, v).astype(np_dtype)
    # Symmetrize to kill fp asymmetry.
    A = np_dtype(0.5) * (A + A.T)
    return A.astype(np_dtype)


def _pack_flat(A_blocks, b_blocks, device):
    """Pack per-block numpy arrays into the flat-batched (mio, vio, A, b) layout."""
    num_blocks = len(A_blocks)
    dims = [A.shape[0] for A in A_blocks]
    mat_sizes = [d * d for d in dims]
    vec_sizes = dims[:]
    mio = [0]
    for s in mat_sizes[:-1]:
        mio.append(mio[-1] + s)
    vio = [0]
    for s in vec_sizes[:-1]:
        vio.append(vio[-1] + s)

    A_flat = np.zeros(sum(mat_sizes), dtype=np.float32)
    b_flat = np.zeros(sum(vec_sizes), dtype=np.float32)
    for i in range(num_blocks):
        A_flat[mio[i] : mio[i] + mat_sizes[i]] = A_blocks[i].astype(np.float32).reshape(-1)
        b_flat[vio[i] : vio[i] + vec_sizes[i]] = b_blocks[i].astype(np.float32)

    A_wp = wp.array(A_flat, dtype=wp.float32, device=device)
    b_wp = wp.array(b_flat, dtype=wp.float32, device=device)
    return A_wp, b_wp, dims, mio, vio


###
# Test case
###


class TestLinAlgLLTBlockedND(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.seed = 42
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose

        if self.verbose:
            print("\n")
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

        self.block_size = 16
        # Loose tolerances to tolerate fp32 round-off differences between the
        # two numerically-equivalent code paths. The dense reference and the
        # ND solver multiply and accumulate in different orders due to the
        # permutation and the tile-skip logic, so bit-identical match is not
        # guaranteed.
        self.rtol = 1e-3
        self.atol = 1e-4

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    ###
    # Internal helpers
    ###

    def _run_reference_solver(self, operator, A_wp, b_wp, x_wp):
        """Factorize with LLTBlockedSolver and solve the system in-place."""
        ref = LLTBlockedSolver(
            operator=operator,
            block_size=self.block_size,
            dtype=float32,
            device=self.default_device,
        )
        ref.compute(A_wp)
        ref.solve(b_wp, x_wp)
        return ref

    def _run_nd_solver(self, operator, A_wp, b_wp, x_wp):
        """Factorize with LLTBlockedNDSolver and solve the system."""
        nd = LLTBlockedNDSolver(
            operator=operator,
            block_size=self.block_size,
            dtype=float32,
            device=self.default_device,
        )
        nd.compute(A_wp)
        nd.solve(b_wp, x_wp)
        return nd

    def _check_matches(self, tag, ref_np, nd_np, A_blocks, b_blocks, dims, vio):
        """Compare solution vectors and residuals per block."""
        num_blocks = len(A_blocks)
        for bi in range(num_blocks):
            s = vio[bi]
            e = s + dims[bi]
            xr = ref_np[s:e]
            xn = nd_np[s:e]

            # Vector equality
            is_close = np.allclose(xn, xr, rtol=self.rtol, atol=self.atol)
            if not is_close or self.verbose:
                diff = np.abs(xn - xr)
                msg.info(
                    "[%s] block %d n=%d: max|x_ref-x_nd|=%.3e rel=%.3e",
                    tag, bi, dims[bi], float(diff.max()),
                    float(diff.max() / max(1e-8, float(np.abs(xr).max()))),
                )
            self.assertTrue(
                is_close,
                msg=f"[{tag}] block {bi}: x from ND solver does not match dense reference",
            )

            # Residual of ND solution against the original (un-reordered) system
            res = A_blocks[bi] @ xn - b_blocks[bi]
            res_inf = float(np.max(np.abs(res)))
            b_inf = float(max(1e-8, np.max(np.abs(b_blocks[bi]))))
            res_rel = res_inf / b_inf
            if self.verbose:
                msg.info("[%s] block %d residual ||Ax-b||_inf=%.3e rel=%.3e",
                         tag, bi, res_inf, res_rel)
            self.assertLess(
                res_rel, 1e-3,
                msg=f"[{tag}] block {bi}: ND solver residual too large ({res_rel:.3e})",
            )

    def _check_tile_pattern_invariants(self, nd, A_blocks, dims, P, vio):
        """Verify tile_pattern (a) has all diagonal tiles set, (b) dominates the
        raw thresholded pattern of the permuted A for every block.
        """
        bs = self.block_size
        tp = nd.tile_pattern.numpy()

        tp_off = 0
        for bi, (A_b, n) in enumerate(zip(A_blocks, dims, strict=False)):
            nt = (n + bs - 1) // bs
            tp_block = tp[tp_off : tp_off + nt * nt].reshape(nt, nt)
            tp_off += nt * nt

            # (a) diagonal forced nonzero
            diag = np.diag(tp_block)
            self.assertTrue(
                bool(np.all(diag != 0)),
                msg=f"block {bi}: tile_pattern has zero diagonal tile(s): {diag.tolist()}",
            )

            # (b) the raw pattern of |P A P^T| must be a subset of tp_block
            # (on the lower triangle). Diagonal of raw is always 1 (SPD).
            P_b = P[vio[bi] : vio[bi] + n]
            A_hat = A_b[P_b][:, P_b]
            raw = (np.abs(A_hat) > 0.0).astype(np.int32)
            # Reduce raw to tile-level 0/1.
            raw_tiles = np.zeros((nt, nt), dtype=np.int32)
            for ti in range(nt):
                for tj in range(nt):
                    r0, r1 = ti * bs, min((ti + 1) * bs, n)
                    c0, c1 = tj * bs, min((tj + 1) * bs, n)
                    raw_tiles[ti, tj] = int(raw[r0:r1, c0:c1].any())

            # On the lower triangle tp >= raw (fill-in never removes entries).
            for ti in range(nt):
                for tj in range(ti + 1):
                    self.assertGreaterEqual(
                        int(tp_block[ti, tj]), int(raw_tiles[ti, tj]),
                        msg=(
                            f"block {bi}: tile_pattern[{ti},{tj}] < raw pattern "
                            f"(fill-in regression)"
                        ),
                    )

    def _check_permutation_is_bijection(self, nd, dims, vio):
        """P and inv_P must be mutually-inverse permutations of [0..n_i) per block."""
        P = nd.P.numpy()
        iP = nd.inv_P.numpy()
        for bi, n in enumerate(dims):
            Pb = P[vio[bi] : vio[bi] + n]
            iPb = iP[vio[bi] : vio[bi] + n]
            self.assertEqual(
                sorted(Pb.tolist()), list(range(n)),
                msg=f"block {bi}: P is not a permutation of [0..{n})",
            )
            # inv_P[P[r]] == r
            comp = iPb[Pb]
            self.assertTrue(
                bool(np.array_equal(comp, np.arange(n, dtype=np.int32))),
                msg=f"block {bi}: inv_P is not the inverse of P",
            )

    ###
    # Tests
    ###

    def test_01_dense_single_block_matches_reference(self):
        """Small dense SPD: ND solver must match LLTBlockedSolver to fp32 rtol."""
        N = 64
        problem = RandomProblemLLT(
            dims=N, seed=self.seed, np_dtype=np.float32, wp_dtype=float32,
            device=self.default_device,
        )
        opinfo = DenseSquareMultiLinearInfo()
        opinfo.finalize(dimensions=problem.dims, dtype=problem.wp_dtype, device=self.default_device)
        operator = DenseLinearOperatorData(info=opinfo, mat=problem.A_wp)

        x_ref = wp.zeros_like(problem.b_wp, device=self.default_device)
        x_nd = wp.zeros_like(problem.b_wp, device=self.default_device)
        self._run_reference_solver(operator, problem.A_wp, problem.b_wp, x_ref)
        nd = self._run_nd_solver(operator, problem.A_wp, problem.b_wp, x_nd)

        # Compare against the numpy reference from problem.x_np and against the dense solver.
        dims = problem.dims
        vio = [0] + [sum(problem.maxdims[:i]) for i in range(1, len(problem.maxdims) + 1)][:-1]
        self._check_matches(
            "dense_n64",
            x_ref.numpy(), x_nd.numpy(),
            problem.A_np, problem.b_np,
            dims, vio,
        )
        self._check_permutation_is_bijection(nd, dims, vio)
        self._check_tile_pattern_invariants(nd, problem.A_np, dims, nd.P.numpy(), vio)

    def test_02_banded_single_block_matches_reference(self):
        """Structurally sparse banded SPD. ND + tile mask actually skips zero tiles here."""
        n = 128
        bandwidth = 5
        A = _make_banded_spd(n, bandwidth, self.seed)
        rng = np.random.default_rng(self.seed + 1)
        b = rng.standard_normal(n).astype(np.float32)
        A_wp, b_wp, dims, mio, vio = _pack_flat([A], [b], self.default_device)

        opinfo = DenseSquareMultiLinearInfo()
        opinfo.finalize(dimensions=dims, dtype=float32, device=self.default_device)
        operator = DenseLinearOperatorData(info=opinfo, mat=A_wp)

        x_ref = wp.zeros_like(b_wp, device=self.default_device)
        x_nd = wp.zeros_like(b_wp, device=self.default_device)
        self._run_reference_solver(operator, A_wp, b_wp, x_ref)
        nd = self._run_nd_solver(operator, A_wp, b_wp, x_nd)

        self._check_matches(
            "banded_n128", x_ref.numpy(), x_nd.numpy(),
            [A], [b], dims, vio,
        )
        self._check_permutation_is_bijection(nd, dims, vio)
        self._check_tile_pattern_invariants(nd, [A], dims, nd.P.numpy(), vio)

    def test_03_random_sparse_single_block_matches_reference(self):
        """Random sparse SPD (low density)."""
        n = 128
        A = _make_random_sparse_spd(n, density=0.05, seed=self.seed)
        rng = np.random.default_rng(self.seed + 2)
        b = rng.standard_normal(n).astype(np.float32)
        A_wp, b_wp, dims, mio, vio = _pack_flat([A], [b], self.default_device)

        opinfo = DenseSquareMultiLinearInfo()
        opinfo.finalize(dimensions=dims, dtype=float32, device=self.default_device)
        operator = DenseLinearOperatorData(info=opinfo, mat=A_wp)

        x_ref = wp.zeros_like(b_wp, device=self.default_device)
        x_nd = wp.zeros_like(b_wp, device=self.default_device)
        self._run_reference_solver(operator, A_wp, b_wp, x_ref)
        nd = self._run_nd_solver(operator, A_wp, b_wp, x_nd)

        self._check_matches(
            "sparse_n128", x_ref.numpy(), x_nd.numpy(),
            [A], [b], dims, vio,
        )
        self._check_permutation_is_bijection(nd, dims, vio)
        self._check_tile_pattern_invariants(nd, [A], dims, nd.P.numpy(), vio)

    def test_04_multi_block_inhomogeneous_dense(self):
        """Multiple dense blocks of different sizes - exercises flat-batched layout."""
        dims_list = [32, 48, 64, 80]
        problem = RandomProblemLLT(
            dims=dims_list, seed=self.seed, np_dtype=np.float32, wp_dtype=float32,
            device=self.default_device,
        )
        opinfo = DenseSquareMultiLinearInfo()
        opinfo.finalize(dimensions=problem.dims, dtype=problem.wp_dtype, device=self.default_device)
        operator = DenseLinearOperatorData(info=opinfo, mat=problem.A_wp)

        x_ref = wp.zeros_like(problem.b_wp, device=self.default_device)
        x_nd = wp.zeros_like(problem.b_wp, device=self.default_device)
        self._run_reference_solver(operator, problem.A_wp, problem.b_wp, x_ref)
        nd = self._run_nd_solver(operator, problem.A_wp, problem.b_wp, x_nd)

        # vio layout from maxdims == dims here (all-active).
        vio = [0] + [sum(problem.maxdims[:i]) for i in range(1, len(problem.maxdims) + 1)][:-1]
        self._check_matches(
            "multi_dense", x_ref.numpy(), x_nd.numpy(),
            problem.A_np, problem.b_np,
            problem.dims, vio,
        )
        self._check_permutation_is_bijection(nd, problem.dims, vio)
        self._check_tile_pattern_invariants(nd, problem.A_np, problem.dims, nd.P.numpy(), vio)

    def test_05_multi_block_inhomogeneous_sparse(self):
        """Multiple sparse blocks of different sizes."""
        A_blocks = [
            _make_banded_spd(48, bandwidth=3, seed=self.seed + 10),
            _make_random_sparse_spd(64, density=0.06, seed=self.seed + 11),
            _make_banded_spd(96, bandwidth=4, seed=self.seed + 12),
        ]
        rng = np.random.default_rng(self.seed + 20)
        b_blocks = [rng.standard_normal(A.shape[0]).astype(np.float32) for A in A_blocks]

        A_wp, b_wp, dims, mio, vio = _pack_flat(A_blocks, b_blocks, self.default_device)
        opinfo = DenseSquareMultiLinearInfo()
        opinfo.finalize(dimensions=dims, dtype=float32, device=self.default_device)
        operator = DenseLinearOperatorData(info=opinfo, mat=A_wp)

        x_ref = wp.zeros_like(b_wp, device=self.default_device)
        x_nd = wp.zeros_like(b_wp, device=self.default_device)
        self._run_reference_solver(operator, A_wp, b_wp, x_ref)
        nd = self._run_nd_solver(operator, A_wp, b_wp, x_nd)

        self._check_matches(
            "multi_sparse", x_ref.numpy(), x_nd.numpy(),
            A_blocks, b_blocks, dims, vio,
        )
        self._check_permutation_is_bijection(nd, dims, vio)
        self._check_tile_pattern_invariants(nd, A_blocks, dims, nd.P.numpy(), vio)

    def test_06_solve_inplace_matches_reference(self):
        """solve_inplace on the ND solver must match solve on the dense reference."""
        n = 96
        A = _make_banded_spd(n, bandwidth=4, seed=self.seed + 3)
        rng = np.random.default_rng(self.seed + 4)
        b = rng.standard_normal(n).astype(np.float32)
        A_wp, b_wp, dims, mio, vio = _pack_flat([A], [b], self.default_device)

        opinfo = DenseSquareMultiLinearInfo()
        opinfo.finalize(dimensions=dims, dtype=float32, device=self.default_device)
        operator = DenseLinearOperatorData(info=opinfo, mat=A_wp)

        # Reference (solve, not solve_inplace)
        x_ref = wp.zeros_like(b_wp, device=self.default_device)
        self._run_reference_solver(operator, A_wp, b_wp, x_ref)

        # ND solver with solve_inplace
        nd = LLTBlockedNDSolver(
            operator=operator, block_size=self.block_size,
            dtype=float32, device=self.default_device,
        )
        nd.compute(A_wp)
        x_inplace = wp.zeros_like(b_wp, device=self.default_device)
        wp.copy(x_inplace, b_wp)
        nd.solve_inplace(x_inplace)

        self._check_matches(
            "solve_inplace", x_ref.numpy(), x_inplace.numpy(),
            [A], [b], dims, vio,
        )

    def test_07_repeat_compute_with_different_A(self):
        """Rebinding the solver to a new A buffer across compute() calls works
        (the ND view attachment cache should detect the pointer change)."""
        n = 80
        A1 = _make_banded_spd(n, bandwidth=3, seed=self.seed + 5)
        A2 = _make_random_sparse_spd(n, density=0.05, seed=self.seed + 6)
        rng = np.random.default_rng(self.seed + 7)
        b = rng.standard_normal(n).astype(np.float32)

        # Build two separate A arrays on device; reuse b.
        A_wp_1, b_wp, dims, mio, vio = _pack_flat([A1], [b], self.default_device)
        A_wp_2 = wp.array(A2.astype(np.float32).reshape(-1), dtype=wp.float32, device=self.default_device)

        opinfo = DenseSquareMultiLinearInfo()
        opinfo.finalize(dimensions=dims, dtype=float32, device=self.default_device)
        operator_1 = DenseLinearOperatorData(info=opinfo, mat=A_wp_1)
        operator_2 = DenseLinearOperatorData(info=opinfo, mat=A_wp_2)

        # Reference solves
        x_ref_1 = wp.zeros_like(b_wp, device=self.default_device)
        x_ref_2 = wp.zeros_like(b_wp, device=self.default_device)
        self._run_reference_solver(operator_1, A_wp_1, b_wp, x_ref_1)
        self._run_reference_solver(operator_2, A_wp_2, b_wp, x_ref_2)

        # Single ND solver used against both A's in sequence.
        nd = LLTBlockedNDSolver(
            operator=operator_1, block_size=self.block_size,
            dtype=float32, device=self.default_device,
        )
        x_nd_1 = wp.zeros_like(b_wp, device=self.default_device)
        x_nd_2 = wp.zeros_like(b_wp, device=self.default_device)
        nd.compute(A_wp_1)
        nd.solve(b_wp, x_nd_1)
        nd.compute(A_wp_2)
        nd.solve(b_wp, x_nd_2)

        self._check_matches("rebind_A1", x_ref_1.numpy(), x_nd_1.numpy(), [A1], [b], dims, vio)
        self._check_matches("rebind_A2", x_ref_2.numpy(), x_nd_2.numpy(), [A2], [b], dims, vio)


if __name__ == "__main__":
    unittest.main()
