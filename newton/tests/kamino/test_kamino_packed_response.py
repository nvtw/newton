# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare production packed response specializations across contact-count changes."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.dvi import kernels


class TestPackedResponse(unittest.TestCase):
    def test_captured_response_capacity_transitions(self):
        """Preserve dense response arithmetic across compact/full transitions and changing couplings."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Captured cooperative responses require CUDA")
        rng = np.random.default_rng(58132)
        dimensions = np.array([0, 17, 33, 65, 258], dtype=np.int32)
        capacity = 65
        tiles = (dimensions + 31) // 32
        vio = np.cumsum(np.r_[0, dimensions[:-1]]).astype(np.int32)
        mio = np.cumsum(np.r_[0, (dimensions * dimensions)[:-1]]).astype(np.int32)
        tpo = np.cumsum(np.r_[0, (tiles * tiles)[:-1]]).astype(np.int32)
        slot_offsets = np.cumsum(np.r_[0, (tiles * (tiles + 1) // 2)[:-1]]).astype(np.int64)
        guard = 11
        sizes = dimensions * capacity
        rio = (guard + np.cumsum(np.r_[0, (sizes + guard)[:-1]])).astype(np.int32)
        total = int(np.sum(sizes + guard) + guard)
        factors, panels, permutations, masks, expected_prefixes = [], [], [], [], []
        for n, count in zip(dimensions, tiles, strict=True):
            lower = np.tril(rng.normal(0.0, 0.025, (n, n))).astype(np.float32)
            np.fill_diagonal(lower, 1.0)
            lower[: min(n, 16)] = np.eye(n, dtype=np.float32)[: min(n, 16)]
            lower[32:, :32] = 0.0
            factors.append(lower)
            expected_prefixes.extend(int(np.flatnonzero(lower[row])[0]) // 16 * 16 for row in range(n))
            padded = np.zeros((count * 32, count * 32), dtype=np.float32)
            padded[:n, :n] = lower
            mask = np.eye(count, dtype=np.int32)
            for row in range(count):
                for col in range(row + 1):
                    panel = padded[row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32]
                    panels.append(panel)
                    if np.any(panel):
                        mask[row, col] = mask[col, row] = 1
            masks.extend(mask.ravel())
            permutations.append(rng.permutation(n).astype(np.int32))

        def ints(values):
            return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)

        def floats(values):
            return wp.array(np.asarray(values, dtype=np.float32), dtype=wp.float32, device=device)

        njc, vector_offsets, dense_offsets, response_offsets, strides, order, pattern_offsets, pattern = map(
            ints,
            (dimensions, vio, mio, rio, np.full(len(dimensions), capacity), np.concatenate(permutations), tpo, masks),
        )
        packed_offsets = wp.array(slot_offsets, dtype=wp.int64, device=device)
        factor_arrays = [floats(np.concatenate([a.ravel() for a in factors])), floats(np.asarray(panels).ravel())]
        factor_offsets = [dense_offsets, packed_offsets]
        scales = rng.uniform(0.5, 2.0, int(dimensions.sum())).astype(np.float32)
        scale = floats(scales)
        coupling = wp.zeros(total, dtype=wp.float32, device=device)
        problem_dim = ints(dimensions)
        prefixes = [wp.zeros(int(dimensions.sum()), dtype=wp.int32, device=device) for _ in range(2)]
        for index in range(2):
            for use_rcm in (False, True):
                factory = (
                    kernels.make_find_bilateral_factor_row_start_rcm_kernel
                    if use_rcm
                    else kernels.make_find_bilateral_factor_row_start_kernel
                )
                arguments = [njc, factor_offsets[index], vector_offsets, factor_arrays[index], prefixes[index]]
                if use_rcm:
                    arguments.extend([pattern_offsets, pattern, 32])
                wp.launch(
                    factory(bool(index)), dim=(len(dimensions), int(dimensions.max())), inputs=arguments, device=device
                )
                np.testing.assert_array_equal(prefixes[index].numpy(), expected_prefixes)

        sentinel = np.float32(-741.25)
        for use_permutation in (False, True):
            for mode in ("scalar", "full", "forward", "split"):
                if mode == "split" and not use_permutation:
                    continue
                outputs = [wp.zeros_like(coupling) for _ in range(2)]
                scratch = [wp.zeros_like(coupling) for _ in range(2)]

                def launch(mode, use_permutation, outputs, scratch):
                    for index in range(2):
                        common = [
                            problem_dim,
                            njc,
                            factor_offsets[index],
                            vector_offsets,
                            scale,
                            factor_arrays[index],
                            order,
                        ]
                        if mode == "scalar":
                            wp.launch(
                                kernels.make_solve_bilateral_unilateral_response_kernel(bool(index)),
                                dim=len(dimensions) * 128,
                                block_dim=128,
                                inputs=[
                                    *common,
                                    use_permutation,
                                    response_offsets,
                                    strides,
                                    coupling,
                                    scratch[index],
                                    outputs[index],
                                ],
                                device=device,
                            )
                        else:
                            wp.launch(
                                kernels.make_solve_bilateral_unilateral_response_cooperative_kernel(bool(index)),
                                dim=len(dimensions) * 4 * 32,
                                block_dim=128,
                                inputs=[
                                    *common,
                                    use_permutation,
                                    response_offsets,
                                    strides,
                                    coupling,
                                    scratch[index],
                                    outputs[index],
                                    0,
                                    4,
                                    mode != "full",
                                    prefixes[index],
                                    mode == "split",
                                ],
                                device=device,
                            )
                            if mode == "split":
                                wp.launch(
                                    kernels.make_solve_bilateral_unilateral_response_compact_kernel(bool(index)),
                                    dim=(len(dimensions), capacity),
                                    block_dim=128,
                                    inputs=[
                                        *common,
                                        response_offsets,
                                        strides,
                                        coupling,
                                        outputs[index],
                                        prefixes[index],
                                    ],
                                    device=device,
                                )

                launch(mode, use_permutation, outputs, scratch)
                with wp.ScopedCapture(device=device) as capture:
                    launch(mode, use_permutation, outputs, scratch)
                for nu in (0, 1, 33, 46, 47, 65, 1):
                    counts = dimensions + nu
                    counts[0] = 0
                    problem_dim.assign(counts)
                    coupling_values = np.full(total, sentinel, dtype=np.float32)
                    expected = np.full(total, sentinel, dtype=np.float64)
                    for world, n in enumerate(dimensions):
                        if not n or not nu:
                            continue
                        offset = rio[world]
                        values = rng.normal(size=(n, nu)).astype(np.float32)
                        values[: min(n, 8)] = 0.0
                        values[:, -1] = 0.0
                        if mode in ("forward", "split") and nu * nu <= n * capacity:
                            # Compact worlds store the coupling densely with row stride nu.
                            coupling_values[offset : offset + n * nu] = values.ravel()
                        else:
                            coupling_values[offset : offset + n * capacity].reshape(n, capacity)[:, :nu] = values
                        local_scale = scales[vio[world] : vio[world] + n]
                        permutation = permutations[world] if use_permutation else np.arange(n)
                        scaled = (local_scale[:, None] * values)[permutation].astype(np.float64)
                        white = np.linalg.solve(factors[world].astype(np.float64), scaled)
                        if mode in ("forward", "split") and nu * nu <= n * capacity:
                            expected[offset : offset + n * nu] = white.ravel()
                        else:
                            solved = np.linalg.solve(factors[world].astype(np.float64).T, white)
                            original = np.empty_like(solved)
                            original[permutation] = solved
                            expected[offset : offset + n * capacity].reshape(n, capacity)[:, :nu] = (
                                local_scale[:, None] * original
                            )
                    coupling.assign(coupling_values)
                    for array in outputs + scratch:
                        array.fill_(sentinel)
                    wp.capture_launch(capture.graph)
                    with self.subTest(permutation=use_permutation, mode=mode, contacts=nu):
                        actual = outputs[1].numpy()
                        np.testing.assert_array_equal(actual.view(np.uint32), outputs[0].numpy().view(np.uint32))
                        np.testing.assert_array_equal(
                            scratch[1].numpy().view(np.uint32), scratch[0].numpy().view(np.uint32)
                        )
                        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-6)
                        inactive = expected == sentinel
                        np.testing.assert_array_equal(actual[inactive], sentinel)


if __name__ == "__main__":
    unittest.main()
