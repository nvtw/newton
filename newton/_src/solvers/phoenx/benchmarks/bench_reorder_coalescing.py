# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Research probe: measures the coalescing penalty of scattered cids in
PhoenX's per-color partition, and the upper-bound speedup from
physically reordering constraint / contact storage to be contiguous
within each color.

Background
----------
The fast-tail iterate kernels gather constraint columns through an
indirection::

    cid = world_element_ids_by_color[start + lane]
    ... data[off, cid] ...

Within a color, lanes within a warp have *different* cids. Because
the cids are not contiguous in storage, those ``data[off, cid]``
loads scatter to up to 32 cache lines per warp instruction, instead
of the single coalesced 128-byte transaction we'd get if consecutive
lanes hit consecutive columns. Physically permuting the storage so
that within each color the cids are dense (e.g. ``cid = start_in_color
+ lane``) would recover full coalescing and -- in principle -- a
proportional speedup on the hot inner loop.

This script measures whether that speedup is real and worth the
implementation cost. It isolates the inner solve (``_solve_main`` for
multi-world layout, ``_solve_main_singleworld`` for single-world) so
the result reflects the kernel-level coalescing impact, not the
end-to-end step throughput. After applying an offline cluster-by-color
permutation to all per-cid containers (joint constraints,
contact-column headers, contact-impulse lambdas + prev-lambdas), it
re-runs the same inner-solve loop and reports the ratio.

Why this benchmark exists
-------------------------
The reorder optimization was prototyped under
``dev/tw3/`` (see project history). The measured kernel-only speedup
on representative scenes was small (+0.5% on tower_32, +1-5% on
h1_flat across sizes) -- below the threshold needed to justify the
implementation cost (handle map, double-buffered storage, permutation
kernel woven into the captured graph). The bench is kept here so that:

  1. The numbers can be re-validated cheaply if the inner-kernel
     access pattern changes (e.g. after a constraint-row layout
     change that alters per-cid memory pressure).
  2. When the reorder design is revived as the foundation for
     sleeping / runtime constraint removal / island stepping, this
     same probe doubles as the perf-regression / perf-gain measurement.

How it works
------------
For each scene:

  1. Build via the existing benchmark scenarios.
  2. Step the scene a few real frames so contacts populate and the
     first JP coloring runs.
  3. Bench: graph-capture and replay ``solver.world._solve_main()``
     ``--n_runs`` times. This skips coloring and integration; only the
     prepare + iterate + relax kernels run, which is the loop the
     reorder targets.
  4. Compute a "joints first then contacts" permutation that
     clusters cids contiguously within each (world, color) slot.
     Apply it to every per-cid container in lockstep using a plain
     gather kernel (Warp's fancy-indexing form
     ``arr[:, perm]`` returns an ``indexedarray`` that ``@wp.struct``
     setters reject, and is a known-slow path anyway).
  5. Bench again. The ratio of (1) and (5) is the kernel-only upper
     bound on the reorder optimization's payoff.

How to interpret
----------------
  * ``contiguous-pair fraction`` reports what fraction of consecutive
    entries in ``element_ids_by_color`` already point to consecutive
    storage rows of the same constraint type. Should jump from
    ~0-10% (baseline) to ~85-100% (post-permutation). If the
    post-permutation number is < 80%, the bench's permutation logic
    has a bug.
  * The headline ``speedup`` is the inner-kernel-only upper bound.
    The end-to-end ``step()`` speedup will be a fraction of this
    because coloring / integration / contact ingest don't benefit.

Reference numbers on RTX PRO 6000 (sm_count=188), inner kernel only::

    tower      32 worlds (multi_world): +0.9%
    tower      1 world  (single_world): +0.5%
    h1_flat   64 worlds: +1.6%
    h1_flat  512 worlds: +4.5%
    h1_flat 2048 worlds: +1.0%

Usage::

    python -m newton._src.solvers.phoenx.benchmarks.bench_reorder_coalescing
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.benchmarks.scenarios import h1_flat, tower
from newton._src.solvers.phoenx.solver import SolverPhoenX
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


def _extract_solver(handle) -> SolverPhoenX:
    """Reach into a :class:`SceneHandle` to grab its underlying solver.

    See ``bench_threads_per_world._extract_solver`` for the rationale;
    we walk ``simulate_one_frame.__closure__`` because the scenarios
    don't expose the solver directly.
    """
    fn = handle.simulate_one_frame
    if fn.__closure__ is None:
        raise RuntimeError("simulate_one_frame has no closure")
    for cell in fn.__closure__:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, SolverPhoenX):
            return value
    raise RuntimeError("SolverPhoenX not found in simulate_one_frame closure")


# ---------------------------------------------------------------------------
# Per-color iteration helpers (handle both layouts)
# ---------------------------------------------------------------------------


def _color_iter_multi_world(world: PhoenXWorld):
    """Yield ``(slot_start, slot_end)`` for every (world, color) range.

    Reads the per-world CSR buckets the multi_world fast-tail kernels
    consume. Empty colors emit a zero-length range and are skipped by
    the caller.
    """
    starts = world._world_color_starts.numpy()
    csr = world._world_csr_offsets.numpy()
    nc_per_world = world._world_num_colors.numpy()
    for w in range(world.num_worlds):
        wbase = int(csr[w])
        for c in range(int(nc_per_world[w])):
            yield wbase + int(starts[w, c]), wbase + int(starts[w, c + 1])


def _color_iter_single_world(world: PhoenXWorld):
    """Yield ``(slot_start, slot_end)`` for the global single_world coloring."""
    starts = world._partitioner.color_starts.numpy()
    nc = int(world._partitioner.num_colors.numpy()[0])
    for c in range(nc):
        yield int(starts[c]), int(starts[c + 1])


def _global_eids(world: PhoenXWorld) -> tuple[np.ndarray, str]:
    """Return ``(element_ids_by_color_buffer, label)`` for the active layout."""
    if world.step_layout == "single_world":
        return world._partitioner.element_ids_by_color.numpy(), "partitioner.element_ids_by_color"
    return world._world_element_ids_by_color.numpy(), "_world_element_ids_by_color"


def _build_permutation(world: PhoenXWorld):
    """Return ``(joint_inv, contact_inv, new_eids, n_active_j, n_active_c)``.

    Within each (world, color) slot, joint cids come first (in
    original encounter order) then contact cids. New joint cids are
    assigned contiguously in ``[0, num_joints)``; new contact cids
    contiguously in ``[num_joints, num_joints + max_contact_columns)``.
    Inactive slots stay at identity so the un-touched portion of each
    container keeps a valid mapping.
    """
    eids, _ = _global_eids(world)
    num_joints = world.num_joints
    num_contacts = world.max_contact_columns

    joint_inv = np.arange(max(1, num_joints), dtype=np.int32)
    contact_inv = np.arange(max(1, num_contacts), dtype=np.int32)
    new_eids = eids.copy()

    next_j = 0
    next_c = 0
    iter_fn = _color_iter_single_world if world.step_layout == "single_world" else _color_iter_multi_world
    for s, e in iter_fn(world):
        joints: list[int] = []
        contacts: list[int] = []
        for i in range(s, e):
            old = int(eids[i])
            if old < 0:
                continue
            if old < num_joints:
                joints.append(old)
            else:
                contacts.append(old - num_joints)
        for old_j in joints:
            joint_inv[next_j] = old_j
            next_j += 1
        for old_k in contacts:
            contact_inv[next_c] = old_k
            next_c += 1
        offs = s
        for jj in range(len(joints)):
            new_eids[offs] = (next_j - len(joints)) + jj
            offs += 1
        for cc in range(len(contacts)):
            new_eids[offs] = num_joints + (next_c - len(contacts)) + cc
            offs += 1

    return joint_inv, contact_inv, new_eids, next_j, next_c


# ---------------------------------------------------------------------------
# Permutation kernel + apply
# ---------------------------------------------------------------------------


@wp.kernel
def _permute_columns_kernel(
    src: wp.array2d[wp.float32],
    dst: wp.array2d[wp.float32],
    inv_perm: wp.array[wp.int32],
):
    """``dst[r, new_c] = src[r, inv_perm[new_c]]`` -- gather columns.

    Plain custom kernel (as opposed to the fancy-indexing form
    ``src[:, inv_perm]``) because (a) the fancy-indexing return type
    is ``warp.indexedarray``, which the ``@wp.struct`` setters reject,
    and (b) per the project conventions plain indexing is the
    fastest path on the device.
    """
    r, new_c = wp.tid()
    old_c = inv_perm[new_c]
    dst[r, new_c] = src[r, old_c]


def _permute_2d(arr: wp.array2d, inv_perm: wp.array) -> wp.array2d:
    """Allocate a same-shape destination and gather columns into it."""
    rows, cols = arr.shape
    dst = wp.zeros((rows, cols), dtype=arr.dtype, device=arr.device)
    wp.launch(_permute_columns_kernel, dim=(rows, cols), inputs=[arr, dst, inv_perm])
    return dst


def _apply_permutation(
    world: PhoenXWorld,
    joint_inv: np.ndarray,
    contact_inv: np.ndarray,
    new_eids: np.ndarray,
) -> None:
    """Permute every per-cid container in lockstep, then rewrite the
    element_ids_by_color buffer to identity-within-color form."""
    device = world.device

    if world.num_joints > 0:
        ji = wp.array(joint_inv, dtype=wp.int32, device=device)
        world.constraints.data = _permute_2d(world.constraints.data, ji)

    if world.max_contact_columns > 0:
        ci = wp.array(contact_inv, dtype=wp.int32, device=device)
        world._contact_cols.data = _permute_2d(world._contact_cols.data, ci)
        world._contact_container.lambdas = _permute_2d(world._contact_container.lambdas, ci)
        world._contact_container.prev_lambdas = _permute_2d(world._contact_container.prev_lambdas, ci)

    new_eids_arr = wp.array(new_eids, dtype=wp.int32, device=device)
    if world.step_layout == "single_world":
        wp.copy(world._partitioner._element_ids_by_color, new_eids_arr)
    else:
        world._world_element_ids_by_color = new_eids_arr
    wp.synchronize_device()


# ---------------------------------------------------------------------------
# Stats + bench
# ---------------------------------------------------------------------------


def _print_color_stats(world: PhoenXWorld) -> None:
    """Print the contiguous-pair fraction + per-color size histogram.

    The contiguous-pair fraction jumps from ~0-10% (scattered cids)
    to ~85-100% (after permutation); if it doesn't, the bench is
    broken.
    """
    eids, label = _global_eids(world)
    num_joints = world.num_joints

    contig_runs = 0
    total_lanes = 0
    color_counts: list[int] = []
    iter_fn = _color_iter_single_world if world.step_layout == "single_world" else _color_iter_multi_world
    for s, e in iter_fn(world):
        count = e - s
        color_counts.append(count)
        if count <= 1:
            continue
        for i in range(s + 1, e):
            if eids[i] == eids[i - 1] + 1 and ((eids[i] < num_joints) == (eids[i - 1] < num_joints)):
                contig_runs += 1
            total_lanes += 1
    if total_lanes:
        print(
            f"  ({label}) contiguous-pair fraction: "
            f"{100.0 * contig_runs / total_lanes:.1f}%  "
            f"(higher = already partly clustered)"
        )
    if color_counts:
        cc = np.asarray(color_counts)
        print(
            f"  per-color cid counts: n={len(cc)} min={cc.min()} "
            f"max={cc.max()} mean={cc.mean():.1f} median={int(np.median(cc))}"
        )


def _bench_solve_main(world: PhoenXWorld, n_runs: int, warmup: int, trials: int) -> tuple[float, float, float]:
    """Capture ``_solve_main`` into a CUDA graph and replay ``n_runs``
    times per trial across ``trials`` trials.

    Returns ``(min_ms, median_ms, mean_ms)``. ``_solve_main`` (or its
    single-world counterpart) is the inner solve only -- coloring,
    contact ingest, and integration are excluded so the result
    isolates the kernel-level access pattern.
    """
    device = wp.get_device()
    solve_fn = world._solve_main_singleworld if world.step_layout == "single_world" else world._solve_main

    for _ in range(warmup):
        solve_fn()
    wp.synchronize_device()

    if device.is_cuda:
        with wp.ScopedCapture() as capture:
            solve_fn()
        graph = capture.graph
        wp.synchronize_device()
        results: list[float] = []
        for _ in range(trials):
            t0 = time.perf_counter()
            for _ in range(n_runs):
                wp.capture_launch(graph)
            wp.synchronize_device()
            results.append((time.perf_counter() - t0) * 1000.0)
    else:
        results = []
        for _ in range(trials):
            t0 = time.perf_counter()
            for _ in range(n_runs):
                solve_fn()
            results.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(results)
    return float(arr.min()), float(np.median(arr)), float(arr.mean())


# ---------------------------------------------------------------------------
# Scene runners
# ---------------------------------------------------------------------------


def _run_scene(
    label: str, build_fn: Callable, kwargs: dict, n_runs: int, warmup: int, trials: int
) -> tuple[float, float, float]:
    print(f"\n=== {label} ===")
    print(f"  args: {kwargs}")

    handle = build_fn(**kwargs)
    solver = _extract_solver(handle)
    world = solver.world

    # Prime: a few full step calls so contacts ingest + first coloring populate.
    for _ in range(5):
        handle.simulate_one_frame()
    wp.synchronize_device()

    print(
        f"  num_worlds={world.num_worlds}  num_joints={world.num_joints}  "
        f"max_contact_columns={world.max_contact_columns}  layout={world.step_layout}"
    )
    _print_color_stats(world)

    base_min, base_med, _ = _bench_solve_main(world, n_runs=n_runs, warmup=warmup, trials=trials)
    print(
        f"  baseline (scattered):  min={base_min:8.2f}  med={base_med:8.2f}  ms / {n_runs} runs   "
        f"({1000.0 * base_min / n_runs:.2f} us/run @min)"
    )

    ji, ci, new_eids, n_active_j, n_active_c = _build_permutation(world)
    print(
        f"  active joint cids: {n_active_j}/{world.num_joints}  "
        f"active contact cids: {n_active_c}/{world.max_contact_columns}"
    )
    _apply_permutation(world, ji, ci, new_eids)
    _print_color_stats(world)

    reordered_min, reordered_med, _ = _bench_solve_main(world, n_runs=n_runs, warmup=warmup, trials=trials)
    print(
        f"  reordered (clustered): min={reordered_min:8.2f}  med={reordered_med:8.2f}  ms / {n_runs} runs   "
        f"({1000.0 * reordered_min / n_runs:.2f} us/run @min)"
    )

    speedup_min = base_min / reordered_min if reordered_min > 0 else float("nan")
    speedup_med = base_med / reordered_med if reordered_med > 0 else float("nan")
    print(
        f"  speedup: min-vs-min={speedup_min:.3f}x ({100 * (speedup_min - 1):+.1f}%)  "
        f"med-vs-med={speedup_med:.3f}x ({100 * (speedup_med - 1):+.1f}%)"
    )
    return base_min, reordered_min, speedup_min


_SCENE_REGISTRY: dict[str, tuple[Callable, dict, str]] = {
    "tower_multi": (
        tower.build,
        {"solver_name": "phoenx", "step_layout": "multi_world"},
        "tower (multi_world, contact-dense, default num_worlds=32)",
    ),
    "tower_single": (
        tower.build,
        {"num_worlds": 1, "solver_name": "phoenx", "step_layout": "single_world"},
        "tower (1 world, single_world, big single-world stress)",
    ),
    "h1_flat_64": (
        h1_flat.build,
        {"num_worlds": 64, "solver_name": "phoenx"},
        "h1_flat (64 worlds, multi_world robot)",
    ),
    "h1_flat_512": (
        h1_flat.build,
        {"num_worlds": 512, "solver_name": "phoenx"},
        "h1_flat (512 worlds, multi_world robot)",
    ),
    "h1_flat_2048": (
        h1_flat.build,
        {"num_worlds": 2048, "solver_name": "phoenx"},
        "h1_flat (2048 worlds, multi_world robot)",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["tower_multi", "tower_single", "h1_flat_64", "h1_flat_512", "h1_flat_2048"],
        choices=sorted(_SCENE_REGISTRY.keys()),
        help="Scenes to bench (default: full sweep).",
    )
    parser.add_argument("--n_runs", type=int, default=1000, help="Inner-solve replays per trial. Higher -> less noise.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--solver_iterations", type=int, default=8)
    parser.add_argument(
        "--tower_worlds", type=int, default=32, help="num_worlds passed to the tower scene's multi_world build."
    )
    args = parser.parse_args()

    wp.init()
    print(f"device: {wp.get_device()}")

    results: dict[str, tuple[float, float, float]] = {}
    for scene_key in args.scenes:
        builder, base_kwargs, label = _SCENE_REGISTRY[scene_key]
        kwargs = {
            **base_kwargs,
            "substeps": args.substeps,
            "solver_iterations": args.solver_iterations,
        }
        if scene_key == "tower_multi":
            kwargs.setdefault("num_worlds", args.tower_worlds)
        results[scene_key] = _run_scene(
            label, builder, kwargs, n_runs=args.n_runs, warmup=args.warmup, trials=args.trials
        )

    print("\n=== Summary ===")
    print(f"{'scene':40s}  {'baseline ms':>12s}  {'reordered ms':>13s}  {'speedup':>8s}")
    for key, (b, r, s) in results.items():
        print(f"{key:40s}  {b:12.2f}  {r:13.2f}  {s:8.3f}x")


if __name__ == "__main__":
    main()
