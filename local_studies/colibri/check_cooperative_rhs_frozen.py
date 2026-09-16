# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare all mutable world arrays for one identical-input grouped sweep."""

import sys

import warp as wp

import newton.examples
from local_studies.colibri.cooperative_bilateral_rhs import install
from local_studies.colibri.freeze_support_star import collect
from newton._src.solvers.phoenx.dispatch import color_groups
from newton.examples.phoenx.example_phoenx_colibri import Example


class Frozen(Exception):
    pass


def main():
    sys.argv = [sys.argv[0], "--viewer", "null", "--num-frames", "1"]
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    example.step()
    world = example.solver.world
    soft = bool(world._dispatch_specialization_flags()["has_soft_contact_pd"])
    wanted = color_groups.get_sweep_kernel("iterate", soft)
    original = wp.launch
    captured = {}

    def intercept(kernel, *positional, **keywords):
        if kernel is wanted:
            captured.update(positional=positional, keywords=keywords)
            raise Frozen
        return original(kernel, *positional, **keywords)

    wp.launch = intercept
    try:
        example.simulate()
    except Frozen:
        pass
    finally:
        wp.launch = original
    assert captured
    saved = collect(world)
    original(wanted, *captured["positional"], **captured["keywords"])
    expected = [a.numpy().copy() for a, _ in saved]
    for destination, source in saved:
        wp.copy(destination, source)
    restore = install()
    try:
        wp.launch(wanted, *captured["positional"], **captured["keywords"])
        for index, ((actual, _), before) in enumerate(zip(saved, expected, strict=True)):
            assert actual.numpy().tobytes() == before.tobytes(), index
    finally:
        restore()
    print(f"PASS: {len(saved)} world arrays byte-exact")


if __name__ == "__main__":
    main()
