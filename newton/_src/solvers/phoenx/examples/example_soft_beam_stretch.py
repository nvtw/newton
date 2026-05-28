# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PhoenX soft-beam stretch demo.

A tetrahedral beam has its left face fixed and its right face held in a
prescribed stretched pose. The default material uses a Poisson ratio of
0.45 and the standard PhoenX ARAP tet constraint path.

Run::

    python -m newton._src.solvers.phoenx.examples.example_soft_beam_stretch
"""

from __future__ import annotations

import newton.examples
from newton._src.solvers.phoenx.examples._soft_beam_common import (
    SoftBeamExample,
    create_soft_beam_parser,
    soft_beam_kwargs_from_args,
)


class Example(SoftBeamExample):
    def __init__(self, viewer, args=None):
        super().__init__(viewer, args, mode="stretch", **soft_beam_kwargs_from_args(args))

    @staticmethod
    def create_parser():
        return create_soft_beam_parser(mode="stretch")


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
