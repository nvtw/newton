# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Skip global-solve copy round trips only for unbounded local block joints."""

import inspect
import runpy
import textwrap

from newton._src.solvers.phoenx.articulations.block_joint_system import BlockJointSystem
from newton._src.solvers.phoenx.dispatch import single_world_mass_splitting as dispatch

original = dispatch.SingleWorldMassSplittingDispatcher.solve
source = textwrap.dedent(inspect.getsource(original))
marker = "        direct.prepare_and_factor(idt)\n"
assert source.count(marker) == 1
source = source.replace(marker, marker + "        direct = None\n")
namespace = dict(dispatch.__dict__)
exec(compile(source, __file__, "exec"), namespace)
candidate = namespace["solve"]


def solve(self, idt):
    world = self._world
    direct = world._direct_equality_system
    assert isinstance(direct, BlockJointSystem) and direct.enabled
    assert not direct.has_bounded_drives
    assert world._direct_contact_response is None
    assert world._maximal_tree_projector is None
    assert not world._reduced_constraints_active_this_step
    assert world._regular_pgs_active_this_step
    return candidate(self, idt)


dispatch.SingleWorldMassSplittingDispatcher.solve = solve
try:
    runpy.run_module("local_studies.colibri.check_canonical_groups", run_name="__main__")
finally:
    dispatch.SingleWorldMassSplittingDispatcher.solve = original
