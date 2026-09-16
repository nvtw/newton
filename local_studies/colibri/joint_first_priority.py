# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Process-local diagnostic giving bilateral rows first choice of colors."""

import warp as wp

from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


@wp.kernel
def _prioritize_joints(enabled: wp.array[wp.int32], priorities: wp.array[wp.int32]):
    joint = wp.tid()
    if enabled[joint] != 0:
        # Six-point chunks have cost 6. Stay below the int32 sign bit so
        # both signed JP and unsigned endpoint-owner comparisons agree.
        priorities[joint] = wp.int32(0x7F000000) | (priorities[joint] & wp.int32(0x00FFFFFF))


def install():
    """Change only priorities after the ordinary element rebuild."""
    original = PhoenXWorld._rebuild_elements
    if getattr(original, "_joint_first", False):
        return

    def rebuild(self):
        original(self)
        if self.num_joints:
            wp.launch(
                _prioritize_joints,
                dim=self.num_joints,
                inputs=[self._joint_pgs_enabled, self._partitioner._packed_priorities],
                device=self.device,
            )

    rebuild._joint_first = True
    PhoenXWorld._rebuild_elements = rebuild

