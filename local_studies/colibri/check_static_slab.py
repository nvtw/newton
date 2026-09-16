# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Trial exact static bilateral preparation with the validated local slab route."""

import runpy

from local_studies.colibri.static_bilateral_prepare import candidate
from newton._src.solvers.phoenx.articulations import block_joint_system

if __name__ == "__main__":
    original = block_joint_system.prepare_bilateral_joint_blocks
    block_joint_system.prepare_bilateral_joint_blocks = candidate()
    try:
        runpy.run_module("local_studies.colibri.check_public_slab_noenvelope", run_name="__main__")
    finally:
        block_joint_system.prepare_bilateral_joint_blocks = original
