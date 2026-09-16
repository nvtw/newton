# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Install the local normal-first iterate candidate before constructing kernels."""

import runpy
import sys

if "--normal-first" in sys.argv:
    sys.argv.remove("--normal-first")
    from local_studies.colibri import prototype_normal_first
    from newton._src.solvers.phoenx import solver_phoenx_kernels

    for name in dir(prototype_normal_first):
        if name.startswith("contact_iterate") and hasattr(solver_phoenx_kernels, name):
            setattr(solver_phoenx_kernels, name, getattr(prototype_normal_first, name))

if __name__ == "__main__":
    runpy.run_module("local_studies.colibri.check_mu0", run_name="__main__")
