# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare the exact-zero warm-start shortcut with saved full-scene state."""

import runpy
import sys

if "--skip-zero-warmstart" in sys.argv:
    sys.argv.remove("--skip-zero-warmstart")
    from local_studies.colibri.zero_warmstart import contact_cached_warmstart_lean
    from newton._src.solvers.phoenx import solver_phoenx_kernels

    solver_phoenx_kernels.contact_cached_warmstart_lean = contact_cached_warmstart_lean

if __name__ == "__main__":
    runpy.run_module("local_studies.colibri.check_tail_launch", run_name="__main__")
