# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run the guarded certified material-domain reference on native Colibri."""

import os
import sys
from pathlib import Path

from .direct_relax_no_skip import NAME, Finder


def main():
    """Install the certified hook and run the ordinary public stage runner."""
    finder = Finder()
    sys.modules.pop(NAME, None)
    sys.meta_path.insert(0, finder)
    import runpy

    from local_studies.colibri.certified_patch_history import install
    from newton.solvers import SolverPhoenX
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    original_init = SolverPhoenX.__init__
    installed = []

    def hooked_init(solver, model, *args, **kwargs):
        original_init(solver, model, *args, **kwargs)
        installed.append(install(solver, model, output, os.environ["COLIBRI_PATCH_CERTIFICATE"]))

    SolverPhoenX.__init__ = hooked_init
    try:
        runpy.run_module("local_studies.colibri.check_public_analytic_gradient", run_name="__main__")
    finally:
        SolverPhoenX.__init__ = original_init
        for finish in installed:
            finish()


if __name__ == "__main__":
    main()
