# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run the source slab diagnostic with local geometric candidate admission."""

import runpy

from local_studies.colibri.conservative_mesh_candidates import install_conservative_mesh_candidates

if __name__ == "__main__":
    restore = install_conservative_mesh_candidates()
    try:
        runpy.run_module("local_studies.colibri.check_slab_colibri", run_name="__main__")
    finally:
        restore()
