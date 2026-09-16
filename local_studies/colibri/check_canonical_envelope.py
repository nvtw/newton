# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check public color groups with geometric admission in the same search envelope."""

import json
import runpy
import sys
from pathlib import Path

from local_studies.colibri.conservative_mesh_candidates import install_conservative_mesh_candidates

if __name__ == "__main__":
    restore = install_conservative_mesh_candidates()
    try:
        runpy.run_module("local_studies.colibri.check_canonical_groups", run_name="__main__")
    finally:
        restore()
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        if output.exists():
            report = json.loads(output.read_text())
            report["candidate_admission"] = "Local geometric admission within the unchanged expanded search envelope"
            output.write_text(json.dumps(report, indent=2))
