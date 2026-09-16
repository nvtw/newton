# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare the local copy-aware preparation with the unchanged scene schedule."""

import runpy
import sys

from local_studies.colibri import check_bilateral_pgs as runner
from local_studies.colibri import check_chunk_coloring  # noqa: F401
from local_studies.colibri.parallel_prepare_split import install

if "--split-parallel-prepare" in sys.argv:
    sys.argv.remove("--split-parallel-prepare")
    runner.install_parallel_prepare = install
    if "--parallel-prepare" not in sys.argv:
        sys.argv.append("--parallel-prepare")

if __name__ == "__main__":
    runpy.run_module("local_studies.colibri.check_batch_aware_tail", run_name="__main__")
