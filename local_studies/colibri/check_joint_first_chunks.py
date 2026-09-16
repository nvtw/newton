# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run chunk exclusivity and physical checks with joint-first priorities."""

from local_studies.colibri.check_chunk_coloring import runner
from local_studies.colibri.joint_first_priority import install

install()

if __name__ == "__main__":
    runner.main()
