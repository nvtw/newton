# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run predictive Colibri contacts with endpoint and contact-owner audits."""

import runpy

from local_studies.colibri import check_chunk_coloring  # noqa: F401

if __name__ == "__main__":
    runpy.run_module("local_studies.colibri.check_speculative_tail", run_name="__main__")
