# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry for the integrated bounded contact-preparation grid."""

import runpy
import sys


def install():
    """Keep old study commands working; the production grid is already bounded."""


if __name__ == "__main__":
    install()
    sys.argv = sys.argv[1:]
    runpy.run_module(sys.argv[0], run_name="__main__")
