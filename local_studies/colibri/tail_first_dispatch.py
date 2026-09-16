# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry for integrated tail-first dispatch and study source helpers."""

import hashlib
import runpy
import sys
import tempfile
from pathlib import Path


def _compile_function(source, module, name):
    directory = Path(tempfile.gettempdir()) / "colibri_tail_first_dispatch"
    directory.mkdir(exist_ok=True)
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    path = directory / f"{name}_{digest}.py"
    path.write_text(source)
    namespace = dict(vars(module))
    namespace["__file__"] = str(path)
    exec(compile(source, str(path), "exec"), namespace)
    return namespace[name]


def install():
    """Keep old study commands working; production dispatch is now tail-first."""


if __name__ == "__main__":
    install()
    sys.argv = sys.argv[1:]
    runpy.run_module(sys.argv[0], run_name="__main__")
