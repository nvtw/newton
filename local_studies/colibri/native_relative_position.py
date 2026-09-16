# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run corrected native fixture with only FP32 relative translation integration."""

import importlib.abc
import importlib.machinery
import runpy
import sys

TARGET = "newton._src.solvers.phoenx.solver_phoenx"


class Loader(importlib.abc.Loader):
    """Install after canonical world module executes, before its first instance."""

    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        from .relative_position_control import install  # noqa: PLC0415

        install(module.PhoenXWorld)
        print("RELATIVE_POSITION_CONTROL_BOUND: canonical rotation; outer device reset; FP32", flush=True)


class Finder(importlib.abc.MetaPathFinder):
    """Delegate canonical module resolution; do not intercept owned callbacks."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None
        spec.loader = Loader(spec.loader)
        return spec


def main():
    assert TARGET not in sys.modules
    sys.meta_path.insert(0, Finder())
    runpy.run_module("local_studies.colibri.native_conditioned_two_body", run_name="__main__")


if __name__ == "__main__":
    main()
