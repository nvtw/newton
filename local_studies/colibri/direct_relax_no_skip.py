"""Diagnostic: disable the direct solver's repeat-relaxation residual skip only.

The physical hard/drive equations, factorization and iteration counts are unchanged.
This is an isolated source import, not a production default modification.
"""

import hashlib
import importlib.abc
import importlib.util
import json
import runpy
import sys
import tempfile
from pathlib import Path

NAME = "newton._src.solvers.phoenx.articulations.direct_equality"


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self):
        source = Path(__file__).resolve().parents[2] / "newton/_src/solvers/phoenx/articulations/direct_equality.py"
        self.original = source.read_text()
        marker = "_DIRECT_RELAX_RESIDUAL_TOLERANCE = 1.0e-4"
        assert self.original.count(marker) == 1
        self.modified = self.original.replace(marker, "_DIRECT_RELAX_RESIDUAL_TOLERANCE = 0.0")
        self.path = Path(tempfile.mkdtemp(prefix="colibri_direct_no_skip_")) / "direct_equality.py"
        self.path.write_text(self.modified)
        self.source = source

    def find_spec(self, fullname, path=None, target=None):
        if fullname == NAME:
            return importlib.util.spec_from_file_location(fullname, self.path)
        return None


def main():
    assert NAME not in sys.modules, "Direct solver already imported; overlay would be ineffective"
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    finder = Finder()
    sys.meta_path.insert(0, finder)
    try:
        runpy.run_module("local_studies.colibri.native_conditioned_two_body", run_name="__main__")
    finally:
        module = sys.modules.get(NAME)
        assert module is not None and Path(module.__file__) == finder.path
        assert module._DIRECT_RELAX_RESIDUAL_TOLERANCE == 0.0
        assert finder.source.read_text() == finder.original, "Canonical direct source changed during diagnostic"
        output.with_suffix(".direct_no_skip.json").write_text(
            json.dumps(
                {
                    "scope": __doc__,
                    "actual_module_path": str(module.__file__),
                    "source_path": str(finder.source),
                    "original_sha256": hashlib.sha256(finder.original.encode()).hexdigest(),
                    "modified_sha256": hashlib.sha256(finder.modified.encode()).hexdigest(),
                    "runtime_tolerance": module._DIRECT_RELAX_RESIDUAL_TOLERANCE,
                    "source_unchanged": True,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
