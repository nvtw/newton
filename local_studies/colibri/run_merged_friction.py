"""Run a module through the staged proposal, with immutable source manifests."""

import hashlib
import importlib.abc
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = Path("/tmp/colibri_merged_friction_proposal")


class StagedFinder(importlib.abc.MetaPathFinder):
    def __init__(self, paths):
        self.paths = paths

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.paths:
            return importlib.util.spec_from_file_location(fullname, self.paths[fullname])
        return None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Install the overlay before importing Newton and retain ordinary module argv."""
    manifest = json.loads((STAGE / "manifest.json").read_text())
    for relative, expected in manifest["original"].items():
        path = ROOT / relative
        assert (digest(path) if path.exists() else None) == expected, relative
    paths = {}
    for relative, expected in manifest["staged"].items():
        path = STAGE / relative
        assert digest(path) == expected, relative
        name = relative[:-3].replace("/", ".")
        assert name not in sys.modules, name
        paths[name] = path
    # All runtime dependencies and authored example assets remain canonical.
    guarded = [*sorted((ROOT / "newton/_src").rglob("*.py")), *sorted((ROOT / "newton/examples").rglob("*.py"))]
    guarded.extend(p for p in (ROOT / "newton/examples/assets/colibri").rglob("*") if p.is_file())
    before = {str(p.relative_to(ROOT)): digest(p) for p in guarded}
    sys.meta_path.insert(0, StagedFinder(paths))
    import warp as wp  # noqa: PLC0415 - install staged imports before initializing dependencies

    wp.config.kernel_cache_dir = "/tmp/colibri_merged_friction_kernel_cache"
    module = sys.argv[1]
    arguments = sys.argv[2:]
    sys.argv = [module, *arguments]
    report_path = Path(os.environ.get("COLIBRI_MERGED_MANIFEST", "/tmp/colibri_merged_friction_run.json"))
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        after = {str(p.relative_to(ROOT)): digest(p) for p in guarded}
        stage_after = {relative: digest(STAGE / relative) for relative in manifest["staged"]}
        loaded = {name: str(sys.modules[name].__file__) for name in paths if name in sys.modules}
        source_ok = before == after
        stage_ok = stage_after == manifest["staged"]
        report_path.write_text(
            json.dumps(
                {
                    "module": module,
                    "arguments": arguments,
                    "production_unchanged": source_ok,
                    "stage_unchanged": stage_ok,
                    "staged_modules_loaded": loaded,
                    "production_sha256": before,
                    "stage_sha256": stage_after,
                },
                indent=2,
            )
        )
        assert source_ok, "Canonical dependency or scene changed"
        assert stage_ok, "Staged proposal changed"
        for name, path in loaded.items():
            assert Path(path).resolve() == paths[name].resolve(), name


if __name__ == "__main__":
    main()
