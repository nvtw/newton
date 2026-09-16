"""Isolated source overlay: retain anchors until an actual friction projection breaks."""

import hashlib
import importlib.abc
import importlib.util
import json
import runpy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "newton._src.solvers.phoenx.constraints."


def sources():
    names = (
        "contact_container",
        "contact_ingest",
        "constraint_contact",
        "contact_projection",
        "constraint_contact_cloth",
    )
    original = {n: (ROOT / "newton/_src/solvers/phoenx/constraints" / f"{n}.py").read_text() for n in names}
    out = dict(original)
    out["contact_container"] = (
        out["contact_container"]
        .replace("CC_DWORDS_PER_CONTACT: int = 12", "CC_DWORDS_PER_CONTACT: int = 13")
        .replace("CC_RIGID_DWORDS_PER_CONTACT: int = 12", "CC_RIGID_DWORDS_PER_CONTACT: int = 13")
    )
    out["contact_ingest"] = (
        out["contact_ingest"]
        .replace("for row in range(6, 12):", "for row in range(6, 13):")
        .replace(
            "    # Friction references follow matched contact identity, independently of",
            "    cc.lambdas[12, k] = wp.float32(0.0)\n    # Friction references follow matched contact identity, independently of",
        )
    )
    out["contact_ingest"] = (
        out["contact_ingest"]
        .replace(
            "if prev_valid and wp.dot(cc_get_prev_normal(cc, prev_k), n) >= wp.float32(0.95):",
            "if prev_valid and cc.prev_lambdas[12, prev_k] == wp.float32(0.0) and wp.dot(cc_get_prev_normal(cc, prev_k), n) >= wp.float32(0.95):",
        )
        .replace(
            "    uses_start_gap = _contact_uses_stale_anchor_start_gap(contacts, k)",
            "    if prev_valid and cc.prev_lambdas[12, prev_k] != wp.float32(0.0):\n        lambda_t1 = wp.float32(0.0)\n        lambda_t2 = wp.float32(0.0)\n    uses_start_gap = _contact_uses_stale_anchor_start_gap(contacts, k)",
        )
    )
    out["constraint_contact"] = out["constraint_contact"].replace(
        "for row in range(6, 12):", "for row in range(6, 13):"
    )
    s = out["constraint_contact_cloth"]
    start = s.index("                lam_t1_prev = cc_get_tangent1_lambda(cc, k)")
    end = s.index("                if solver_gap >", start)
    s = s[:start] + s[end:]
    s = s.replace(" or saturated:", ":")
    out["constraint_contact_cloth"] = s
    s = out["contact_projection"]
    # Internal metric implementation returns its actual branch decision as a third component.
    start = s.index("def _make_contact_project_friction_metric(")
    end = s.index("def _make_contact_project_coupled_velocity_update(", start)
    chunk = s[start:end]
    chunk = chunk.replace(") -> wp.vec2f:", ") -> wp.vec3f:")
    chunk = chunk.replace(
        "if scale > wp.float32(0.0) and static_radius > wp.float32(0.0):", "if scale > wp.float32(0.0):"
    )
    chunk = chunk.replace(
        "        result = wp.vec2f(0.0)", "        broken = wp.float32(0.0)\n        result = wp.vec2f(0.0)", 1
    )
    chunk = chunk.replace(
        "            if wp.length_sq(result) > static_radius * static_radius:\n",
        "            if wp.length_sq(result) > static_radius * static_radius:\n                broken = wp.float32(1.0)\n",
    )
    chunk = chunk.replace("        return result", "        return wp.vec3f(result[0], result[1], broken)")
    chunk = chunk.replace("def contact_project_friction_metric(", "def contact_project_friction_metric_with_break(")
    # Keep all existing external callers and exact two-component result unchanged.
    signature = original["contact_projection"][
        original["contact_projection"].index("@wp.func\ndef contact_project_friction_metric(") : original[
            "contact_projection"
        ].index('    """Use float64 only', original["contact_projection"].index("def contact_project_friction_metric("))
    ]
    args = "mobility_t1, mobility_t1t2, mobility_t2, rhs_t1, rhs_t2, lambda_t1_old, lambda_t2_old, static_radius, dynamic_radius"
    chunk += (
        signature
        + f"    value = contact_project_friction_metric_with_break({args})\n    return wp.vec2f(value[0], value[1])\n\n\n"
    )
    s = s[:start] + chunk + s[end:]
    s = s.replace(
        "friction = contact_project_friction_metric(", "friction = contact_project_friction_metric_with_break("
    )
    # Record the same actual trial used by the diagonal projector; no impulse arithmetic changes.
    marker = "        tangents = block_project_friction_delta_sor_2("
    pieces = s.split(marker)
    rebuilt = pieces[0]
    for part in pieces[1:]:
        if "friction[0]" in rebuilt[-400:]:
            flag = "        cc.lambdas[12, k] = friction[2]\n"
        else:
            flag = "        cc.lambdas[12, k] = wp.float32(0.0)\n        trial = wp.vec2f(lam_t1_old + d_lambda_t1 * sor_boost, lam_t2_old + d_lambda_t2 * sor_boost)\n        if wp.length_sq(trial) > (mu_s * lambda_n_friction) * (mu_s * lambda_n_friction):\n            cc.lambdas[12, k] = wp.float32(1.0)\n"
        rebuilt += flag + marker + part
    s = rebuilt
    s = s.replace(
        "    tangent = block_project_friction_delta_sor_2(",
        "    cc.lambdas[12, k] = friction[2]\n    tangent = block_project_friction_delta_sor_2(",
    )
    out["contact_projection"] = s
    return original, out


class Overlay(importlib.abc.MetaPathFinder):
    def __init__(self, paths):
        self.paths = paths

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.paths:
            return importlib.util.spec_from_file_location(fullname, self.paths[fullname])
        return None


def install():
    import warp as wp

    wp.config.kernel_cache_dir = "/tmp/colibri_break_kernel_cache"
    original, modified = sources()
    assert not any(PACKAGE + n in sys.modules for n in original), "Install before importing Newton contact modules"
    directory = Path(tempfile.mkdtemp(prefix="colibri_friction_break_"))
    paths = {}
    for name, text in modified.items():
        path = directory / f"{name}.py"
        path.write_text(text)
        compile(text, str(path), "exec")
        paths[PACKAGE + name] = path
    finder = Overlay(paths)
    sys.meta_path.insert(0, finder)
    return original, modified, directory


if __name__ == "__main__":
    original, modified, directory = install()
    module = sys.argv[1]
    arguments = sys.argv[2:]
    sys.argv = [module, *arguments]
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        for name, text in original.items():
            assert (ROOT / "newton/_src/solvers/phoenx/constraints" / f"{name}.py").read_text() == text
        if "--output" in arguments:
            output = Path(arguments[arguments.index("--output") + 1])
            output.with_suffix(".friction_break.json").write_text(
                json.dumps(
                    {
                        "production_unchanged": True,
                        "overlay": str(directory),
                        "hashes": {n: hashlib.sha256(t.encode()).hexdigest() for n, t in modified.items()},
                        "arguments": arguments,
                    },
                    indent=2,
                )
            )


def clear_fresh_positive_warmstart(source):
    """Optional companion for identity-preserving matching; never enable attraction.

    This is deliberately not used by the flag-only control. It preserves material
    history while discarding carried force impulses for fresh positive gaps.
    """
    old = "    if uses_start_gap or reuse:\n"
    assert source.count(old) == 1
    source = source.replace(old, "    if True:\n")
    old = "        if reuse and gap > wp.float32(0.0):"
    assert source.count(old) == 1
    return source.replace(old, "        if gap > wp.float32(0.0):")
