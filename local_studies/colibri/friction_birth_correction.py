"""Local rigid-anchor high/low material-coordinate correction; no dead zone."""

import hashlib
import json
import runpy
import sys
from pathlib import Path

from local_studies.colibri import friction_break_state

HELPERS = """

@wp.func
def _friction_normalized_orientation64(q: wp.quatf) -> wp.quatd:
    return wp.normalize(wp.quatd(wp.float64(q[0]), wp.float64(q[1]), wp.float64(q[2]), wp.float64(q[3])))


@wp.func
def _friction_material_point64(p: wp.vec3f, q: wp.quatf, com: wp.vec3f, a: wp.vec3f) -> wp.vec3d:
    local = wp.vec3d(a) - wp.vec3d(com)
    return wp.vec3d(p) + wp.quat_rotate(_friction_normalized_orientation64(q), local)


@wp.func
def cc_set_friction_birth_correction(cc: ContactContainer, k: wp.int32, p0: wp.vec3f, q0: wp.quatf, com0: wp.vec3f, p1: wp.vec3f, q1: wp.quatf, com1: wp.vec3f):
    a0 = cc_get_friction_anchor0(cc, k)
    a1 = cc_get_friction_anchor1(cc, k)
    error = _friction_material_point64(p1, q1, com1, a1) - _friction_material_point64(p0, q0, com0, a0)
    local_error = wp.quat_rotate_inv(_friction_normalized_orientation64(q0), error)
    for d in range(3):
        cc.lambdas[13 + d, k] = wp.float32(local_error[d])


@wp.func
def cc_get_friction_material_delta(cc: ContactContainer, k: wp.int32, p0: wp.vec3f, q0: wp.quatf, com0: wp.vec3f, p1: wp.vec3f, q1: wp.quatf, com1: wp.vec3f) -> wp.vec3f:
    error = _friction_material_point64(p1, q1, com1, cc_get_friction_anchor1(cc, k)) - _friction_material_point64(p0, q0, com0, cc_get_friction_anchor0(cc, k))
    local_error = wp.vec3d(wp.float64(cc.lambdas[13, k]), wp.float64(cc.lambdas[14, k]), wp.float64(cc.lambdas[15, k]))
    return wp.vec3f(error - wp.quat_rotate(_friction_normalized_orientation64(q0), local_error))
"""


def install():
    previous = friction_break_state.sources

    def sources():
        original, modified = previous()
        s = modified["contact_container"]
        assert "CC_DWORDS_PER_CONTACT: int = 13" in s
        s = s.replace("CC_DWORDS_PER_CONTACT: int = 13", "CC_DWORDS_PER_CONTACT: int = 16").replace(
            "CC_RIGID_DWORDS_PER_CONTACT: int = 13", "CC_RIGID_DWORDS_PER_CONTACT: int = 16"
        )
        modified["contact_container"] = s + HELPERS
        for name in ("contact_ingest", "constraint_contact"):
            modified[name] = modified[name].replace("for row in range(6, 13):", "for row in range(6, 16):")
        for name in ("contact_ingest", "constraint_contact_cloth"):
            s = modified[name]
            marker = "    cc_set_friction_anchor1,"
            assert marker in s
            s = s.replace(
                marker, marker + "\n    cc_set_friction_birth_correction,\n    cc_get_friction_material_delta,"
            )
            if name == "contact_ingest":
                marker = "            cc_set_friction_anchor1(cc, k, com2 + wp.quat_rotate_inv(q2, point - p2))"
                assert s.count(marker) == 1
                s = s.replace(
                    marker, marker + "\n            cc_set_friction_birth_correction(cc, k, p1, q1, com1, p2, q2, com2)"
                )
            else:
                old = """                friction_p0 = position1 + wp.quat_rotate(orientation1, cc_get_friction_anchor0(cc, k) - body_com1)
                friction_p1 = position2 + wp.quat_rotate(orientation2, cc_get_friction_anchor1(cc, k) - body_com2)
                p_diff = friction_p1 - friction_p0"""
                assert s.count(old) == 1
                s = s.replace(
                    old,
                    "                p_diff = cc_get_friction_material_delta(cc, k, position1, orientation1, body_com1, position2, orientation2, body_com2)",
                )
                marker = "                    cc_set_friction_anchor1(cc, k, anchor1)"
                assert s.count(marker) == 1
                s = s.replace(
                    marker,
                    marker
                    + "\n                    cc_set_friction_birth_correction(cc, k, position1, orientation1, body_com1, position2, orientation2, body_com2)",
                )
            modified[name] = s
        return original, modified

    friction_break_state.sources = sources
    return sources


def main():
    raise RuntimeError("Abandoned FP64 live candidate: use friction_differential_reference (FP32) instead")
    sources = install()
    original, modified = sources()
    module, *arguments = sys.argv[1:]
    sys.argv = [module, *arguments]
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        unchanged = all(
            (friction_break_state.ROOT / "newton/_src/solvers/phoenx/constraints" / f"{name}.py").read_text() == text
            for name, text in original.items()
        )
        if "--output" in arguments:
            output = Path(arguments[arguments.index("--output") + 1])
            output.with_suffix(".birth_correction.json").write_text(
                json.dumps(
                    dict(
                        production_unchanged=unchanged,
                        source_sha256={name: hashlib.sha256(s.encode()).hexdigest() for name, s in modified.items()},
                        scope="Rigid material anchors as FP32 high plus FP32 low correction, evaluated with normalized FP64 transforms; correction covaries with body0; no threshold/dead zone or force-law change",
                    ),
                    indent=2,
                )
            )
        assert unchanged


if __name__ == "__main__":
    main()
