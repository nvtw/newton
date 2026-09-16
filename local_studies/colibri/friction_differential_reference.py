"""Local all-FP32 differential material-reference prototype; no dead zone."""

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

from local_studies.colibri import friction_break_state

HELPERS = """
@wp.func
def _friction_conjugate32(q: wp.quatf) -> wp.quatf:
    return wp.quatf(-q[0], -q[1], -q[2], q[3])

@wp.func
def _friction_rotate32(q: wp.quatf, point: wp.vec3f) -> wp.vec3f:
    vector = wp.vec3f(q[0], q[1], q[2])
    cross = wp.cross(vector, point)
    return point + wp.float32(2.0) * (q[3] * cross + wp.cross(vector, cross))

@wp.func
def _friction_rotation_increment32(q: wp.quatf, delta: wp.quatf, point: wp.vec3f) -> wp.vec3f:
    midpoint = q + delta * wp.float32(0.5)
    vm = wp.vec3f(midpoint[0], midpoint[1], midpoint[2])
    dv = wp.vec3f(delta[0], delta[1], delta[2])
    return wp.float32(2.0) * (delta[3] * wp.cross(vm, point) + midpoint[3] * wp.cross(dv, point) + wp.cross(dv, wp.cross(vm, point)) + wp.cross(vm, wp.cross(dv, point)))

@wp.func
def cc_set_friction_reference_pose(cc: ContactContainer, k: wp.int32, p0: wp.vec3f, q0: wp.quatf, com0: wp.vec3f, p1: wp.vec3f, q1: wp.quatf, com1: wp.vec3f):
    for d in range(3):
        cc.lambdas[13+d,k] = p0[d]
        cc.lambdas[16+d,k] = p1[d]
    for d in range(4):
        cc.lambdas[19+d,k] = q0[d]
        cc.lambdas[23+d,k] = q1[d]

@wp.func
def cc_get_friction_reference_delta(cc: ContactContainer, k: wp.int32, p0: wp.vec3f, q0: wp.quatf, com0: wp.vec3f, p1: wp.vec3f, q1: wp.quatf, com1: wp.vec3f) -> wp.vec3f:
    p0b = wp.vec3f(cc.lambdas[13,k],cc.lambdas[14,k],cc.lambdas[15,k])
    p1b = wp.vec3f(cc.lambdas[16,k],cc.lambdas[17,k],cc.lambdas[18,k])
    q0b = wp.quatf(cc.lambdas[19,k],cc.lambdas[20,k],cc.lambdas[21,k],cc.lambdas[22,k])
    q1b = wp.quatf(cc.lambdas[23,k],cc.lambdas[24,k],cc.lambdas[25,k],cc.lambdas[26,k])
    dq0 = q0-q0b
    dq1 = q1-q1b
    rel_birth = _friction_conjugate32(q0b)*q1b
    delta_rel = _friction_conjugate32(dq0)*q1b + _friction_conjugate32(q0)*dq1
    delta_position = (p1-p1b)-(p0-p0b)
    delta_translation = _friction_rotate32(_friction_conjugate32(q0),delta_position) + _friction_rotation_increment32(_friction_conjugate32(q0b),_friction_conjugate32(dq0),p1b-p0b)
    local = delta_translation + _friction_rotation_increment32(rel_birth,delta_rel,cc_get_friction_anchor1(cc,k)-com1)
    return _friction_rotate32(q0,local)
"""


def selected_helpers():
    """Keep raw and normalized-increment experiments separately selectable."""
    if os.environ.get("COLIBRI_DIFFERENTIAL_NORMALIZED") != "1":
        return HELPERS
    helpers = HELPERS.replace("def _friction_rotation_increment32(", "def _friction_rotation_increment_raw32(")
    helpers = helpers.replace(
        "return point + wp.float32(2.0) * (q[3] * cross + wp.cross(vector, cross))",
        "return point + (wp.float32(2.0) / wp.dot(q, q)) * (q[3] * cross + wp.cross(vector, cross))",
    )
    marker = "@wp.func\ndef cc_set_friction_reference_pose"
    normalized = """@wp.func
def _friction_rotation_increment32(q: wp.quatf, delta: wp.quatf, point: wp.vec3f) -> wp.vec3f:
    norm_birth = wp.dot(q, q)
    norm_delta = wp.float32(2.0) * wp.dot(q, delta) + wp.dot(delta, delta)
    vector = wp.vec3f(q[0], q[1], q[2])
    cross = wp.cross(vector, point)
    birth_correction = wp.float32(2.0) * (q[3] * cross + wp.cross(vector, cross))
    numerator = _friction_rotation_increment_raw32(q, delta, point) - birth_correction * (norm_delta / norm_birth)
    return numerator / (norm_birth + norm_delta)


"""
    assert marker in helpers
    return helpers.replace(marker, normalized + marker)


def install():
    previous = friction_break_state.sources

    def sources():
        original, modified = previous()
        s = modified["contact_container"]
        assert "CC_DWORDS_PER_CONTACT: int = 13" in s
        s = s.replace("CC_DWORDS_PER_CONTACT: int = 13", "CC_DWORDS_PER_CONTACT: int = 27").replace(
            "CC_RIGID_DWORDS_PER_CONTACT: int = 13", "CC_RIGID_DWORDS_PER_CONTACT: int = 27"
        )
        modified["contact_container"] = s + selected_helpers()
        for name in ("contact_ingest", "constraint_contact"):
            modified[name] = modified[name].replace("for row in range(6, 13):", "for row in range(6, 27):")
        for name in ("contact_ingest", "constraint_contact_cloth"):
            s = modified[name]
            marker = "    cc_set_friction_anchor1,"
            assert marker in s
            s = s.replace(
                marker, marker + "\n    cc_set_friction_reference_pose,\n    cc_get_friction_reference_delta,"
            )
            if name == "contact_ingest":
                marker = "            cc_set_friction_anchor1(cc, k, com2 + wp.quat_rotate_inv(q2, point - p2))"
                assert s.count(marker) == 1
                s = s.replace(
                    marker, marker + "\n            cc_set_friction_reference_pose(cc, k, p1, q1, com1, p2, q2, com2)"
                )
            else:
                old = """                friction_p0 = position1 + wp.quat_rotate(orientation1, cc_get_friction_anchor0(cc, k) - body_com1)
                friction_p1 = position2 + wp.quat_rotate(orientation2, cc_get_friction_anchor1(cc, k) - body_com2)
                p_diff = friction_p1 - friction_p0"""
                assert s.count(old) == 1
                s = s.replace(
                    old,
                    "                p_diff = cc_get_friction_reference_delta(cc, k, position1, orientation1, body_com1, position2, orientation2, body_com2)",
                )
                marker = "                    cc_set_friction_anchor1(cc, k, anchor1)"
                assert s.count(marker) == 1
                s = s.replace(
                    marker,
                    marker
                    + "\n                    cc_set_friction_reference_pose(cc, k, position1, orientation1, body_com1, position2, orientation2, body_com2)",
                )
            modified[name] = s
        return original, modified

    friction_break_state.sources = sources
    return sources


def main():
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
            output.with_suffix(".differential_reference.json").write_text(
                json.dumps(
                    {
                        "production_unchanged": unchanged,
                        "source_sha256": {name: hashlib.sha256(s.encode()).hexdigest() for name, s in modified.items()},
                        "scope": "All-FP32 differential shared material reference;14 persistent reference floats; same-code zero birth; native force law and common-point impulse unchanged",
                    },
                    indent=2,
                )
            )
        assert unchanged


if __name__ == "__main__":
    main()
