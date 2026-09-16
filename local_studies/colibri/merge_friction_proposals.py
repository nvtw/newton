"""Merge the two reviewed proposals into an isolated, importable staged tree."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/tmp/colibri_friction_break_production_proposal")
DESTINATION = Path("/tmp/colibri_merged_friction_proposal")
files = {p.relative_to(SOURCE): p.read_text() for p in SOURCE.rglob("*.py")}
base = Path("newton/_src/solvers/phoenx")
p = base / "constraints/contact_projection.py"
s = files[p]
start = s.index("    load = lambda_n + mass_coeff_n * eff_n * bias_n * sor_boost")
end = s.index("\n\n", start)
s = (
    s[:start]
    + """    # Stabilization changes real velocity in this single-channel solver.
    # A row-diagonal estimate cannot separate its coupled normal impulses.
    return wp.max(lambda_n, wp.float32(0.0))"""
    + s[end:]
)
s = s.replace(
    "Normal load for Coulomb friction, excluding Baumgarte correction.",
    "Actual normal impulse supplying the single-velocity Coulomb cone.",
)
# The metric minimizer may stick but an overrelaxed final trial can still clip.
needle = "        cc.lambdas[CC_FRICTION_BROKEN, k] = friction[2]\n"
replacement = (
    needle
    + """        final_trial = wp.vec2f(lam_t1_old + d_lambda_t1 * sor_boost, lam_t2_old + d_lambda_t2 * sor_boost)
        if wp.length_sq(final_trial) > (mu_s * lambda_n_friction) * (mu_s * lambda_n_friction):
            cc.lambdas[CC_FRICTION_BROKEN, k] = wp.float32(1.0)
"""
)
assert s.count(needle) == 1
s = s.replace(needle, replacement)
needle = "    cc.lambdas[CC_FRICTION_BROKEN, k] = friction[2]\n"
# Anchor at the uniquely named tangent-only entry point, not nested functions.
start = s.index("def contact_project_tangent_delta(")
tail = s[start:]
assert tail.count(needle) == 1
tail = tail.replace(
    needle,
    needle
    + """    final_trial = wp.vec2f(old1 + (friction[0] - old1) * sor, old2 + (friction[1] - old2) * sor)
    if wp.length_sq(final_trial) > (mu_s * normal_load) * (mu_s * normal_load):
        cc.lambdas[CC_FRICTION_BROKEN, k] = wp.float32(1.0)
""",
)
files[p] = s[:start] + tail
p = base / "constraints/constraint_contact_cloth.py"
s = files[p]
start = s.index(
    "                    if pd_eff_soft_n <= wp.float32(0.0):\n",
    s.index("                    lambda_n_load = cc_get_normal_lambda(cc, k)"),
)
end = s.index("                    if is_speculative", start)
s = s[:start] + s[end:]
# Only true patch rows retain their old separate history policy. Patch-ineligible
# columns take the corrected ordinary point path.
start = s.index("                saturated = wp.bool(False)")
end = s.index("                if solver_gap >", start)
chunk = s[start:end]
chunk = chunk.replace(
    "                if wp.static(patch_friction):\n",
    "                if wp.static(patch_friction):\n                    if use_patch:\n",
)
lines = chunk.splitlines(True)
for i, line in enumerate(lines):
    if i >= 3:
        lines[i] = "    " + line
s = s[:start] + "".join(lines) + s[end:]
files[p] = s
for name in ("direct_contact_gs", "maximal_contact_gs"):
    p = base / "articulations" / f"{name}.py"
    s, n = re.subn(
        r"(?m)^( +)if pd_eff <= wp.float32\(0.0\):\n\1    normal_load \+= .*\n\1    normal_load = wp.clamp\(normal_load, wp.float32\(0.0\), normal_lambda\)\n",
        "",
        files[p],
    )
    assert n == 1
    files[p] = s
p = base / "articulations/reduced_contact_block.py"
s = (ROOT / p).read_text()
old = """                        load = normal_update.lambda_new + row_mass_coeff * effective_mass * bias * sor_boost
                        normal_load[packed_articulation, point] = wp.clamp(
                            load, wp.float32(0.0), normal_update.lambda_new
                        )"""
assert s.count(old) == 1
files[p] = s.replace(
    old,
    "                        normal_load[packed_articulation, point] = wp.max(normal_update.lambda_new, wp.float32(0.0))",
)
p = base / "rl_training/g1_diagnostics.py"
s = (ROOT / p).read_text()
start = s.index("    _bias_rate, mass_coeff, _impulse_coeff = soft_constraint_coefficients(")
end = s.index("    friction_load_ratio =", start)
files[p] = s[:start] + "    friction_load = wp.max(normal_impulse, wp.float32(0.0))\n" + s[end:]
p = base / "tests/test_stabilized_coulomb_support.py"
files[p] = (ROOT / "local_studies/colibri/test_stabilized_coulomb_support.py").read_text()
p = base / "tests/test_contact_coupling.py"
s = (ROOT / p).read_text()
s = s.replace(
    "from newton._src.solvers.phoenx.constraints.contact_container import ContactContainer",
    "from newton._src.solvers.phoenx.constraints.contact_container import CC_DWORDS_PER_CONTACT, ContactContainer",
)
files[p] = s.replace(
    "        cc = ContactContainer()\n",
    '        cc = ContactContainer()\n        cc.lambdas = wp.zeros((CC_DWORDS_PER_CONTACT, 1), dtype=wp.float32, device="cpu")\n',
)
for p, source_text in list(files.items()):
    s = source_text
    if "/tests/" in str(p):
        s = re.sub(r": wp.array\(dtype=([^\)]+)\)", r": wp.array[\1]", s)
        lines = s.splitlines(True)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].lstrip().startswith("def test_") and not lines[i + 1].lstrip().startswith('"""'):
                title = lines[i].strip().split("(")[0][9:].replace("_", " ")
                lines.insert(i + 1, '        """Verify ' + title + '."""\n')
        s = "".join(lines)
        if not s.startswith("# SPDX-"):
            s = (
                "# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers\n# SPDX-License-Identifier: Apache-2.0\n"
                + s
            )
        files[p] = s
for p, s in files.items():
    target = DESTINATION / p
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(s)
    compile(s, str(target), "exec")
manifest = {
    "root": str(ROOT),
    "staged_root": str(DESTINATION),
    "staged": {str(p): hashlib.sha256(s.encode()).hexdigest() for p, s in files.items()},
    "original": {
        str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() if (ROOT / p).exists() else None for p in files
    },
}
(DESTINATION / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(DESTINATION)
