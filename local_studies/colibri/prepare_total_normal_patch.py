"""Stage the independently tested single-velocity stabilized Coulomb load law."""

import difflib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
paths = [
    Path("newton/_src/solvers/phoenx") / p
    for p in (
        "constraints/contact_projection.py",
        "constraints/constraint_contact_cloth.py",
        "articulations/direct_contact_gs.py",
        "articulations/maximal_contact_gs.py",
    )
]
old = {p: (ROOT / p).read_text() for p in paths}
new = dict(old)
p = paths[0]
s = new[p].replace(
    "Normal load for Coulomb friction, excluding Baumgarte correction.",
    "Actual normal impulse supplying the single-velocity Coulomb cone.",
)
start = s.index("    load = lambda_n + mass_coeff_n * eff_n * bias_n * sor_boost")
end = s.index("\n\n", start)
s = (
    s[:start]
    + """    # The normal impulse, including stabilization, changes the real velocity.
    # Its actual nonnegative multiplier supplies the single-channel Coulomb cap.
    # Row-diagonal bias subtraction cannot separate coupled recovery impulses.
    return wp.max(lambda_n, wp.float32(0.0))"""
    + s[end:]
)
new[p] = s
p = paths[1]
s = new[p]
start = s.index(
    "                    if pd_eff_soft_n <= wp.float32(0.0):\n",
    s.index("                    lambda_n_load = cc_get_normal_lambda(cc, k)"),
)
end = s.index("                    if is_speculative", start)
new[p] = s[:start] + s[end:]
for p in paths[2:]:
    s = new[p]
    s, n = re.subn(
        r"(?m)^( +)if pd_eff <= wp.float32\(0.0\):\n\1    normal_load \+= .*\n\1    normal_load = wp.clamp\(normal_load, wp.float32\(0.0\), normal_lambda\)\n",
        "",
        s,
    )
    assert n == 1, (p, n)
    new[p] = s
p = Path("newton/_src/solvers/phoenx/tests/test_stabilized_coulomb_support.py")
new[p] = (ROOT / "local_studies/colibri/test_stabilized_coulomb_support.py").read_text()
old[p] = ""
destination = Path("/tmp/colibri_total_normal_production_proposal")
for p, s in new.items():
    target = destination / p
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(s)
    compile(s, str(target), "exec")
patch = "".join(
    "".join(
        difflib.unified_diff(old[p].splitlines(True), s.splitlines(True), fromfile="a/" + str(p), tofile="b/" + str(p))
    )
    for p, s in new.items()
)
Path("/tmp/colibri_total_normal_production.patch").write_text(patch)
print(destination)
