"""Stage a reviewable production diff without changing any runtime source."""

import difflib
from pathlib import Path

from local_studies.colibri.friction_break_state import ROOT, sources

original, changed = sources()
base = Path("newton/_src/solvers/phoenx/constraints")
files = {base / f"{key}.py": value for key, value in changed.items()}
old = {base / f"{key}.py": value for key, value in original.items()}
container = base / "contact_container.py"
files[container] = (
    files[container]
    .replace(
        "#: 12 = normal(3) + tangent1(3) + side0_bary(3) + side1_bary(3). The two",
        "#: 13 = normal(3) + tangent1(3) + side0_bary(3) + side1_bary(3) + friction break. The two",
    )
    .replace(
        "#: Frame directions plus two body-local friction anchors. Collision witnesses",
        "#: Frame directions, two body-local friction anchors, and latest friction break. Collision witnesses",
    )
)
files[container] = files[container].replace(
    "# Compile-time dword offsets.",
    "# Latest actual tangent projection exceeded its static cone. Matched ingest\n"
    "# consumes this state at contact refresh; microstep preparation retains it.\n"
    "CC_FRICTION_BROKEN: int = 12\n\n# Compile-time dword offsets.",
)
for path, source_text in files.items():
    text = source_text
    if path != container:
        if ".lambdas[12," in text or ".prev_lambdas[12," in text:
            text = text.replace(
                "from newton._src.solvers.phoenx.constraints.contact_container import (",
                "from newton._src.solvers/phoenx/constraints/contact_container import (".replace("/", ".")
                + "\n    CC_FRICTION_BROKEN,",
            )
            text = text.replace(".lambdas[12,", ".lambdas[CC_FRICTION_BROKEN,").replace(
                ".prev_lambdas[12,", ".prev_lambdas[CC_FRICTION_BROKEN,"
            )
        files[path] = text
# Retain the existing patch-friction preparation policy until its independent
# patch warm-history lifecycle is migrated and tested. Point friction is fixed.
cloth = base / "constraint_contact_cloth.py"
start = original["constraint_contact_cloth"].index("                lam_t1_prev = cc_get_tangent1_lambda(cc, k)")
end = original["constraint_contact_cloth"].index("                if solver_gap >", start)
legacy = original["constraint_contact_cloth"][start:end]
legacy = "".join("    " + line if line.strip() else line for line in legacy.splitlines(keepends=True))
needle = "                if solver_gap > wp.float32(0.0) or drift_sq > slip_threshold * slip_threshold:"
files[cloth] = files[cloth].replace(
    needle,
    "                saturated = wp.bool(False)\n"
    "                if wp.static(patch_friction):\n" + legacy + needle[:-1] + " or saturated:",
)
# Articulation-owned point solvers use the same metric operation and now publish
# the same actual projection decision, without changing their impulse response.
for module in ("direct_contact_gs", "maximal_contact_gs"):
    path = Path("newton/_src/solvers/phoenx/articulations") / f"{module}.py"
    text = (ROOT / path).read_text()
    old[path] = text
    text = text.replace("contact_project_friction_metric", "contact_project_friction_metric_with_break")
    text = text.replace(
        "from newton._src.solvers.phoenx.constraints.contact_container import (",
        "from newton._src.solvers.phoenx.constraints.contact_container import (\n    CC_FRICTION_BROKEN,",
    )
    if module == "direct_contact_gs":
        text = text.replace(
            "                        tangent_delta = tangent_new - tangent_old",
            "                        contacts.lambdas[CC_FRICTION_BROKEN, contact] = tangent_new[2]\n"
            "                        tangent_delta = wp.vec2f(tangent_new[0], tangent_new[1]) - tangent_old",
        )
    else:
        text = text.replace(
            "                    cc_set_tangent1_lambda(contacts, contact, tangents[0])",
            "                    contacts.lambdas[CC_FRICTION_BROKEN, contact] = tangents[2]\n"
            "                    cc_set_tangent1_lambda(contacts, contact, tangents[0])",
        )
    files[path] = text
path = Path("newton/_src/solvers/phoenx/tests/test_friction_break_state.py")
files[path] = (
    (ROOT / "local_studies/colibri/test_friction_break_state.py")
    .read_text()
    .replace(
        '"""Diagnostic: native preparation erases history at99% of the actual static cone. This asserts the observed premature reset, not acceptance of correct sticking."""',
        '"""Retain sticking material history and consume actual sliding decisions at refresh."""',
    )
    .replace(
        '"""Reset a broken material reference without changing collision witnesses."""',
        '"""Microstep preparation must retain interior anchors and pending break history."""',
    )
)
old[path] = ""
path = Path("newton/_src/solvers/phoenx/tests/test_rigid_friction_anchors.py")
old[path] = (ROOT / path).read_text()
files[path] = (
    old[path]
    .replace("point[0, 0] += 1.0e-5", "point[0, 0] += 0.003", 1)
    .replace(
        "Reset a broken material reference without changing collision witnesses.",
        "Reset a geometrically broken material reference without changing collision witnesses.",
    )
)
destination = Path("/tmp/colibri_friction_break_production_proposal")
for path, text in files.items():
    target = destination / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    compile(text, str(target), "exec")
patch = "".join(
    "".join(
        difflib.unified_diff(
            old[path].splitlines(True), text.splitlines(True), fromfile="a/" + str(path), tofile="b/" + str(path)
        )
    )
    for path, text in files.items()
)
Path("/tmp/colibri_friction_break_production.patch").write_text(patch)
print(destination)
print("Files", len(files), "patch lines", len(patch.splitlines()))
