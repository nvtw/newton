"""Run the same coupled reference directly from certified initial rest.

Compose after start_analytical_equilibrium and the friction overlays. Skip the
native warm-up so it cannot replace the independently certified initial pose.
The original coupled solver and all post-solve physical checks remain intact.
"""

import hashlib
from pathlib import Path

from local_studies.colibri import two_body_condensed_gpu
from local_studies.colibri.two_body_condensed_adaptive import solve


def main():
    """Replace only the reference's initial-state selection and validate it."""
    path = Path(__file__).with_name("two_body_coupled_live.py")
    original = path.read_text()
    source = original.replace("for _ in range(330):", "for _ in range(0):")
    assert source != original and original.count("for _ in range(330):") == 1
    start = source.index("    ref = np.load(args.reference)")
    end = source.index("    history = [state.body_q.numpy()]", start)
    source = (
        source[:start]
        + """    certificate = np.load("/tmp/colibri_static_certificate_exact_flat-base_analytical_equilibrium.npz")
    initial_q = state.body_q.numpy()
    for index, label in enumerate(("FrameGround", "Frame")):
        body = list(model.body_label).index(label)
        np.testing.assert_array_equal(initial_q[body], certificate["q"][index].astype(np.float32))
    np.testing.assert_array_equal(state.body_qd.numpy(), np.zeros_like(state.body_qd.numpy()))
    gate = dict(analytical_initial_pose=True, zero_initial_velocity=True, native_warmup_skipped=True)
"""
        + source[end:]
    )
    source = source.replace(
        "            reference_only=True,",
        '            reference_only=True, initial_condition="Independent exact-plane equilibrium; no warm-up or impulse seed",',
    )
    marker = "        d = snapshot(world)\n"
    assert source.count(marker) == 1
    extra = """        if not records:
            birth = dict(d)
            birth["body_com"] = world.bodies.body_com.numpy()
            birth["shape_body"] = model.shape_body.numpy()
            views = world._contact_views
            for name in ("rigid_contact_count", "rigid_contact_shape0", "rigid_contact_shape1",
                         "rigid_contact_point0", "rigid_contact_point1", "rigid_contact_normal",
                         "rigid_contact_margin0", "rigid_contact_margin1"):
                birth[name] = getattr(views, name).numpy()
            np.savez(Path(args.output).with_suffix(".birth.npz"), **birth)
"""
    source = source.replace(marker, marker + extra)
    two_body_condensed_gpu.solve = solve
    print("ANALYTICAL_COUPLED_SOURCE", hashlib.sha256(original.encode()).hexdigest(), flush=True)
    try:
        exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
    finally:
        assert path.read_text() == original, "Reference implementation changed during diagnostic"


if __name__ == "__main__":
    main()
