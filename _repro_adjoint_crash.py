"""
Minimal repro: narrow_phase_primitive_kernel adjoint crashes at large grid sizes.

Key finding: block_dim (threads per block) does NOT cause the crash.
block_dim=1024 with 1 block works fine. The crash correlates with grid size
(total number of blocks), which should NOT matter for per-SM GPU resources.
This points to a bug in Warp's adjoint code generation or replay, not a
fundamental CUDA resource limit.
"""

import warp as wp
wp.init()
wp.config.verify_cuda = True

DEVICE = "cuda:0"

import newton
from newton._src.sim.collide import (
    prepare_geom_data_kernel, ContactWriterData, write_contact,
)
from newton._src.geometry.narrow_phase import (
    create_narrow_phase_primitive_kernel,
)

with wp.ScopedDevice(DEVICE):
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(ke=1e4, kd=10.0, mu=0.5)
    builder.default_shape_cfg = cfg
    builder.add_ground_plane()
    b = builder.add_body(
        xform=wp.transform((0.0, 0.0, 0.15), wp.quat_identity()),
    )
    builder.add_shape_sphere(body=b, radius=0.2)
    model = builder.finalize(requires_grad=True)
    state = model.state(requires_grad=True)

    prim = create_narrow_phase_primitive_kernel(
        write_contact, enable_backward=True,
    )

    gd = wp.zeros(model.shape_count, dtype=wp.vec4)
    gt = wp.zeros(model.shape_count, dtype=wp.transform, requires_grad=True)
    wp.launch(
        prepare_geom_data_kernel, dim=model.shape_count,
        inputs=[model.shape_transform, model.shape_body,
                model.shape_type, model.shape_scale,
                model.shape_margin, state.body_q],
        outputs=[gd, gt],
    )
    wp.synchronize()

    cp = wp.array([[0, 1]], dtype=wp.vec2i)
    cpc = wp.array([1], dtype=int)
    mx = 100
    gjk_p = wp.zeros(mx, dtype=wp.vec2i)
    gjk_c = wp.zeros(1, dtype=int)
    mp = wp.zeros(mx, dtype=wp.vec2i)
    mc = wp.zeros(1, dtype=int)
    mpp = wp.zeros(mx, dtype=wp.vec2i)
    mpc = wp.zeros(mx, dtype=int)
    mppc = wp.zeros(1, dtype=int)
    mpvc = wp.zeros(1, dtype=int)
    mmp = wp.zeros(mx, dtype=wp.vec2i)
    mmc = wp.zeros(1, dtype=int)
    ssp = wp.zeros(mx, dtype=wp.vec2i)
    ssc = wp.zeros(1, dtype=int)

    @wp.kernel
    def loss_k(n: wp.array(dtype=wp.vec3),
                c: wp.array(dtype=int),
                l: wp.array(dtype=float)):
        if c[0] > 0:
            l[0] = wp.dot(n[0], n[0])

    contacts = newton.Contacts(
        rigid_contact_max=100, soft_contact_max=0, requires_grad=True,
    )
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    wd = ContactWriterData()
    wd.contact_max = 100
    wd.body_q = state.body_q
    wd.shape_body = model.shape_body
    wd.shape_gap = model.shape_gap
    wd.contact_count = contacts.rigid_contact_count
    wd.out_shape0 = contacts.rigid_contact_shape0
    wd.out_shape1 = contacts.rigid_contact_shape1
    wd.out_point0 = contacts.rigid_contact_point0
    wd.out_point1 = contacts.rigid_contact_point1
    wd.out_offset0 = contacts.rigid_contact_offset0
    wd.out_offset1 = contacts.rigid_contact_offset1
    wd.out_normal = contacts.rigid_contact_normal
    wd.out_margin0 = contacts.rigid_contact_margin0
    wd.out_margin1 = contacts.rigid_contact_margin1
    wd.out_tids = contacts.rigid_contact_tids
    wd.out_stiffness = wp.zeros(0, dtype=float)
    wd.out_damping = wp.zeros(0, dtype=float)
    wd.out_friction = wp.zeros(0, dtype=float)

    DIM = 32768
    print(f"GPU: {wp.get_device(DEVICE).name}")
    print(f"Single test: dim={DIM}, block_dim=128, {DIM // 128} blocks")
    tape = wp.Tape()
    with tape:
        wp.launch(
            prim, dim=DIM,
            inputs=[cp, cpc, model.shape_type, gd, gt,
                    model.shape_source_ptr, model.shape_gap,
                    model.shape_flags, wd, DIM],
            outputs=[gjk_p, gjk_c, mp, mc, mpp, mpc, mppc, mpvc,
                     mmp, mmc, ssp, ssc],
            block_dim=128,
        )
        wp.launch(loss_k, dim=1,
                  inputs=[contacts.rigid_contact_normal,
                          contacts.rigid_contact_count, loss])
    wp.synchronize()
    print(f"Forward OK, contacts={contacts.rigid_contact_count.numpy()[0]}")

    try:
        tape.backward(loss)
        wp.synchronize()
        print(f"Backward OK at dim={DIM}")
    except Exception as e:
        print(f"Backward CRASHED at dim={DIM}: {e}")
