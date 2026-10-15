# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Upper bound for packing body scalars into the w lane of PhoenX's vec3 body state.

Mimics the TGS contact iterate body access: each thread owns one contact pair,
loads velocity, angular velocity, inverse mass and the packed world inverse
inertia of both bodies, applies an impulse and stores both velocities back.
Layouts:

* ``current``: ``vec3`` velocity / angular velocity, separate ``float`` inverse
  mass, ``vec6`` inertia (as in ``BodyContainer`` today).
* ``packed``: ``vec4`` (velocity, inverse mass) and ``vec4`` (angular velocity,
  unused) read and written with one 16-byte access each, inertia as three
  8-byte loads.
* ``current+raw access``: the current layout read and written through raw
  ``func_native`` accessors, to separate Warp's array-indexing overhead from the
  layout itself.

Pairs are bodies with nearby indices, visited in colored (shuffled) order like a
graph-colored sweep. Run: ``uv run python local_studies/phoenx_body_packing/body_layout_bench.py``.
"""

import numpy as np
import warp as wp

vec6 = wp.types.vector(length=6, dtype=wp.float32)


@wp.func_native("""
const float4 v = reinterpret_cast<const float4*>(arr.data)[i];
return wp::vec4(v.x, v.y, v.z, v.w);
""")
def load4(arr: wp.array[wp.vec4], i: wp.int32) -> wp.vec4: ...


@wp.func_native("""
reinterpret_cast<float4*>(arr.data)[i] = make_float4(v[0], v[1], v[2], v[3]);
""")
def store4(arr: wp.array[wp.vec4], i: wp.int32, v: wp.vec4): ...


@wp.func_native("""
const float2* p = reinterpret_cast<const float2*>(arr.data) + 3 * (size_t)i;
const float2 a = p[0];
const float2 b = p[1];
const float2 c = p[2];
wp::vec_t<6, wp::float32> r;
r.c[0] = a.x; r.c[1] = a.y; r.c[2] = b.x; r.c[3] = b.y; r.c[4] = c.x; r.c[5] = c.y;
return r;
""")
def load6(arr: wp.array[vec6], i: wp.int32) -> vec6: ...


@wp.func_native("""
const float* p = reinterpret_cast<const float*>(arr.data) + 3 * (size_t)i;
return wp::vec3(p[0], p[1], p[2]);
""")
def load3(arr: wp.array[wp.vec3], i: wp.int32) -> wp.vec3: ...


@wp.func_native("""
float* p = reinterpret_cast<float*>(arr.data) + 3 * (size_t)i;
p[0] = v[0];
p[1] = v[1];
p[2] = v[2];
""")
def store3(arr: wp.array[wp.vec3], i: wp.int32, v: wp.vec3): ...


@wp.func_native("""
return reinterpret_cast<const float*>(arr.data)[i];
""")
def load1(arr: wp.array[float], i: wp.int32) -> float: ...


@wp.func
def apply_inertia(s: vec6, x: wp.vec3) -> wp.vec3:
    return wp.vec3(
        s[0] * x[0] + s[3] * x[1] + s[4] * x[2],
        s[3] * x[0] + s[1] * x[1] + s[5] * x[2],
        s[4] * x[0] + s[5] * x[1] + s[2] * x[2],
    )


@wp.kernel
def iterate_current(
    pairs: wp.array[wp.vec2i],
    r: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    w: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    inertia: wp.array[vec6],
):
    t = wp.tid()
    p = pairs[t]
    v0 = v[p[0]]
    w0 = w[p[0]]
    v1 = v[p[1]]
    w1 = w[p[1]]
    m0 = inv_mass[p[0]]
    m1 = inv_mass[p[1]]
    i0 = inertia[p[0]]
    i1 = inertia[p[1]]
    arm = r[t]
    rel = v1 + wp.cross(w1, arm) - v0 - wp.cross(w0, arm)
    impulse = -0.5 * rel / (m0 + m1 + 1.0e-6)
    v[p[0]] = v0 - m0 * impulse
    w[p[0]] = w0 - apply_inertia(i0, wp.cross(arm, impulse))
    v[p[1]] = v1 + m1 * impulse
    w[p[1]] = w1 + apply_inertia(i1, wp.cross(arm, impulse))


@wp.kernel
def iterate_current_raw(
    pairs: wp.array[wp.vec2i],
    r: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    w: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    inertia: wp.array[vec6],
):
    t = wp.tid()
    p = pairs[t]
    v0 = load3(v, p[0])
    w0 = load3(w, p[0])
    v1 = load3(v, p[1])
    w1 = load3(w, p[1])
    m0 = load1(inv_mass, p[0])
    m1 = load1(inv_mass, p[1])
    i0 = load6(inertia, p[0])
    i1 = load6(inertia, p[1])
    arm = r[t]
    rel = v1 + wp.cross(w1, arm) - v0 - wp.cross(w0, arm)
    impulse = -0.5 * rel / (m0 + m1 + 1.0e-6)
    store3(v, p[0], v0 - m0 * impulse)
    store3(w, p[0], w0 - apply_inertia(i0, wp.cross(arm, impulse)))
    store3(v, p[1], v1 + m1 * impulse)
    store3(w, p[1], w1 + apply_inertia(i1, wp.cross(arm, impulse)))


@wp.kernel
def iterate_packed(
    pairs: wp.array[wp.vec2i],
    r: wp.array[wp.vec3],
    vm: wp.array[wp.vec4],
    wx: wp.array[wp.vec4],
    inertia: wp.array[vec6],
):
    t = wp.tid()
    p = pairs[t]
    a0 = load4(vm, p[0])
    b0 = load4(wx, p[0])
    a1 = load4(vm, p[1])
    b1 = load4(wx, p[1])
    v0 = wp.vec3(a0[0], a0[1], a0[2])
    w0 = wp.vec3(b0[0], b0[1], b0[2])
    v1 = wp.vec3(a1[0], a1[1], a1[2])
    w1 = wp.vec3(b1[0], b1[1], b1[2])
    m0 = a0[3]
    m1 = a1[3]
    i0 = load6(inertia, p[0])
    i1 = load6(inertia, p[1])
    arm = r[t]
    rel = v1 + wp.cross(w1, arm) - v0 - wp.cross(w0, arm)
    impulse = -0.5 * rel / (m0 + m1 + 1.0e-6)
    v0 = v0 - m0 * impulse
    w0 = w0 - apply_inertia(i0, wp.cross(arm, impulse))
    v1 = v1 + m1 * impulse
    w1 = w1 + apply_inertia(i1, wp.cross(arm, impulse))
    store4(vm, p[0], wp.vec4(v0[0], v0[1], v0[2], m0))
    store4(wx, p[0], wp.vec4(w0[0], w0[1], w0[2], b0[3]))
    store4(vm, p[1], wp.vec4(v1[0], v1[1], v1[2], m1))
    store4(wx, p[1], wp.vec4(w1[0], w1[1], w1[2], b1[3]))


@wp.func
def sym6_from(a: wp.vec4, b4: float, b5: float) -> vec6:
    return vec6(a[0], a[1], a[2], a[3], b4, b5)


@wp.kernel
def iterate_split_position(
    pairs: wp.array[wp.vec2i],
    r: wp.array[wp.vec3],
    vm: wp.array[wp.vec4],
    wx: wp.array[wp.vec4],
    inertia4: wp.array[wp.vec4],
    px: wp.array[wp.vec4],
):
    """Proposal: inertia 0-3 in a vec4, 4 in angular_velocity.w, 5 in position.w."""
    t = wp.tid()
    p = pairs[t]
    a0 = load4(vm, p[0])
    b0 = load4(wx, p[0])
    a1 = load4(vm, p[1])
    b1 = load4(wx, p[1])
    i0 = sym6_from(load4(inertia4, p[0]), b0[3], load4(px, p[0])[3])
    i1 = sym6_from(load4(inertia4, p[1]), b1[3], load4(px, p[1])[3])
    v0 = wp.vec3(a0[0], a0[1], a0[2])
    w0 = wp.vec3(b0[0], b0[1], b0[2])
    v1 = wp.vec3(a1[0], a1[1], a1[2])
    w1 = wp.vec3(b1[0], b1[1], b1[2])
    m0 = a0[3]
    m1 = a1[3]
    arm = r[t]
    rel = v1 + wp.cross(w1, arm) - v0 - wp.cross(w0, arm)
    impulse = -0.5 * rel / (m0 + m1 + 1.0e-6)
    v0 = v0 - m0 * impulse
    w0 = w0 - apply_inertia(i0, wp.cross(arm, impulse))
    v1 = v1 + m1 * impulse
    w1 = w1 + apply_inertia(i1, wp.cross(arm, impulse))
    store4(vm, p[0], wp.vec4(v0[0], v0[1], v0[2], m0))
    store4(wx, p[0], wp.vec4(w0[0], w0[1], w0[2], b0[3]))
    store4(vm, p[1], wp.vec4(v1[0], v1[1], v1[2], m1))
    store4(wx, p[1], wp.vec4(w1[0], w1[1], w1[2], b1[3]))


@wp.kernel
def iterate_split_scalar(
    pairs: wp.array[wp.vec2i],
    r: wp.array[wp.vec3],
    vm: wp.array[wp.vec4],
    wx: wp.array[wp.vec4],
    inertia4: wp.array[wp.vec4],
    last: wp.array[float],
):
    """Inertia 0-3 in a vec4, 4 in angular_velocity.w, 5 in its own float array."""
    t = wp.tid()
    p = pairs[t]
    a0 = load4(vm, p[0])
    b0 = load4(wx, p[0])
    a1 = load4(vm, p[1])
    b1 = load4(wx, p[1])
    i0 = sym6_from(load4(inertia4, p[0]), b0[3], load1(last, p[0]))
    i1 = sym6_from(load4(inertia4, p[1]), b1[3], load1(last, p[1]))
    v0 = wp.vec3(a0[0], a0[1], a0[2])
    w0 = wp.vec3(b0[0], b0[1], b0[2])
    v1 = wp.vec3(a1[0], a1[1], a1[2])
    w1 = wp.vec3(b1[0], b1[1], b1[2])
    m0 = a0[3]
    m1 = a1[3]
    arm = r[t]
    rel = v1 + wp.cross(w1, arm) - v0 - wp.cross(w0, arm)
    impulse = -0.5 * rel / (m0 + m1 + 1.0e-6)
    v0 = v0 - m0 * impulse
    w0 = w0 - apply_inertia(i0, wp.cross(arm, impulse))
    v1 = v1 + m1 * impulse
    w1 = w1 + apply_inertia(i1, wp.cross(arm, impulse))
    store4(vm, p[0], wp.vec4(v0[0], v0[1], v0[2], m0))
    store4(wx, p[0], wp.vec4(w0[0], w0[1], w0[2], b0[3]))
    store4(vm, p[1], wp.vec4(v1[0], v1[1], v1[2], m1))
    store4(wx, p[1], wp.vec4(w1[0], w1[1], w1[2], b1[3]))


@wp.kernel
def iterate_inertia8(
    pairs: wp.array[wp.vec2i],
    r: wp.array[wp.vec3],
    vm: wp.array[wp.vec4],
    wx: wp.array[wp.vec4],
    inertia8: wp.array[wp.vec4],
):
    """Inertia as two vec4 (two spare lanes), angular_velocity.w unused."""
    t = wp.tid()
    p = pairs[t]
    a0 = load4(vm, p[0])
    b0 = load4(wx, p[0])
    a1 = load4(vm, p[1])
    b1 = load4(wx, p[1])
    h0 = load4(inertia8, 2 * p[0] + 1)
    h1 = load4(inertia8, 2 * p[1] + 1)
    i0 = sym6_from(load4(inertia8, 2 * p[0]), h0[0], h0[1])
    i1 = sym6_from(load4(inertia8, 2 * p[1]), h1[0], h1[1])
    v0 = wp.vec3(a0[0], a0[1], a0[2])
    w0 = wp.vec3(b0[0], b0[1], b0[2])
    v1 = wp.vec3(a1[0], a1[1], a1[2])
    w1 = wp.vec3(b1[0], b1[1], b1[2])
    m0 = a0[3]
    m1 = a1[3]
    arm = r[t]
    rel = v1 + wp.cross(w1, arm) - v0 - wp.cross(w0, arm)
    impulse = -0.5 * rel / (m0 + m1 + 1.0e-6)
    v0 = v0 - m0 * impulse
    w0 = w0 - apply_inertia(i0, wp.cross(arm, impulse))
    v1 = v1 + m1 * impulse
    w1 = w1 + apply_inertia(i1, wp.cross(arm, impulse))
    store4(vm, p[0], wp.vec4(v0[0], v0[1], v0[2], m0))
    store4(wx, p[0], wp.vec4(w0[0], w0[1], w0[2], b0[3]))
    store4(vm, p[1], wp.vec4(v1[0], v1[1], v1[2], m1))
    store4(wx, p[1], wp.vec4(w1[0], w1[1], w1[2], b1[3]))


@wp.kernel
def iterate_inertia_in_w(
    pairs: wp.array[wp.vec2i],
    r: wp.array[wp.vec3],
    va: wp.array[wp.vec4],
    wb: wp.array[wp.vec4],
    inertia4: wp.array[wp.vec4],
    inv_mass: wp.array[float],
):
    """Inertia 4/5 in velocity.w / angular_velocity.w, 0-3 in a vec4, inverse mass in a float array."""
    t = wp.tid()
    p = pairs[t]
    a0 = load4(va, p[0])
    b0 = load4(wb, p[0])
    a1 = load4(va, p[1])
    b1 = load4(wb, p[1])
    i0 = sym6_from(load4(inertia4, p[0]), a0[3], b0[3])
    i1 = sym6_from(load4(inertia4, p[1]), a1[3], b1[3])
    m0 = load1(inv_mass, p[0])
    m1 = load1(inv_mass, p[1])
    v0 = wp.vec3(a0[0], a0[1], a0[2])
    w0 = wp.vec3(b0[0], b0[1], b0[2])
    v1 = wp.vec3(a1[0], a1[1], a1[2])
    w1 = wp.vec3(b1[0], b1[1], b1[2])
    arm = r[t]
    rel = v1 + wp.cross(w1, arm) - v0 - wp.cross(w0, arm)
    impulse = -0.5 * rel / (m0 + m1 + 1.0e-6)
    v0 = v0 - m0 * impulse
    w0 = w0 - apply_inertia(i0, wp.cross(arm, impulse))
    v1 = v1 + m1 * impulse
    w1 = w1 + apply_inertia(i1, wp.cross(arm, impulse))
    store4(va, p[0], wp.vec4(v0[0], v0[1], v0[2], a0[3]))
    store4(wb, p[0], wp.vec4(w0[0], w0[1], w0[2], b0[3]))
    store4(va, p[1], wp.vec4(v1[0], v1[1], v1[2], a1[3]))
    store4(wb, p[1], wp.vec4(w1[0], w1[1], w1[2], b1[3]))


def colored_pairs(rng, bodies, pairs_per_color, colors, reach):
    """Pairs of nearby bodies, each color touching every body at most once."""
    out = []
    for _ in range(colors):
        order = rng.permutation(bodies)
        a = order[: 2 * pairs_per_color : 2]
        b = np.clip(a + rng.integers(1, reach, a.size), 0, bodies - 1)
        # A color may touch each body once: drop pairs that reuse a body.
        uses = np.bincount(np.concatenate([a, b]), minlength=bodies)
        keep = (a != b) & (uses[a] == 1) & (uses[b] == 1)
        out.append(np.stack([a[keep], b[keep]], 1))
    return out


def time_graph(fn, replays=50, trials=7):
    fn()
    with wp.ScopedCapture() as capture:
        for _ in range(replays):
            fn()
    best = float("inf")
    for _ in range(trials):
        start = wp.Event(enable_timing=True)
        end = wp.Event(enable_timing=True)
        wp.record_event(start)
        wp.capture_launch(capture.graph)
        wp.record_event(end)
        wp.synchronize()
        best = min(best, wp.get_event_elapsed_time(start, end) / replays)
    return best


def run_size(rng, bodies, pairs):
    colors = colored_pairs(rng, bodies, pairs // 8, 8, 64)
    pair_arrays = [wp.array(c.astype(np.int32), dtype=wp.vec2i) for c in colors]
    arms = [wp.array(rng.standard_normal((len(c), 3)).astype(np.float32) * 0.1, dtype=wp.vec3) for c in colors]
    v_np = rng.standard_normal((bodies, 3)).astype(np.float32)
    w_np = rng.standard_normal((bodies, 3)).astype(np.float32)
    m_np = rng.uniform(0.1, 1.0, bodies).astype(np.float32)
    i_np = rng.uniform(0.1, 1.0, (bodies, 6)).astype(np.float32)
    v, w = wp.array(v_np, dtype=wp.vec3), wp.array(w_np, dtype=wp.vec3)
    m, inertia = wp.array(m_np), wp.array(i_np, dtype=vec6)
    vm = wp.array(np.concatenate([v_np, m_np[:, None]], 1), dtype=wp.vec4)
    wx = wp.array(np.concatenate([w_np, np.zeros((bodies, 1), np.float32)], 1), dtype=wp.vec4)

    def sweep_current():
        for pa, ar in zip(pair_arrays, arms, strict=True):
            wp.launch(iterate_current, dim=pa.shape[0], inputs=[pa, ar, v, w, m, inertia])

    def sweep_current_raw():
        for pa, ar in zip(pair_arrays, arms, strict=True):
            wp.launch(iterate_current_raw, dim=pa.shape[0], inputs=[pa, ar, v, w, m, inertia])

    def sweep_packed():
        for pa, ar in zip(pair_arrays, arms, strict=True):
            wp.launch(iterate_packed, dim=pa.shape[0], inputs=[pa, ar, vm, wx, inertia])

    t_cur = time_graph(sweep_current)
    inertia4 = wp.array(i_np[:, :4].copy(), dtype=wp.vec4)
    wx_i4 = wp.array(np.concatenate([w_np, i_np[:, 4:5]], 1), dtype=wp.vec4)
    px = wp.array(np.concatenate([rng.standard_normal((bodies, 3)).astype(np.float32), i_np[:, 5:6]], 1), dtype=wp.vec4)
    last = wp.array(i_np[:, 5].copy())
    inertia8 = wp.array(np.concatenate([i_np, np.zeros((bodies, 2), np.float32)], 1).reshape(-1, 4), dtype=wp.vec4)

    def sweep(kernel, *arrays):
        for pa, ar in zip(pair_arrays, arms, strict=True):
            wp.launch(kernel, dim=pa.shape[0], inputs=[pa, ar, *arrays])

    t_raw = time_graph(sweep_current_raw)
    t_pack = time_graph(sweep_packed)
    t_split_pos = time_graph(lambda: sweep(iterate_split_position, vm, wx_i4, inertia4, px))
    t_split_scalar = time_graph(lambda: sweep(iterate_split_scalar, vm, wx_i4, inertia4, last))
    t_inertia8 = time_graph(lambda: sweep(iterate_inertia8, vm, wx, inertia8))
    va = wp.array(np.concatenate([v_np, i_np[:, 4:5]], 1), dtype=wp.vec4)
    wb = wp.array(np.concatenate([w_np, i_np[:, 5:6]], 1), dtype=wp.vec4)
    t_in_w = time_graph(lambda: sweep(iterate_inertia_in_w, va, wb, inertia4, m))
    # Same arithmetic on the same data: results must agree.
    v.assign(v_np)
    w.assign(w_np)
    vm.assign(np.concatenate([v_np, m_np[:, None]], 1))
    wx.assign(np.concatenate([w_np, np.zeros((bodies, 1), np.float32)], 1))
    sweep_current()
    sweep_packed()
    same = np.allclose(v.numpy(), vm.numpy()[:, :3], atol=1e-4) and np.allclose(w.numpy(), wx.numpy()[:, :3], atol=1e-4)
    print(
        f"bodies {bodies:>9,}: current {t_cur:.3f}  raw {t_raw:.3f}  "
        f"A packed {t_pack:.3f}  B split->position.w {t_split_pos:.3f}  "
        f"C inertia 2x vec4 {t_inertia8:.3f}  D split->float array {t_split_scalar:.3f}  "
        f"E inertia in w + mass array {t_in_w:.3f} ms  match {same}"
    )


def main():
    wp.init()
    rng = np.random.default_rng(3)
    for bodies, pairs in ((20_000, 60_000), (200_000, 600_000), (2_000_000, 6_000_000)):
        run_size(rng, bodies, pairs)


if __name__ == "__main__":
    main()
