# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A/B: edge-sampling vs cached-sampling SDF contacts on PhoenX scenes.

Usage:
    uv run python local_studies/sdf_dcd/compare.py SCENE [SCENE ...] [--frames N] [--methods edge cached_sampling]

SCENE is ``scenes:make_box_pyramid`` style (``module:function``); bare names are
looked up in ``local_studies/sdf_dcd/scenes.py`` and then in
``newton.examples.phoenx.tune_phoenx_scenes``.

Timing is CUDA-graph replay of one full frame (collide + step) and of one
collide call alone. Penetration is always measured by an edge-sampling
pipeline (independent of the method under test) at every sample.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import sys
import time

import numpy as np
import warp as wp

import newton
from newton.examples.phoenx.tune_phoenx import _contact_separation, _joint_data, _joint_errors

sys.path.insert(0, os.path.dirname(__file__))


@contextlib.contextmanager
def sdf_method(method: str, config=None):
    """Inject ``sdf_contact_method`` into every CollisionPipeline built inside."""
    original = newton.CollisionPipeline.__init__

    def patched(self, model, **kw):
        kw.setdefault("sdf_contact_method", method)
        if method == "cached_sampling" and config is not None:
            kw.setdefault("sdf_cached_sampling_config", config)
        original(self, model, **kw)

    newton.CollisionPipeline.__init__ = patched
    try:
        yield
    finally:
        newton.CollisionPipeline.__init__ = original


def load_scene(name):
    if ":" in name:
        mod, fn = name.split(":", 1)
        return getattr(importlib.import_module(mod), fn)()
    import scenes  # noqa: PLC0415

    if hasattr(scenes, name):
        return getattr(scenes, name)()
    from newton.examples.phoenx import tune_phoenx_scenes  # noqa: PLC0415

    return getattr(tune_phoenx_scenes, name)()


class Oracle:
    """Edge-sampling pipeline that measures true penetration of a state."""

    def __init__(self, model):
        with sdf_method("edge"):
            self.pipeline = newton.CollisionPipeline(model, rigid_contact_max=200000)
        self.contacts = self.pipeline.contacts()
        self.sep = wp.empty(self.contacts.rigid_contact_max, dtype=wp.float32, device=model.device)
        self.model = model

    def penetration(self, state):
        self.pipeline.collide(state, self.contacts)
        n = int(self.contacts.rigid_contact_count.numpy()[0])
        n = min(n, self.contacts.rigid_contact_max)
        if n == 0:
            return 0.0
        c = self.contacts
        wp.launch(
            _contact_separation,
            dim=n,
            inputs=[
                state.body_q,
                self.model.shape_body,
                c.rigid_contact_shape0,
                c.rigid_contact_shape1,
                c.rigid_contact_point0,
                c.rigid_contact_point1,
                c.rigid_contact_normal,
                c.rigid_contact_margin0,
                c.rigid_contact_margin1,
                self.sep,
            ],
        )
        return max(0.0, -float(self.sep.numpy()[:n].min()))


def time_graph(g, reps):
    for _ in range(3):
        wp.capture_launch(g)
    wp.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        wp.capture_launch(g)
    wp.synchronize()
    return (time.perf_counter() - t) / reps


def run(scene_name, method, frames, sample_stride, config=None, timing_reps=50):
    with sdf_method(method, config):
        scene = load_scene(scene_name)
        model = scene.model
        pipeline = (
            scene.pipeline_factory()
            if scene.pipeline_factory
            else newton.CollisionPipeline(model, contact_matching="sticky")
        )
        opts = dict(scene.solver_options)
        solver = newton.solvers.SolverPhoenX(model, collision_pipeline=pipeline, **opts)
    contacts = pipeline.contacts()
    control = model.control()
    states = [model.state()] if scene.in_place else [model.state(), model.state()]
    if scene.initialize_state:
        scene.initialize_state(states[0])
    oracle = Oracle(model)
    joint_data = _joint_data(model) if model.joint_count else None
    upd_dt = scene.frame_dt / scene.collision_updates
    idx = [0]
    frame_counter = [0]

    def frame():
        for u in range(scene.collision_updates):
            cur = states[idx[0]]
            nxt = cur if scene.in_place else states[1 - idx[0]]
            cur.clear_forces()
            if scene.before_update:
                scene.before_update(cur, control, frame_counter[0] * scene.collision_updates + u, upd_dt)
            pipeline.collide(cur, contacts, dt=upd_dt)
            solver.step(cur, nxt, control, contacts, upd_dt)
            if not scene.in_place:
                idx[0] = 1 - idx[0]

    # Capture one frame; with two states the graph holds an even number of
    # swaps only when collision_updates is even, so capture two frames then.
    two = (not scene.in_place) and scene.collision_updates % 2 == 1
    with wp.ScopedCapture() as cap:
        frame()
        if two:
            frame()
    graph = cap.graph
    per_launch = 2 if two else 1

    metrics = {"penetration_m": 0.0, "joint_m": 0.0, "joint_rad": 0.0}
    samples = {}
    contact_counts = []
    finite = True
    f = 0
    while f < frames:
        wp.capture_launch(graph)
        f += per_launch
        frame_counter[0] = f
        if f % sample_stride < per_launch or f >= frames:
            st = states[idx[0]]
            q = st.body_q.numpy()
            if not np.isfinite(q).all():
                finite = False
                break
            metrics["penetration_m"] = max(metrics["penetration_m"], oracle.penetration(st))
            contact_counts.append(int(contacts.rigid_contact_count.numpy()[0]))
            if joint_data is not None:
                lin, ang, _ = _joint_errors(model, q, joint_data)
                metrics["joint_m"] = max(metrics["joint_m"], lin)
                metrics["joint_rad"] = max(metrics["joint_rad"], ang)
            if scene.extra_metrics:
                for k, v in scene.extra_metrics(st, contacts).items():
                    metrics[k] = max(metrics.get(k, 0.0), float(v))
            if scene.observables:
                for k, v in scene.observables(st).items():
                    samples.setdefault(k, []).append(float(v))

    # Timing (after the trial, from the final state).
    frame_s = time_graph(graph, timing_reps) / per_launch
    with wp.ScopedCapture() as cap:
        pipeline.collide(states[idx[0]], contacts, dt=upd_dt)
    collide_s = time_graph(cap.graph, timing_reps)
    cs = getattr(pipeline.narrow_phase, "sdf_cached_sampling", None)
    out = {
        "scene": scene_name,
        "method": method,
        "finite": finite,
        "frames": f,
        "fps": 1.0 / frame_s,
        "collide_ms": collide_s * 1e3,
        "collide_ms_per_frame": collide_s * 1e3 * scene.collision_updates,
        "contacts_mean": float(np.mean(contact_counts)) if contact_counts else 0.0,
        "contacts_last": contact_counts[-1] if contact_counts else 0,
        "metrics": metrics,
        "observables_last": {k: v[-1] for k, v in samples.items()},
        "observables_tail_mean": {k: float(np.mean(v[len(v) // 2 :])) for k, v in samples.items()},
        "observables_trace": {k: v[:: max(1, len(v) // 8)] for k, v in samples.items()},
    }
    if cs is not None:
        out["cs_pairs"] = int(cs.counts.numpy()[0])
    return out


def fmt(r):
    m = r["metrics"]
    extra = " ".join(
        f"{k}={v * 1000:.2f}mm" if k.endswith("_m") else f"{k}={v:.3g}"
        for k, v in m.items()
        if k not in ("penetration_m", "joint_m", "joint_rad")
    )
    obs = " ".join(f"{k}={v:.4g}" for k, v in r["observables_tail_mean"].items())
    obs += " trace " + " ".join(
        f"{k}:[" + ",".join(f"{x:.1e}" for x in v) + "]" for k, v in r["observables_trace"].items()
    )
    return (
        f"{r['scene']:24} {r['method']:16} fin={r['finite']!s:5} fps={r['fps']:8.1f} "
        f"collide/frame={r['collide_ms_per_frame']:7.3f}ms contacts={r['contacts_mean']:7.1f} "
        f"pen={m['penetration_m'] * 1000:6.2f}mm joint={m['joint_m'] * 1000:6.2f}mm/{math.degrees(m['joint_rad']):5.2f}deg "
        f"{extra} {obs}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--sample-stride", type=int, default=10)
    ap.add_argument("--methods", nargs="+", default=["edge", "cached_sampling"])
    ap.add_argument("--cache-size", type=int, default=None)
    ap.add_argument("--random-samples", type=int, default=None)
    ap.add_argument("--cfg", nargs="*", default=[], help="SDFCachedSampling.Config overrides key=value")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    cfg = None
    if args.cache_size is not None or args.random_samples is not None or args.cfg:
        from newton.geometry import SDFCachedSampling  # noqa: PLC0415

        cfg = SDFCachedSampling.Config()
        if args.cache_size is not None:
            cfg.cache_size = args.cache_size
        if args.random_samples is not None:
            cfg.random_samples = args.random_samples
        for kv in args.cfg:
            k, v = kv.split("=")
            cur = getattr(cfg, k)
            setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else v.lower() in ("1", "true"))
    rows = []
    for s in args.scenes:
        for m in args.methods:
            r = run(s, m, args.frames, args.sample_stride, cfg)
            rows.append(r)
            print(fmt(r), flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
