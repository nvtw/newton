# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# PhoenX Test Visualizer
#
# Live viewer for any scene a test decorates with
# :func:`newton._src.solvers.phoenx.examples.scene_registry.scene`. At
# startup it imports every sibling ``test_*.py`` so all decorators fire,
# then renders the first registered scene. An imgui combo in the
# bottom-left lets you switch scenes at runtime.
#
# The visualizer drives :class:`PhoenXWorld` directly (no contacts --
# registered scenes here are joint-only). Scenes that need a Newton
# :class:`CollisionPipeline` drive their own per-frame pipeline (see
# :mod:`example_tower`, :mod:`example_rabbit_pile`, ...) and are not
# registered with :func:`scene`.
#
# Run::
#
#     python -m newton._src.solvers.phoenx.examples.example_test_visualizer
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.examples.scene_registry import (
    Scene,
    discover_scenes,
    registered_scenes,
)
from newton._src.solvers.phoenx.picking import Picking, register_with_viewer_gl
from newton._src.solvers.phoenx.solver_phoenx import pack_body_xforms_kernel


@wp.kernel
def _gather_xforms_kernel(
    src: wp.array[wp.transform],
    indices: wp.array[wp.int32],
    dst: wp.array[wp.transform],
):
    """Copy ``src[indices[i]]`` into ``dst[i]``.

    Used to fan the flat per-body pose array out to per-render-group
    buffers, since :meth:`viewer.log_shapes` takes a single geometry
    scale per call.
    """
    i = wp.tid()
    dst[i] = src[indices[i]]


class _RenderGroup:
    """One ``log_shapes`` call's worth of state: body indices + pre-allocated
    per-group xform buffer."""

    def __init__(
        self,
        path: str,
        half_extent: tuple[float, float, float],
        indices: wp.array,
        xforms: wp.array,
    ):
        self.path = path
        self.half_extent = half_extent
        self.indices = indices
        self.xforms = xforms


def _build_render_groups(half_extents_np: np.ndarray, device) -> list[_RenderGroup]:
    """Bucket bodies by their non-zero half-extent into render groups.
    Bodies with any non-positive component are skipped."""
    groups: dict[tuple[float, float, float], list[int]] = {}
    for body_idx, he in enumerate(half_extents_np):
        if he[0] <= 0.0 or he[1] <= 0.0 or he[2] <= 0.0:
            continue
        key = (float(he[0]), float(he[1]), float(he[2]))
        groups.setdefault(key, []).append(body_idx)

    out: list[_RenderGroup] = []
    for gid, (key, idxs) in enumerate(groups.items()):
        idx_arr = wp.array(np.asarray(idxs, dtype=np.int32), dtype=wp.int32, device=device)
        xforms = wp.zeros(len(idxs), dtype=wp.transform, device=device)
        out.append(
            _RenderGroup(
                path=f"/world/scene/group_{gid}",
                half_extent=key,
                indices=idx_arr,
                xforms=xforms,
            )
        )
    return out


class _LiveScene:
    """Per-scene state the visualizer holds between frames."""

    def __init__(self, scene_name: str, scene: Scene, viewer, device):
        self.name = scene_name
        self.scene = scene
        self.world = scene.world
        self.device = device
        self.frame_dt = float(scene.frame_dt)
        self.substeps = max(int(scene.substeps), 1)
        self.sim_time = 0.0

        # Push the scene's substep count into the world so a single
        # ``world.step(frame_dt, ...)`` per rendered frame does the
        # right amount of work.
        self.world.substeps = self.substeps

        num_bodies = scene.body_half_extents.shape[0]
        self._xforms = wp.zeros(num_bodies, dtype=wp.transform, device=device)
        self._render_groups = _build_render_groups(scene.body_half_extents, device)

        he_arr = wp.array(
            scene.body_half_extents.astype(np.float32, copy=False),
            dtype=wp.vec3f,
            device=device,
        )
        self.picking = Picking(self.world, he_arr)
        register_with_viewer_gl(viewer, self.picking)

        self._capture()

    def _capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def _simulate(self):
        self.world.step(self.frame_dt, picking=self.picking)

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate()
        self.sim_time += self.frame_dt

    def render(self, viewer):
        wp.launch(
            pack_body_xforms_kernel,
            dim=self._xforms.shape[0],
            inputs=[self.world.bodies, self._xforms],
            device=self.device,
        )
        viewer.begin_frame(self.sim_time)
        for group in self._render_groups:
            wp.launch(
                _gather_xforms_kernel,
                dim=group.xforms.shape[0],
                inputs=[self._xforms, group.indices, group.xforms],
                device=self.device,
            )
            viewer.log_shapes(
                group.path,
                newton.GeoType.BOX,
                group.half_extent,
                group.xforms,
            )
        viewer.end_frame()


class _SceneSelectorUI:
    """Free-floating imgui window with the scene dropdown."""

    def __init__(self, owner: Example):
        self.owner = owner

    def render(self, imgui):
        viewer = self.owner.viewer
        if not getattr(viewer, "ui", None) or not viewer.ui.is_available:
            return

        io = viewer.ui.io
        window_width = 360
        window_height = 200
        imgui.set_next_window_pos(imgui.ImVec2(10, io.display_size[1] - window_height - 10))
        imgui.set_next_window_size(imgui.ImVec2(window_width, window_height))
        flags = imgui.WindowFlags_.no_resize.value

        if imgui.begin("PhoenX Test Scenes", flags=flags):
            names = self.owner.scene_names()
            current = self.owner.current_scene_name() or names[0]
            cur_idx = names.index(current) if current in names else 0

            imgui.text("Scene:")
            changed, new_idx = imgui.combo("##phoenx_scene", cur_idx, names)
            if changed and 0 <= new_idx < len(names):
                self.owner.request_scene(names[new_idx])

            imgui.separator()

            description = self.owner.current_scene_description()
            if description:
                imgui.text_wrapped(description)

        imgui.end()


class Example:
    """Visualizer that swaps live :class:`PhoenXWorld` instances on demand."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        scenes = discover_scenes()
        if not scenes:
            raise RuntimeError(
                "no scenes registered -- decorate a builder with "
                "@newton._src.solvers.phoenx.examples.scene_registry.scene"
            )

        self._scene_names = [s.name for s in scenes]
        self._pending_scene: str | None = None
        self._live: _LiveScene | None = None
        self._load_scene(self._scene_names[0])

        if hasattr(self.viewer, "register_ui_callback"):
            self._selector_ui = _SceneSelectorUI(self)
            self.viewer.register_ui_callback(self._selector_ui.render, "free")

    # ------------------------------------------------------------------
    # Selector UI hooks
    # ------------------------------------------------------------------

    def scene_names(self) -> list[str]:
        return list(self._scene_names)

    def current_scene_name(self) -> str | None:
        return self._live.name if self._live is not None else None

    def current_scene_description(self) -> str:
        if self._live is None:
            return ""
        return self._live.scene.description

    def request_scene(self, name: str) -> None:
        """Queue ``name`` for loading at the top of the next step. Decoupled
        from the UI render thread so the previous scene's CUDA graph can be
        dropped safely from the main sim loop."""
        if name not in self._scene_names:
            return
        self._pending_scene = name

    # ------------------------------------------------------------------
    # Scene loading
    # ------------------------------------------------------------------

    def _load_scene(self, name: str) -> None:
        rec = next((r for r in registered_scenes() if r.name == name), None)
        if rec is None:
            raise KeyError(f"scene {name!r} is not registered")
        # Drop old scene first so we don't briefly hold two finalised
        # worlds (each pinning GPU memory).
        self._live = None
        scene_obj = rec.builder(self.device)
        self._live = _LiveScene(name, scene_obj, self.viewer, self.device)

    # ------------------------------------------------------------------
    # newton.examples.run() hooks
    # ------------------------------------------------------------------

    def step(self):
        if self._pending_scene is not None and (self._live is None or self._pending_scene != self._live.name):
            target = self._pending_scene
            self._pending_scene = None
            self._load_scene(target)

        assert self._live is not None
        self._live.step()

    def render(self):
        assert self._live is not None
        self._live.render(self.viewer)


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
