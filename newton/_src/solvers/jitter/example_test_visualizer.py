# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Test Visualizer
#
# Live viewer for any scene a test (or other module in this package)
# decorates with :func:`newton._src.solvers.jitter.scene_registry.scene`.
# At startup it imports every sibling ``test_*.py`` so all decorators
# fire, then renders the first registered scene; an imgui combo in the
# left UI panel lets you switch to any other registered scene at
# runtime. The previous scene's :class:`World` (and its CUDA graph) are
# dropped on switch.
#
# Run: ``python -m newton._src.solvers.jitter.example_test_visualizer``
#
# Each registered scene supplies its own ``substeps`` and ``frame_dt``
# (see :class:`Scene`); we honour them so e.g. a stiff hinge demo can
# ask for 16 substeps while a loose chain stays at 4.
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.jitter.picking import JitterPicking, register_with_viewer_gl
from newton._src.solvers.jitter.scene_registry import (
    Scene,
    discover_scenes,
    registered_scenes,
)
from newton._src.solvers.jitter.solver_jitter import pack_body_xforms_kernel


@wp.kernel
def _gather_xforms_kernel(
    src: wp.array[wp.transform],
    indices: wp.array[wp.int32],
    dst: wp.array[wp.transform],
):
    """Copy ``src[indices[i]]`` into ``dst[i]`` for ``i`` in
    ``[0, len(indices))``.

    Used by the visualizer to fan a single packed-pose array (one
    transform per body) out to per-render-group buffers, since
    :meth:`viewer.log_shapes` takes one shared geometry scale per call
    -- so bodies with different half-extents have to be split across
    multiple ``log_shapes`` invocations, each fed by its own gathered
    transform array.
    """
    i = wp.tid()
    dst[i] = src[indices[i]]


class _RenderGroup:
    """One ``log_shapes`` call's worth of state.

    A render group bundles the indices of all bodies that share an
    identical (non-zero) half-extent triple, plus pre-allocated device
    buffers so the per-frame fast path is just two ``wp.launch``
    invocations (pack-all + gather-this-group) followed by one viewer
    call.

    Attributes:
        path: Hierarchical name passed to ``viewer.log_shapes`` -- the
            visualizer makes it unique per group so the GL viewer
            doesn't collide instances across groups.
        half_extent: Box half-extents in body-local frame; passed
            verbatim as the ``geo_scale`` argument.
        indices: ``wp.array[wp.int32]`` of body indices into the full
            ``num_bodies`` xforms buffer.
        xforms: Output buffer, sized ``len(indices)``; rewritten every
            frame by :func:`_gather_xforms_kernel`.
    """

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


def _build_render_groups(
    half_extents_np: np.ndarray, device
) -> list[_RenderGroup]:
    """Bucket bodies by their (non-zero) half-extent into render groups.

    Bodies whose half-extent has any non-positive component are skipped
    entirely (matches the picking convention for the static world
    anchor) -- they neither render nor get a transform-gather.

    Args:
        half_extents_np: ``(num_bodies, 3)`` ``np.float32`` half-extents.
        device: Warp device on which to allocate the per-group buffers.

    Returns:
        One :class:`_RenderGroup` per unique non-zero half-extent
        triple, in the order they were first seen (so render paths are
        stable across runs of the same scene).
    """
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
    """All per-scene state the visualizer holds onto between frames.

    Owning this in its own class makes scene swapping a single
    ``self._live = _LiveScene(...)`` assignment from the UI handler;
    the previous instance (and its World, picking arrays, CUDA graph,
    etc.) becomes garbage and is collected.
    """

    def __init__(self, scene_name: str, scene: Scene, viewer, device):
        self.name = scene_name
        self.scene = scene
        self.world = scene.world
        self.device = device
        self.frame_dt = float(scene.frame_dt)
        self.substeps = max(int(scene.substeps), 1)
        self.sim_time = 0.0

        # Scene builders default to ``substeps=1`` when calling
        # ``WorldBuilder.finalize`` (substepping used to live in the
        # visualizer). Now that ``World.step`` owns the substep loop
        # internally, push the scene's desired substep count into the
        # world so one ``world.step(frame_dt, ...)`` call per frame
        # does the right amount of work. Cheap -- this is a plain int
        # attribute on :class:`World`.
        self.world.substeps = self.substeps

        num_bodies = scene.body_half_extents.shape[0]
        # One transform per body (filled by pack_body_xforms_kernel
        # every frame). Per-render-group gathers slice from this.
        self._xforms = wp.zeros(num_bodies, dtype=wp.transform, device=device)
        self._render_groups = _build_render_groups(scene.body_half_extents, device)

        # Picking uses the same half-extents array (zero entries =>
        # non-pickable, e.g. world anchor at body 0).
        he_arr = wp.array(
            scene.body_half_extents.astype(np.float32, copy=False),
            dtype=wp.vec3f,
            device=device,
        )
        self.picking = JitterPicking(self.world, he_arr)
        register_with_viewer_gl(viewer, self.picking)

        # CUDA graph capture of the substep loop. Same pattern as the
        # other jitter examples; CPU runs immediate-mode.
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
        # 1) Pack every body's pose into the shared transforms buffer.
        wp.launch(
            pack_body_xforms_kernel,
            dim=self._xforms.shape[0],
            inputs=[self.world.bodies, self._xforms],
            device=self.device,
        )

        viewer.begin_frame(self.sim_time)
        # 2) For every (half-extent) bucket, gather just those bodies'
        #    transforms and emit one log_shapes call.
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


class JitterSceneSelectorUI:
    """Free-floating imgui window with the jitter-scene dropdown.

    Modelled after ``ReplayUI`` in :mod:`example_replay_viewer`: an
    independent UI extension registered as a ``"free"`` callback on
    :class:`ViewerGL`, which renders a dedicated window
    (``"Jitter Test Scenes"``) instead of piggy-backing on the
    built-in left-side ``Example Options`` panel. Keeps the dropdown
    visually separate and means we can stop providing an
    ``Example.gui`` (which :func:`newton.examples.run` would otherwise
    auto-mount in the side panel as a duplicate).

    The selector talks back to its owning :class:`Example` through
    :meth:`Example.request_scene`, which records the requested name
    into a "pending" slot processed at the start of the next
    :meth:`Example.step` -- swapping ``World`` from inside an imgui
    callback would tear down CUDA buffers mid-frame.
    """

    def __init__(self, owner: Example):
        self.owner = owner

    def render(self, imgui):
        viewer = self.owner.viewer
        # ViewerGL routes through ``viewer.ui``; bail out if the GL UI
        # isn't actually live (headless / file viewer, ``--test`` runs
        # without a real window, etc.).
        if not getattr(viewer, "ui", None) or not viewer.ui.is_available:
            return

        io = viewer.ui.io
        # Anchor the window to the bottom-left so it doesn't fight
        # the built-in left panel (top-left) or the stats overlay
        # (top-right). Same constants as ReplayUI's choice of corner.
        window_width = 360
        window_height = 200
        imgui.set_next_window_pos(
            imgui.ImVec2(10, io.display_size[1] - window_height - 10)
        )
        imgui.set_next_window_size(imgui.ImVec2(window_width, window_height))
        flags = imgui.WindowFlags_.no_resize.value

        if imgui.begin("Jitter Test Scenes", flags=flags):
            names = self.owner.scene_names()
            current = self.owner.current_scene_name() or names[0]
            cur_idx = names.index(current) if current in names else 0

            imgui.text("Scene:")
            changed, new_idx = imgui.combo("##jitter_scene", cur_idx, names)
            if changed and 0 <= new_idx < len(names):
                self.owner.request_scene(names[new_idx])

            imgui.separator()

            description = self.owner.current_scene_description()
            if description:
                imgui.text_wrapped(description)

        imgui.end()


class Example:
    """Visualizer ``Example`` that swaps live :class:`World` instances
    on demand.

    Implements only the bare ``Example`` contract from
    :mod:`newton.examples` (``__init__`` / ``step`` / ``render`` /
    ``test_final``); the scene-picker UI is delegated to a separate
    :class:`JitterSceneSelectorUI` registered with the viewer as a
    ``"free"`` callback so it shows up as an independent imgui
    window rather than another entry under the side panel's
    ``Example Options`` header. We deliberately do *not* expose a
    ``gui`` method -- :func:`newton.examples.run` would otherwise
    auto-mount it into the side panel as a second copy of the
    dropdown.
    """

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        # Discover decorators by importing every sibling test module.
        # Idempotent: subsequent calls are no-ops.
        scenes = discover_scenes()
        if not scenes:
            raise RuntimeError(
                "no scenes registered -- decorate a builder with "
                "@newton._src.solvers.jitter.scene_registry.scene"
            )

        self._scene_names = [s.name for s in scenes]
        self._pending_scene: str | None = None
        self._live: _LiveScene | None = None
        self._load_scene(self._scene_names[0])

        # Register the scene-picker as its own free-floating window.
        # ``register_ui_callback`` is ViewerGL-only; the file/null
        # viewers don't have it, so guard with hasattr to keep the
        # example runnable under ``--viewer null`` / ``--test``.
        if hasattr(self.viewer, "register_ui_callback"):
            self._selector_ui = JitterSceneSelectorUI(self)
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
        """Queue ``name`` for loading at the top of the next step.

        Decoupled from the UI render thread: the actual ``World``
        rebuild has to run from the main sim loop so the previous
        :class:`_LiveScene`'s CUDA graph can be safely dropped.
        """
        if name not in self._scene_names:
            return
        self._pending_scene = name

    # ------------------------------------------------------------------
    # Scene loading
    # ------------------------------------------------------------------

    def _load_scene(self, name: str) -> None:
        """Build (or rebuild) the scene named ``name`` and make it live.

        Drops the previous :class:`_LiveScene` (and its CUDA graph) by
        overwriting ``self._live`` first; Python's refcount / CUDA
        runtime handle the actual teardown. The new scene's builder is
        looked up fresh from the registry every time so re-decorating
        a builder during an interactive session takes effect on the
        next switch.
        """
        rec = next((r for r in registered_scenes() if r.name == name), None)
        if rec is None:
            raise KeyError(f"scene {name!r} is not registered")

        # Drop the old scene first so we don't briefly hold *two*
        # finalized worlds (each can pin a non-trivial chunk of GPU
        # memory) and so the old picking callbacks lose their
        # JitterPicking ref.
        self._live = None
        scene_obj = rec.builder(self.device)
        self._live = _LiveScene(name, scene_obj, self.viewer, self.device)

    # ------------------------------------------------------------------
    # newton.examples.run() hooks
    # ------------------------------------------------------------------

    def step(self):
        # Handle deferred scene switch from the UI thread.
        if self._pending_scene is not None and (
            self._live is None or self._pending_scene != self._live.name
        ):
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
