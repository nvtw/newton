# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import math
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import warp as wp

from ..core.types import override
from ..geometry.types import GeoType
from .camera import Camera
from .viewer import _DEFAULT_LAYER_ID, Layer, ViewerBase
from .viewer_gui import ViewerGui

try:
    from warp_optix.pathtracing import PathTracingViewerBackend as _PathTracingViewerBackend
except ImportError as error:
    _WARP_OPTIX_IMPORT_ERROR: ImportError | None = error

    class _PathTracingViewerBackend:
        """Keep :mod:`newton.viewer` importable without the optional backend."""

        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise ImportError(
                "ViewerOptix requires the warp_optix package from otk-pyoptix. "
                "Install otk-pyoptix and its path-tracing extras before creating the viewer."
            ) from _WARP_OPTIX_IMPORT_ERROR

        def end_frame(self):
            return

        def log_mesh(self, *args, **kwargs):
            del args, kwargs

        def log_instances(self, *args, **kwargs):
            del args, kwargs

        def log_lines(self, *args, **kwargs):
            del args, kwargs

        def log_points(self, *args, **kwargs):
            del args, kwargs

        def log_array(self, *args, **kwargs):
            del args, kwargs

        def log_scalar(self, *args, **kwargs):
            del args, kwargs

        def apply_forces(self, *args, **kwargs):
            del args, kwargs

        def close(self):
            return

else:
    _WARP_OPTIX_IMPORT_ERROR = None


@wp.kernel
def _apply_optix_color_palette(
    colors: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    defaults: wp.array[wp.vec3],
    eligible: wp.array[wp.int32],
    palette: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    index = wp.tid()
    color = colors[index]
    default = defaults[index]
    if (
        eligible[index] != 0
        and wp.abs(color[0] - default[0]) <= 1.0e-6
        and wp.abs(color[1] - default[1]) <= 1.0e-6
        and wp.abs(color[2] - default[2]) <= 1.0e-6
    ):
        color = palette[indices[index] % len(palette)]
    output[index] = color


@wp.kernel
def _apply_optix_plane_checker_material(
    source: wp.array[wp.vec4],
    u_subdiv: float,
    v_subdiv: float,
    output: wp.array[wp.vec4],
):
    index = wp.tid()
    source_index = 0 if len(source) == 1 else index
    material = source[source_index]
    output[index] = wp.vec4(material[0], material[1], u_subdiv, v_subdiv)


class _OptixOverlayRenderer:
    """Rendering controls consumed by the shared Newton GUI."""

    def __init__(self, window):
        self.window = window
        self.line_width = 1.5
        self.arrow_scale = 1.0
        self.arrow_length_scale = 1.0
        self.joint_scale = 1.0


class ViewerOptix(_PathTracingViewerBackend, ViewerBase):
    """Interactive OptiX path-tracing viewer with DLSS Ray Reconstruction.

    This viewer adapts :class:`warp_optix.pathtracing.PathTracingViewerBackend`
    to Newton's common viewer interface. It supports Newton model/state logging,
    camera navigation, body picking, an ImGui overlay, and headless frame
    extraction. Rendering and DLSS are provided by the separately installed
    ``otk-pyoptix`` project.
    """

    _DEFAULT_COLOR_PALETTE = (
        (0.86, 0.06, 0.02),  # red
        (0.92, 0.22, 0.01),  # vermilion
        (0.94, 0.42, 0.01),  # orange
        (0.88, 0.62, 0.01),  # amber
        (0.78, 0.72, 0.01),  # yellow
        (0.46, 0.72, 0.01),  # chartreuse
        (0.03, 0.60, 0.12),  # green
        (0.00, 0.58, 0.38),  # turquoise
        (0.00, 0.62, 0.68),  # cyan
        (0.00, 0.53, 0.88),  # azure
        (0.06, 0.30, 0.88),  # blue
        (0.22, 0.12, 0.78),  # indigo
        (0.42, 0.10, 0.76),  # violet
        (0.58, 0.08, 0.68),  # purple
    )

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        vsync: bool = False,
        headless: bool = False,
        paused: bool = False,
        fps: int = 0,
        num_frames: int | None = None,
        enable_dlss_rr: bool = True,
        dlss_quality: str = "performance",
        samples_per_frame: int = 1,
        max_bounces: int = 3,
        direct_light_samples: int = 1,
        enable_imgui: bool = True,
        max_instances: int = 16384,
        ground_color: tuple[float, float, float] = (0.7, 0.7, 0.7),
        ground_roughness: float = 0.8,
        ground_checker_size: float | None = 1.0,
        default_roughness: float = 0.42,
        default_ior: float = 1.46,
        default_specular: float = 0.75,
        default_clearcoat: float = 0.03,
        default_clearcoat_roughness: float = 0.4,
        default_color_palette: Sequence[Sequence[float]] | None = None,
        time_of_day: float = 12.0,
        sky_intensity: float = 1.0,
        grayscale_sky: bool = False,
        exposure: float = 0.68,
        contrast: float = 1.08,
        saturation: float = 1.1,
        **kwargs: Any,
    ):
        """Initialize the OptiX path-tracing viewer.

        Args:
            width: Render width in pixels.
            height: Render height in pixels.
            vsync: Enable vertical synchronization for the presentation window.
            headless: Render without opening a window.
            paused: Start with simulation stepping paused.
            fps: Standalone presentation frame rate. Use ``0`` for uncapped
                rendering. Recording has its own frame-rate setting.
            num_frames: Maximum number of rendered frames, or ``None`` to run
                until the window closes.
            enable_dlss_rr: Enable DLSS Ray Reconstruction when available.
            dlss_quality: DLSS input-resolution/quality mode. Supported values
                are ``"performance"``, ``"balanced"``, ``"quality"``,
                ``"ultra_performance"``, and ``"native"``.
            samples_per_frame: Path-traced samples per frame when DLSS is disabled.
                DLSS Ray Reconstruction always consumes one sample per frame.
            max_bounces: Maximum number of path bounces. Lower values render faster
                but lose some indirect illumination and reflection depth.
            direct_light_samples: Direct-light samples evaluated at each surface hit.
            enable_imgui: Enable the interactive ImGui overlay.
            max_instances: Maximum number of OptiX scene instances.
            ground_color: Display-sRGB color used for plane geometry.
            ground_roughness: Roughness used for plane geometry.
            ground_checker_size: Checker size [m] used for plane geometry.
                ``None`` disables the checker overlay.
            default_roughness: Roughness used for primitive geometry without
                authored material properties.
            default_ior: Index of refraction used for un-authored dielectrics.
            default_specular: Dielectric specular strength in the range [0, 1].
            default_clearcoat: Secondary clearcoat lobe strength in the range [0, 1].
            default_clearcoat_roughness: Secondary clearcoat lobe roughness in
                the range (0, 1].
            default_color_palette: Display-sRGB colors used in place of
                Newton's automatically assigned shape colors. Explicit shape
                and authored mesh colors are preserved. ``None`` uses the
                saturated OptiX palette.
            time_of_day: Procedural-sky time in hours, in the range [0, 24].
            sky_intensity: Procedural-sky illumination multiplier.
            grayscale_sky: Render the physical sky in grayscale while
                preserving its maximum RGB component as brightness.
            exposure: Linear display exposure multiplier.
            contrast: Display contrast multiplier.
            saturation: Display saturation multiplier.
            **kwargs: Additional arguments forwarded to
                ``PathTracingViewerBackend``.
        """
        if _WARP_OPTIX_IMPORT_ERROR is not None:
            raise ImportError(
                "ViewerOptix requires the warp_optix package from otk-pyoptix. "
                "Install otk-pyoptix and its path-tracing extras before creating the viewer."
            ) from _WARP_OPTIX_IMPORT_ERROR

        self.gui: ViewerGui | None = None
        self.show_ui = True
        self._vsync = bool(vsync)
        self._camera_dirty = True
        self._step_requested = False
        self._reset_callback: Callable[[], None] | None = None
        self._ground_color = tuple(float(channel) for channel in ground_color)
        if len(self._ground_color) != 3 or not all(0.0 <= channel <= 1.0 for channel in self._ground_color):
            raise ValueError("ground_color must contain three values in the range [0, 1]")
        self._ground_roughness = float(ground_roughness)
        self._ground_checker_size = None if ground_checker_size is None else float(ground_checker_size)
        if self._ground_checker_size is not None and self._ground_checker_size <= 0.0:
            raise ValueError("ground_checker_size must be positive or None")
        self._default_roughness = float(default_roughness)
        for name, value in (
            ("ground_roughness", self._ground_roughness),
            ("default_roughness", self._default_roughness),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in the range [0, 1]")

        self._optix_ground_meshes: set[str] = set()
        self._optix_ground_subdivisions: dict[str, tuple[float, float]] = {}
        self._ground_color_arrays: dict[int, wp.array[wp.vec3]] = {}
        self._manual_ground_material_arrays: dict[str, wp.array[wp.vec4]] = {}
        self._optix_default_material_meshes: set[str] = set()
        self._optix_model_shape_batches: dict[str, ViewerBase.ShapeInstances] = {}
        self._optix_palette_metadata: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._optix_palette_device_metadata: dict[str, tuple[wp.array, wp.array, wp.array]] = {}
        self._optix_palette_color_arrays: dict[str, wp.array[wp.vec3]] = {}
        self._optix_palette_array: wp.array[wp.vec3] | None = None
        self._default_color_palette = self._validate_color_palette(
            self._DEFAULT_COLOR_PALETTE if default_color_palette is None else default_color_palette
        )
        self._time_of_day = self._validate_time_of_day(time_of_day)
        self._sky_intensity = self._validate_sky_intensity(sky_intensity)
        self._grayscale_sky = bool(grayscale_sky)
        self.lines = {}
        self._material_arrays: dict[tuple[int, float, float, float], wp.array[wp.vec4]] = {}
        self.arrows = {}
        self.renderer: _OptixOverlayRenderer | None = None
        self._overlay_line_shader = None
        self._overlay_arrow_shader = None
        self._overlay_depth_texture = None
        self._overlay_depth_pbo = None
        self._overlay_depth_cuda = None
        self._overlay_depth_size = (0, 0)
        super().__init__(
            width=width,
            height=height,
            title="Newton OptiX Viewer",
            fps=fps,
            headless=headless,
            paused=paused,
            render_when_paused=True,
            num_frames=num_frames,
            enable_dlss_rr=enable_dlss_rr,
            dlss_quality=dlss_quality,
            samples_per_frame=samples_per_frame,
            max_bounces=max_bounces,
            direct_light_samples=direct_light_samples,
            enable_imgui=False,
            vsync=vsync,
            max_instances=max_instances,
            **kwargs,
            default_ior=default_ior,
            default_specular=default_specular,
            default_clearcoat=default_clearcoat,
            default_clearcoat_roughness=default_clearcoat_roughness,
        )
        camera_width, camera_height = self.width, self.height
        self.exposure = exposure
        self.tonemap_contrast = contrast
        self.tonemap_saturation = saturation
        self._apply_time_of_day(reset_history=False)
        if self._presenter is not None:
            camera_width, camera_height = self._presenter.window.get_framebuffer_size()
        self.camera = Camera(width=camera_width, height=camera_height, up_axis="Z")
        if self._presenter is not None:
            self._init_gl_overlay()
            self._presenter._on_draw_overlay = self._draw_newton_overlay
        if enable_imgui and self._presenter is not None:
            self.gui = ViewerGui(self, self._presenter.window)
            self.gui.register_ui_callback(self._ui_populate_rendering_panel, position="rendering")

    @property
    def picking(self):
        """Return the Newton picking helper, if picking is available."""
        return getattr(self, "_picking", None)

    @property
    def paused(self) -> bool:
        """Return whether simulation stepping is paused."""
        return bool(getattr(self, "_paused", False))

    @paused.setter
    def paused(self, value: bool) -> None:
        self._paused = bool(value)

    @property
    def ui(self):
        """Return the shared UI object, matching :class:`ViewerGL`."""
        return self.gui.ui if self.gui is not None else None

    @property
    def vsync(self) -> bool:
        """Return whether presentation waits for vertical synchronization."""
        return self._vsync

    @vsync.setter
    def vsync(self, enabled: bool) -> None:
        self._vsync = bool(enabled)
        if self._presenter is not None:
            self._presenter.window.set_vsync(self._vsync)

    @property
    def exposure(self) -> float:
        """Return the linear display exposure multiplier."""
        return self.tonemap_exposure

    @exposure.setter
    def exposure(self, value: float) -> None:
        """Set the nonnegative linear display exposure multiplier."""
        self.tonemap_exposure = value

    def _init_gl_overlay(self) -> None:
        """Initialize shared Newton GL shaders in the OptiX window context."""
        from .gl.opengl import RendererGL  # noqa: PLC0415
        from .gl.shaders import ShaderArrow, ShaderLine  # noqa: PLC0415

        self._presenter.window.switch_to()
        RendererGL.initialize_gl()
        self._overlay_line_shader = ShaderLine(RendererGL.gl)
        self._overlay_arrow_shader = ShaderArrow(RendererGL.gl)
        self.renderer = _OptixOverlayRenderer(self._presenter.window)

    @override
    def _arrow_scale(self) -> float:
        """Return the shared UI's contact-arrow length multiplier."""
        if self.renderer is None:
            return 1.0
        return self.renderer.arrow_length_scale

    @override
    def _joint_scale(self) -> float:
        """Return the shared UI's joint-axis length multiplier."""
        if self.renderer is None:
            return 1.0
        return self.renderer.joint_scale

    @property
    @override
    def supports_simulation_render_overlap(self) -> bool:
        """Whether this viewer can overlap a CUDA step with OptiX rendering."""
        return False

    def should_step(self) -> bool:
        """Return whether the simulation should advance by one step."""
        if not self.is_paused():
            self._step_requested = False
            return True
        if self._step_requested:
            self._step_requested = False
            return True
        return False

    def set_reset_callback(self, callback: Callable[[], None] | None) -> None:
        """Register the callback invoked by the overlay's Reset button.

        Args:
            callback: Callback to invoke, or ``None`` to remove it.
        """
        if callback is None:
            self._reset_callback = None
            return

        def reset_and_discard_history() -> None:
            try:
                callback()
            finally:
                self.reset_temporal_history()

        self._reset_callback = reset_and_discard_history

    @override
    def register_ui_callback(
        self,
        callback: Callable[[Any], None],
        position: Literal["side", "stats", "free", "panel", "rendering"] = "side",
    ) -> None:
        """Register an ImGui callback using the same positions as ViewerGL.

        Args:
            callback: Function called during UI rendering.
            position: Logical UI location used for callback lifetime and
                compatibility with :class:`ViewerGL`.
        """
        valid_positions = {"side", "stats", "free", "panel", "rendering"}
        if not callable(callback):
            raise TypeError("callback must be callable")
        if position not in valid_positions:
            raise ValueError(f"Invalid position {position!r}. Must be one of: {sorted(valid_positions)}")
        if self.gui is not None:
            self.gui.register_ui_callback(callback, position=position)

    def _ui_populate_rendering_panel(self, imgui) -> None:
        """Render OptiX-specific controls inside the shared rendering panel."""
        imgui.text("DLSS RR: active" if self._api.dlss_enabled else "DLSS RR: inactive")
        quality_modes = list(self._api.viewer.DLSS_QUALITY_MODES)
        quality_index = quality_modes.index(self.dlss_quality)
        changed, quality_index = imgui.combo("DLSS Quality", quality_index, quality_modes)
        if changed:
            self.dlss_quality = quality_modes[quality_index]
        changed, max_bounces = imgui.slider_int("Max Bounces", self.max_bounces, 1, self._api.max_compiled_bounces)
        if changed:
            self.set_ray_budget(max_bounces=max_bounces)
        changed, direct_light_samples = imgui.slider_int("Direct Light Samples", self.direct_light_samples, 1, 4)
        if changed:
            self.set_ray_budget(direct_light_samples=direct_light_samples)
        if not self._api.dlss_enabled:
            changed, samples_per_frame = imgui.slider_int("Samples Per Frame", self.samples_per_frame, 1, 8)
            if changed:
                self.set_ray_budget(samples_per_frame=samples_per_frame)
        changed, grayscale_sky = imgui.checkbox("Grayscale Sky", self.grayscale_sky)
        if changed:
            self.grayscale_sky = grayscale_sky
        changed, sky_intensity = imgui.slider_float("Sky Intensity", self.sky_intensity, 0.0, 3.0)
        if changed:
            self.sky_intensity = sky_intensity
        changed, time_of_day = imgui.slider_float("Time of Day", self.time_of_day, 0.0, 24.0)
        if changed:
            self.time_of_day = time_of_day
        mode_names = [
            "Final",
            "Radiance",
            "Depth",
            "Motion",
            "Normal",
            "Roughness",
            "Diffuse",
            "Specular",
            "Specular Hit Distance",
        ]
        changed, mode = imgui.combo("Debug Buffer", self._debug_buffer_mode, mode_names)
        if changed:
            self.set_debug_buffer_mode(mode)
        changed, picking = imgui.checkbox("Enable Picking", self.picking_enabled)
        if changed:
            self.picking_enabled = picking
            if not picking and self._picking is not None:
                self._picking.release()

    def show_loading_splash(self, text: str | None = None) -> None:
        """Display the shared Newton loading splash.

        Args:
            text: Optional label shown under the loading animation.
        """
        if self.gui is not None:
            self.gui.show_loading_splash(text)

    def hide_loading_splash(self) -> None:
        """Remove the shared Newton loading splash."""
        if self.gui is not None:
            self.gui.hide_loading_splash()

    def _draw_newton_gui(self) -> None:
        if self.gui is not None:
            self.gui.render_frame(update_fps=True)

    def _draw_newton_overlay(self) -> None:
        self._draw_gl_debug_overlay()
        self._draw_newton_gui()

    def _destroy_overlay_depth_resources(self) -> None:
        """Release the CUDA-GL depth bridge while its GL context is current."""
        self._overlay_depth_cuda = None
        if self._presenter is None:
            self._overlay_depth_texture = None
            self._overlay_depth_pbo = None
            self._overlay_depth_size = (0, 0)
            return

        gl = self._presenter.gl
        self._presenter.window.switch_to()
        if self._overlay_depth_texture is not None:
            gl.glDeleteTextures(1, self._overlay_depth_texture)
        if self._overlay_depth_pbo is not None:
            gl.glDeleteBuffers(1, self._overlay_depth_pbo)
        self._overlay_depth_texture = None
        self._overlay_depth_pbo = None
        self._overlay_depth_size = (0, 0)

    def _ensure_overlay_depth_resources(self, width: int, height: int) -> None:
        if self._overlay_depth_size == (width, height):
            return
        self._destroy_overlay_depth_resources()

        gl = self._presenter.gl
        texture = gl.GLuint()
        gl.glGenTextures(1, texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_R32F,
            width,
            height,
            0,
            gl.GL_RED,
            gl.GL_FLOAT,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        pbo = gl.GLuint()
        gl.glGenBuffers(1, pbo)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo)
        gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, width * height * 4, None, gl.GL_STREAM_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

        self._overlay_depth_texture = texture
        self._overlay_depth_pbo = pbo
        self._overlay_depth_cuda = wp.RegisteredGLBuffer(int(pbo.value), device=self.device)
        self._overlay_depth_size = (width, height)

    def _upload_overlay_depth(self) -> bool:
        depth = self._api.linear_depth_output
        if depth is None:
            return False
        width, height = self._api.render_resolution
        self._ensure_overlay_depth_resources(width, height)

        mapped = self._overlay_depth_cuda.map(dtype=wp.float32, shape=(height, width))
        wp.copy(mapped, depth)
        self._overlay_depth_cuda.unmap()

        gl = self._presenter.gl
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self._overlay_depth_pbo)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._overlay_depth_texture)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            width,
            height,
            gl.GL_RED,
            gl.GL_FLOAT,
            ctypes.c_void_p(0),
        )
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        return True

    def _draw_gl_debug_overlay(self) -> None:
        line_batches = [batch for batch in self.lines.values() if not batch.hidden and batch.num_lines > 0]
        arrow_batches = [batch for batch in self.arrows.values() if not batch.hidden and batch.num_lines > 0]
        if not line_batches and not arrow_batches:
            return
        if not self._upload_overlay_depth():
            return

        gl = self._presenter.gl
        framebuffer_width, framebuffer_height = self._presenter.window.get_framebuffer_size()
        inv_aspect = float(framebuffer_height) / float(max(framebuffer_width, 1))
        identity = np.eye(4, dtype=np.float32)
        view = self.camera.get_view_matrix()
        projection = self.camera.get_projection_matrix()
        texture_unit = 3

        gl.glViewport(0, 0, framebuffer_width, framebuffer_height)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._overlay_depth_texture)

        if line_batches:
            clip_width = max(0.0, self.renderer.line_width) * 2.0 / max(framebuffer_height, 1)
            with self._overlay_line_shader:
                self._overlay_line_shader.update_frame(
                    view,
                    projection,
                    inv_aspect,
                    line_width=clip_width,
                    alpha=1.0,
                    scene_depth_texture_unit=texture_unit,
                    viewport_size=(framebuffer_width, framebuffer_height),
                    camera_near=self.camera.near,
                    camera_far=self.camera.far,
                )
                self._overlay_line_shader.set_world(identity)
                for batch in line_batches:
                    batch.render()

        if arrow_batches:
            scale = max(0.0, self.renderer.arrow_scale)
            with self._overlay_arrow_shader:
                self._overlay_arrow_shader.update_frame(
                    view,
                    projection,
                    inv_aspect,
                    line_width=4.0 * scale / max(framebuffer_height, 1),
                    arrow_size=16.0 * scale / max(framebuffer_height, 1),
                    alpha=1.0,
                    scene_depth_texture_unit=texture_unit,
                    viewport_size=(framebuffer_width, framebuffer_height),
                    camera_near=self.camera.near,
                    camera_far=self.camera.far,
                )
                self._overlay_arrow_shader.set_world(identity)
                for batch in arrow_batches:
                    batch.render()

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glDisable(gl.GL_BLEND)

    @override
    def clear_model(self) -> None:
        """Clear the active Newton model and its OptiX scene resources."""
        owns = self._is_layer_owned_path
        for batches in (getattr(self, "lines", {}), getattr(self, "arrows", {})):
            for name in list(batches):
                if owns(name):
                    batches.pop(name).destroy()
        if hasattr(self, "_optix_ground_meshes"):
            self._optix_ground_meshes.clear()
        if hasattr(self, "_optix_ground_subdivisions"):
            self._optix_ground_subdivisions.clear()
            self._optix_default_material_meshes.clear()
            self._optix_model_shape_batches.clear()
            self._optix_palette_metadata.clear()
            self._optix_palette_device_metadata.clear()
            self._optix_palette_color_arrays.clear()
            self._optix_palette_array = None
            self._ground_color_arrays.clear()
            for name in list(self._manual_ground_material_arrays):
                if owns(name):
                    del self._manual_ground_material_arrays[name]
            self._material_arrays.clear()
        if self.gui is not None:
            self.gui.clear_example_callbacks()
        if hasattr(self, "_mesh_ids") and (self._mesh_ids or self._batches):
            self.clear()
        if hasattr(self, "_picking"):
            self._picking = None
        if hasattr(self, "_last_state"):
            self._last_state = None
        ViewerBase.clear_model(self)

    @override
    def clear_all_layers(self) -> None:
        """Clear all Newton layers and reset the complete OptiX scene."""
        if hasattr(self, "_mesh_ids") and (self._mesh_ids or self._batches):
            self.clear()
        for layer_id in [layer_id for layer_id in self._layers if layer_id != _DEFAULT_LAYER_ID]:
            del self._layers[layer_id]
        self._active_layer_id = _DEFAULT_LAYER_ID
        self._layers[_DEFAULT_LAYER_ID] = Layer(_DEFAULT_LAYER_ID)
        self.clear_model()

    @override
    def set_model(self, model) -> None:
        """Set the active model and synchronize shared UI state."""
        super().set_model(model)
        if model is not None:
            self.camera.up_axis = model.up_axis
        if self.gui is not None:
            self.gui.update_shape_counts(model)
        self._optix_ground_meshes = {
            shapes.mesh for shapes in self._shape_instances.values() if int(shapes.geo_type) == int(GeoType.PLANE)
        }
        self._optix_ground_subdivisions = {
            mesh: self._checker_subdivisions_for_mesh(mesh) for mesh in self._optix_ground_meshes
        }
        self._optix_default_material_meshes = {
            shapes.mesh
            for shapes in self._shape_instances.values()
            if int(shapes.geo_type) != int(GeoType.PLANE) and not self._has_authored_mesh_material(model, shapes)
        }
        self._optix_model_shape_batches = {shapes.name: shapes for shapes in self._shape_instances.values()}
        self._optix_palette_metadata.clear()
        self._optix_palette_device_metadata.clear()
        for shapes in self._shape_instances.values():
            metadata = self._create_palette_metadata(shapes)
            self._optix_palette_metadata[shapes.name] = metadata
            self._optix_palette_device_metadata[shapes.name] = (
                wp.array(metadata[0], dtype=wp.int32, device=self.device),
                wp.array(metadata[1], dtype=wp.vec3, device=self.device),
                wp.array(metadata[2].astype(np.int32), dtype=wp.int32, device=self.device),
            )
        self._optix_palette_color_arrays.clear()
        self._camera_dirty = True

    def _checker_subdivisions_for_mesh(self, mesh: str) -> tuple[float, float]:
        """Return subdivisions that produce world-meter checkers on a plane mesh."""
        if self._ground_checker_size is None:
            return 0.0, 0.0
        mesh_id = self._mesh_ids.get(mesh)
        if mesh_id is None:
            return 0.0, 0.0
        vertices = np.asarray(self._api.scene._meshes[mesh_id].vertices, dtype=np.float32)
        if len(vertices) == 0:
            return 0.0, 0.0
        extents = np.ptp(vertices, axis=0)
        return float(extents[0] / self._ground_checker_size), float(extents[1] / self._ground_checker_size)

    @override
    def log_shapes(
        self,
        name: str,
        geo_type: int,
        geo_scale: float | tuple[float, ...],
        xforms: wp.array[wp.transform],
        colors: wp.array[wp.vec3] | None = None,
        materials: wp.array[wp.vec4] | None = None,
        geo_thickness: float = 0.0,
        geo_is_solid: bool = True,
        geo_src: Any = None,
        hidden: bool = False,
    ) -> None:
        """Log shapes, adding metric checker materials to plane geometry."""
        if int(geo_type) == int(GeoType.PLANE) and self._ground_checker_size is not None:
            scale = [float(value) for value in geo_scale] if isinstance(geo_scale, tuple | list) else [float(geo_scale)]
            width = scale[0] if scale[0] > 0.0 else 1000.0
            length = scale[1] if len(scale) > 1 and scale[1] > 0.0 else width
            u_subdiv = width / self._ground_checker_size
            v_subdiv = length / self._ground_checker_size
            count = len(xforms)
            qualified_name = self._qualify(name)
            output = self._manual_ground_material_arrays.get(qualified_name)
            if output is None or len(output) != count:
                output = wp.empty(count, dtype=wp.vec4, device=self.device)
                self._manual_ground_material_arrays[qualified_name] = output
            if materials is None:
                output.fill_(wp.vec4(self._ground_roughness, 0.0, u_subdiv, v_subdiv))
            else:
                if len(materials) not in (1, count):
                    raise ValueError(f"Expected 1 or {count} materials, got {len(materials)}")
                wp.launch(
                    _apply_optix_plane_checker_material,
                    dim=count,
                    inputs=[materials, u_subdiv, v_subdiv],
                    outputs=[output],
                    device=self.device,
                )
            materials = output

        return super().log_shapes(
            name,
            geo_type,
            geo_scale,
            xforms,
            colors,
            materials,
            geo_thickness,
            geo_is_solid,
            geo_src,
            hidden,
        )

    def _create_palette_metadata(self, shapes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.fromiter(shapes.model_shapes, dtype=np.intp)
        defaults = np.asarray([ViewerBase._shape_color_map(int(index)) for index in indices], dtype=np.float32)
        eligible = np.asarray(
            [
                self.model.shape_source[int(index)] is None
                or getattr(self.model.shape_source[int(index)], "color", None) is None
                for index in indices
            ],
            dtype=np.bool_,
        )
        return indices, defaults, eligible

    @staticmethod
    def _validate_color_palette(palette: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
        colors = tuple(tuple(float(channel) for channel in color) for color in palette)
        if not colors:
            raise ValueError("default_color_palette must contain at least one color")
        if any(len(color) != 3 or not all(0.0 <= channel <= 1.0 for channel in color) for color in colors):
            raise ValueError("default_color_palette colors must contain three values in the range [0, 1]")
        return colors

    def set_default_color_palette(self, palette: Sequence[Sequence[float]]) -> None:
        """Set the OptiX palette for automatically colored shapes.

        The palette may be changed before or after :meth:`set_model`. Explicit
        shape colors and colors authored by mesh assets are not changed.

        Args:
            palette: Non-empty sequence of display-sRGB colors in [0, 1].
        """
        self._default_color_palette = self._validate_color_palette(palette)
        self._optix_palette_color_arrays.clear()
        self._optix_palette_array = None
        for shapes in self._shape_instances.values():
            shapes.colors_changed = True

    @staticmethod
    def _validate_time_of_day(value: float) -> float:
        value = float(value)
        if not 0.0 <= value <= 24.0:
            raise ValueError("time_of_day must be in the range [0, 24]")
        return value

    @staticmethod
    def _validate_sky_intensity(value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("sky_intensity must be finite and non-negative")
        return value

    @property
    def grayscale_sky(self) -> bool:
        """Whether the physical sky is rendered without chroma."""
        return self._grayscale_sky

    @grayscale_sky.setter
    def grayscale_sky(self, value: bool) -> None:
        value = bool(value)
        if value == self._grayscale_sky:
            return
        self._grayscale_sky = value
        self._apply_time_of_day(reset_history=True)

    @property
    def sky_intensity(self) -> float:
        """Procedural-sky illumination multiplier."""
        return self._sky_intensity

    @sky_intensity.setter
    def sky_intensity(self, value: float) -> None:
        value = self._validate_sky_intensity(value)
        if value == self._sky_intensity:
            return
        self._sky_intensity = value
        self._apply_time_of_day(reset_history=True)

    @property
    def time_of_day(self) -> float:
        """Procedural-sky time in hours, in the range [0, 24]."""
        return self._time_of_day

    @time_of_day.setter
    def time_of_day(self, value: float) -> None:
        value = self._validate_time_of_day(value)
        if value == self._time_of_day:
            return
        self._time_of_day = value
        self._apply_time_of_day(reset_history=True)

    @staticmethod
    def _smoothstep(edge0: float, edge1: float, value: float) -> float:
        value = min(max((value - edge0) / (edge1 - edge0), 0.0), 1.0)
        return value * value * (3.0 - 2.0 * value)

    def _apply_time_of_day(self, reset_history: bool) -> None:
        phase = math.pi * (self._time_of_day - 6.0) / 12.0
        sun_height = math.sin(phase)
        elevation = math.radians(60.0 * sun_height)
        azimuth = math.radians(15.0 * (self._time_of_day - 12.0))
        cos_elevation = math.cos(elevation)
        sun_direction = (
            cos_elevation * math.sin(azimuth),
            math.sin(elevation),
            cos_elevation * math.cos(azimuth),
        )

        daylight = self._smoothstep(-0.12, 0.08, sun_height)
        horizon = daylight * (1.0 - self._smoothstep(0.0, 0.55, sun_height))
        # The backend stores these values in linear RGB; this wrapper accepts
        # display-sRGB and converts them before forwarding.
        day_ground = (0.665185, 0.665185, 0.665185)
        night_ground = (0.151704, 0.172593, 0.220916)
        ground_color = tuple(
            night + daylight * (day - night) for night, day in zip(night_ground, day_ground, strict=True)
        )
        self.set_sky_parameters(
            sun_direction=sun_direction,
            multiplier=self._sky_intensity * (0.01 + 0.99 * daylight),
            haze=4.0 * horizon,
            red_blue_shift=0.55 * horizon,
            saturation=0.5 + 0.5 * daylight + 0.2 * horizon,
            ground_color=ground_color,
            horizon_blur=1.0 + horizon,
            night_color=(0.0, 0.0, 0.0),
            sun_disk_intensity=daylight * (1.0 - 0.25 * horizon),
            sun_disk_scale=1.0 + horizon,
            sun_glow_intensity=daylight * (1.0 + 2.0 * horizon),
            y_is_up=1,
            grayscale=self._grayscale_sky,
        )
        if reset_history:
            self.reset_temporal_history()

    def _palette_colors(self, name: str, colors: wp.array[wp.vec3]) -> wp.array[wp.vec3]:
        shapes = self._optix_model_shape_batches.get(name)
        if shapes is None or self.model is None:
            return colors

        device_metadata = self._optix_palette_device_metadata.get(name)
        if device_metadata is None:
            metadata = self._create_palette_metadata(shapes)
            self._optix_palette_metadata[name] = metadata
            device_metadata = (
                wp.array(metadata[0], dtype=wp.int32, device=self.device),
                wp.array(metadata[1], dtype=wp.vec3, device=self.device),
                wp.array(metadata[2].astype(np.int32), dtype=wp.int32, device=self.device),
            )
            self._optix_palette_device_metadata[name] = device_metadata
        indices, defaults, eligible = device_metadata

        if self._optix_palette_array is None:
            self._optix_palette_array = wp.array(self._default_color_palette, dtype=wp.vec3, device=self.device)

        palette_colors = self._optix_palette_color_arrays.get(name)
        if palette_colors is None or len(palette_colors) != len(colors):
            palette_colors = wp.empty(len(colors), dtype=wp.vec3, device=self.device)
            self._optix_palette_color_arrays[name] = palette_colors
        wp.launch(
            _apply_optix_color_palette,
            dim=len(colors),
            inputs=[colors, indices, defaults, eligible, self._optix_palette_array, palette_colors],
            device=self.device,
        )
        return palette_colors

    @staticmethod
    def _has_authored_mesh_material(model, shapes) -> bool:
        """Return whether a mesh batch carries authored PBR or texture data."""
        if int(shapes.geo_type) not in (int(GeoType.MESH), int(GeoType.CONVEX_MESH)):
            return False
        for shape_index in shapes.model_shapes:
            source = model.shape_source[int(shape_index)]
            if source is not None and any(
                getattr(source, name, None) is not None for name in ("roughness", "metallic", "texture")
            ):
                return True
        return False

    @override
    def set_camera(self, pos: wp.vec3, pitch: float, yaw: float) -> None:
        """Set camera pose using the same convention as :class:`ViewerGL`.

        Args:
            pos: Camera position [m].
            pitch: Camera pitch in degrees.
            yaw: Camera yaw in degrees.
        """
        self.camera.pos = self.camera._as_vec3(pos)
        self.camera.pitch = max(min(float(pitch), 89.0), -89.0)
        self.camera.yaw = (float(yaw) + 180.0) % 360.0 - 180.0
        self.camera.sync_pivot_to_view()
        self._camera_dirty = True
        self._sync_camera()

    def _sync_camera(self) -> None:
        if not hasattr(self, "camera"):
            return
        self._camera_position = np.asarray(self.camera.pos, dtype=np.float32)
        self._camera_pitch = float(self.camera.pitch)
        self._camera_yaw = float(self.camera.yaw)
        self._camera_fov = float(self.camera.fov)
        if not self._initialized:
            return
        rotation = self._global_transform[:3, :3]
        position = rotation @ self._camera_position
        front = rotation @ np.asarray(self.camera.get_front(), dtype=np.float32)
        up = rotation @ np.asarray(self.camera.get_up(), dtype=np.float32)
        self._api.set_camera_look_at(position, position + front, up, self.camera.fov)
        self._camera_dirty = False

    def _copy_backend_camera(self) -> None:
        self.camera.pos = self.camera._as_vec3(self._camera_position)
        self.camera.pitch = float(self._camera_pitch)
        self.camera.yaw = float(self._camera_yaw)
        self.camera.fov = float(self._camera_fov)
        self.camera.sync_pivot_to_view()
        self._camera_dirty = True

    def _update_camera_from_input(self, dt: float) -> None:
        if self.gui is not None:
            self.gui.update_camera_from_keys(dt, self.is_key_down)
        else:
            super()._update_camera_from_input(dt)
            self._copy_backend_camera()
        if self._camera_dirty:
            self._sync_camera()

    def _to_framebuffer_coords(self, x: float, y: float) -> tuple[float, float]:
        if self._presenter is None:
            return float(x), float(y)
        window = self._presenter.window
        fb_width, fb_height = window.get_framebuffer_size()
        window_width, window_height = window.get_size()
        if window_width <= 0 or window_height <= 0:
            return float(x), float(y)
        return float(x) * fb_width / window_width, float(y) * fb_height / window_height

    def _is_ctrl_down(self) -> bool:
        if self._presenter is None:
            return False
        key = self._presenter.pyglet.window.key
        return self.is_key_down(key.LCTRL) or self.is_key_down(key.RCTRL)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y) -> None:
        del x, y, scroll_x
        if self.gui is not None:
            self.gui.handle_mouse_scroll(scroll_y, is_ctrl_down=self._is_ctrl_down())
        else:
            super().on_mouse_scroll(0, 0, 0, scroll_y)
            self._copy_backend_camera()

    def on_mouse_press(self, x, y, button, modifiers) -> None:
        del modifiers
        if self.gui is not None:
            self.gui.handle_mouse_press(x, y, button, self._to_framebuffer_coords)
        else:
            super().on_mouse_press(x, y, button, 0)

    def on_mouse_release(self, x, y, button, modifiers) -> None:
        del modifiers
        if self.gui is not None:
            self.gui.handle_mouse_release(x, y, button)
        else:
            super().on_mouse_release(x, y, button, 0)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers) -> None:
        if self.gui is not None:
            self.gui.handle_mouse_drag(x, y, dx, dy, buttons, self._to_framebuffer_coords, modifiers)
        else:
            super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)
            self._copy_backend_camera()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        """Handle common ViewerGL keys and OptiX debug/recording keys."""
        if self.gui is not None:
            if self.gui.should_ignore_keyboard_input():
                return
            key = self._presenter.pyglet.window.key
            if symbol in (key.SPACE, key.PERIOD, key.H, key.F, key.ESCAPE):
                self.gui.handle_key_press(symbol, close_fn=self.close)
                return
        super().on_key_press(symbol, modifiers)

    def _on_resize(self, width: int, height: int) -> None:
        super()._on_resize(width, height)
        if hasattr(self, "camera"):
            self.camera.update_screen_size(width, height)
            self._camera_dirty = True
        self._destroy_overlay_depth_resources()

    @override
    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden: bool = False):
        """Log instances with OptiX defaults for un-authored materials."""
        is_ground = mesh in self._optix_ground_meshes
        count = 0 if xforms is None else len(xforms)
        if colors is not None and not is_ground:
            colors = self._palette_colors(name, colors)
        if is_ground and colors is not None:
            if count not in self._ground_color_arrays:
                ground_colors = wp.zeros(count, dtype=wp.vec3, device=self.device)
                if count > 0:
                    ground_colors.fill_(wp.vec3(*self._ground_color))
                self._ground_color_arrays[count] = ground_colors
            colors = self._ground_color_arrays[count]

        roughness = None
        u_subdiv = 0.0
        v_subdiv = 0.0
        if is_ground:
            roughness = self._ground_roughness
            u_subdiv, v_subdiv = self._optix_ground_subdivisions.get(mesh, (0.0, 0.0))
        elif mesh in self._optix_default_material_meshes:
            roughness = self._default_roughness
        if materials is not None and roughness is not None:
            key = (count, roughness, u_subdiv, v_subdiv)
            if key not in self._material_arrays:
                material_array = wp.zeros(count, dtype=wp.vec4, device=self.device)
                if count > 0:
                    material_array.fill_(wp.vec4(roughness, 0.0, u_subdiv, v_subdiv))
                self._material_arrays[key] = material_array
            materials = self._material_arrays[key]
        if materials is None and isinstance(colors, wp.array) and colors.device.is_cuda:
            key = (count, self._default_roughness, 0.0, 0.0)
            if key not in self._material_arrays:
                material_array = wp.zeros(count, dtype=wp.vec4, device=self.device)
                if count > 0:
                    material_array.fill_(wp.vec4(self._default_roughness, 0.0, 0.0, 0.0))
                self._material_arrays[key] = material_array
            materials = self._material_arrays[key]
        return super().log_instances(name, mesh, xforms, scales, colors, materials, hidden=hidden)

    @override
    def log_lines(
        self,
        name: str,
        starts: wp.array[wp.vec3] | None,
        ends: wp.array[wp.vec3] | None,
        colors: wp.array[wp.vec3] | wp.array[wp.float32] | tuple[float, float, float] | list[float] | None,
        width: float = 0.01,
        hidden: bool = False,
    ) -> None:
        """Log debug lines through Newton's shared OpenGL overlay path."""
        del width
        if self._presenter is None:
            return
        from .gl.opengl import update_line_batch  # noqa: PLC0415

        update_line_batch(self.lines, self._qualify(name), starts, ends, colors, self.device, hidden=hidden)

    @override
    def log_arrows(
        self,
        name: str,
        starts: wp.array[wp.vec3] | None,
        ends: wp.array[wp.vec3] | None,
        colors: wp.array[wp.vec3] | wp.array[wp.float32] | tuple[float, float, float] | list[float] | None,
        width: float = 0.01,
        hidden: bool = False,
    ) -> None:
        """Log contact arrows through Newton's shared OpenGL overlay path."""
        del width
        if self._presenter is None:
            return
        from .gl.opengl import update_line_batch  # noqa: PLC0415

        update_line_batch(self.arrows, self._qualify(name), starts, ends, colors, self.device, hidden=hidden)

    @override
    def end_frame(self) -> None:
        """Render and present a frame, including while simulation is paused."""
        super().end_frame()

    @override
    def is_key_down(self, key: str | int) -> bool:
        """Return whether a keyboard key is currently held.

        Args:
            key: Character, special-key name, or pyglet key code.

        Returns:
            Whether the key is held.
        """
        if self._presenter is None:
            return False
        key_module = self._presenter.pyglet.window.key
        if isinstance(key, str):
            normalized = key.lower()
            if len(normalized) == 1 and normalized.isalpha():
                key = getattr(key_module, normalized.upper(), None)
            elif len(normalized) == 1 and normalized.isdigit():
                key = getattr(key_module, f"_{normalized}", None)
            else:
                key = {
                    "space": key_module.SPACE,
                    "escape": key_module.ESCAPE,
                    "esc": key_module.ESCAPE,
                    "enter": key_module.ENTER,
                    "return": key_module.ENTER,
                    "tab": key_module.TAB,
                    "shift": key_module.LSHIFT,
                    "ctrl": key_module.LCTRL,
                    "alt": key_module.LALT,
                    "up": key_module.UP,
                    "down": key_module.DOWN,
                    "left": key_module.LEFT,
                    "right": key_module.RIGHT,
                    "backspace": key_module.BACKSPACE,
                    "delete": key_module.DELETE,
                }.get(normalized)
        return key is not None and key in self._keys_down

    @override
    def close(self) -> None:
        """Release the shared UI and OptiX presentation resources."""
        self._destroy_simulation_stream()
        ui = self.ui
        if ui is not None:
            ui.shutdown()
        if self._presenter is not None:
            self._presenter._on_draw_overlay = None
            for batches in (self.lines, self.arrows):
                for batch in batches.values():
                    batch.destroy()
                batches.clear()
            self._destroy_overlay_depth_resources()
        self.gui = None
        super().close()

    def get_frame(self, target_image: wp.array[Any] | None = None, render_ui: bool = False) -> wp.array[Any]:
        """Return the latest rendered RGB frame as a Warp array.

        Args:
            target_image: Optional ``(height, width, 3)`` uint8 output array.
            render_ui: Reserved for compatibility with :class:`ViewerGL`.

        Returns:
            RGB image with a top-left origin.
        """
        del render_ui
        frame = np.ascontiguousarray(self._api.get_frame_uint8()[..., :3])
        expected_shape = (self.height, self.width, 3)
        if frame.shape != expected_shape:
            raise RuntimeError(f"OptiX returned frame shape {frame.shape}, expected {expected_shape}")
        if target_image is None:
            return wp.array(frame, dtype=wp.uint8, device=self.device)
        if target_image.shape != expected_shape:
            raise ValueError(f"Shape of target_image must be {expected_shape}, got {target_image.shape}")
        if target_image.dtype != wp.uint8:
            raise ValueError(f"Dtype of target_image must be wp.uint8, got {target_image.dtype}")
        target_image.assign(frame)
        return target_image
