# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OptiX Path Tracing Viewer.
Python/OptiX path tracing viewer with DLSS RR support.

This viewer renders a scene using OptiX ray tracing with PBR materials,
displaying raw buffers (radiance, normals, depth, etc.) for debugging.
"""

import sys
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import warp as wp

# Initialize warp
wp.init()

logger = logging.getLogger(__name__)

# Import local modules
from ..optix_context import _create_optix_context
from ..hit_kernels import HitKernel
from ..sbt_helpers import SbtKernelManager
from .camera import Camera
from .environment_map import EnvironmentMap
from .ptx_compiler import build_ptx, get_optix_include_dir
from .scene import Scene
from .tonemap import Tonemapper


@wp.kernel
def _reset_accum_buffer(accum: wp.array2d(dtype=wp.vec4)):
    x, y = wp.tid()
    accum[y, x] = wp.vec4(0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _accumulate_sample(
    sample: wp.array2d(dtype=wp.vec4),
    accum: wp.array2d(dtype=wp.vec4),
    sample_index: int,
):
    x, y = wp.tid()
    s = sample[y, x]
    a = accum[y, x]
    t = 1.0 / float(sample_index + 1)
    accum[y, x] = a + (s - a) * t


class PathTracingViewer:
    """
    OptiX Path Tracing Viewer.

    Renders a scene using hardware ray tracing with PBR materials.
    """

    # Output modes
    OUTPUT_FINAL = 0
    OUTPUT_RADIANCE = 1
    OUTPUT_DEPTH = 2
    OUTPUT_MOTION = 3
    OUTPUT_NORMAL = 4
    OUTPUT_ROUGHNESS = 5
    OUTPUT_DIFFUSE = 6
    OUTPUT_SPECULAR = 7
    OUTPUT_SPEC_HITDIST = 8

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        scene_setup: Optional[Callable[[Scene], None]] = None,
        camera: Optional[Camera] = None,
        accumulate_samples: bool = False,
        samples_per_frame: int = 1,
        max_bounces: int = 4,
        direct_light_samples: int = 1,
        use_halton_jitter: bool = True,
        enable_dlss_rr: bool = True,
    ):
        """
        Initialize the path tracing viewer.

        Args:
            width: Render width
            height: Render height
        """
        self.width = width
        self.height = height
        self._render_width = width
        self._render_height = height
        self.frame_index = 0
        self.sample_index = 0
        self.accumulate_samples = accumulate_samples
        self.samples_per_frame = max(1, int(samples_per_frame))
        self.max_bounces = max(1, int(max_bounces))
        self.direct_light_samples = max(1, int(direct_light_samples))
        self.use_halton_jitter = bool(use_halton_jitter)
        self.enable_dlss_rr = bool(enable_dlss_rr)

        # Camera
        if camera is None:
            self.camera = Camera(
                position=(0.0, 0.0, 6.0),
                target=(0.0, 0.0, 0.0),
                fov=45.0,
                aspect_ratio=width / height,
            )
        else:
            self.camera = camera
            self.camera.set_aspect_ratio(width, height)

        # Optional external scene configuration callback
        self._scene_setup = scene_setup

        # Default to path-traced final output.
        self.output_mode = self.OUTPUT_FINAL

        # OptiX state (initialized in build())
        self._optix = None
        self._ctx = None
        self._pipeline = None
        self._sbt = None
        self._ptx = None

        # Scene
        self._scene = None

        # Tonemapper
        self._tonemapper = Tonemapper(width, height)

        # Output buffers
        self._color_buffer = None
        self._accum_buffer = None
        self._normal_roughness_buffer = None
        self._motion_buffer = None
        self._depth_buffer = None
        self._diffuse_buffer = None
        self._specular_buffer = None
        self._spec_hit_dist_buffer = None
        self._dlss_output_buffer = None
        self._instance_transforms_buffer = None
        self._prev_instance_transforms_buffer = None
        self._prev_instance_transforms_valid = False

        # Launch params buffer — cached to avoid per-frame allocation
        self._launch_params_buffer = None
        self._launch_params_dtype = None
        self._launch_params_np = None
        self._launch_params_size = 0
        self._instance_transform_count = 0

        # CUDA surface objects
        self._color_surface = None
        self._dlss_context = None
        self._dlss_denoiser = None
        self._dlss_color_in_tex = None
        self._dlss_normal_roughness_tex = None
        self._dlss_motion_tex = None
        self._dlss_depth_tex = None
        self._dlss_diffuse_tex = None
        self._dlss_specular_tex = None
        self._dlss_spec_hit_dist_tex = None
        self._dlss_color_out_tex = None
        self._dlss_output_surface = 0
        self._dlss_enabled = False
        self._dlss_reset_history = True
        self._last_jitter = (0.0, 0.0)

        # Previous-frame camera matrices for motion vectors.
        self._prev_view = None
        self._prev_proj = None
        self._prev_mvp = None
        self._last_accum_view = None
        self._last_accum_proj = None
        self._sync_prev_camera_matrices_to_current()
        self._last_output_mode = self.output_mode

        # Physical sky defaults aligned with the upstream DLSS-RR sample behavior.
        self.sky_rgb_unit_conversion = (1.0 / 80000.0, 1.0 / 80000.0, 1.0 / 80000.0)
        self.sky_multiplier = 1.0
        self.sky_haze = 0.0
        self.sky_redblueshift = 0.0
        self.sky_saturation = 1.0
        self.sky_horizon_height = 0.0
        self.sky_ground_color = (0.4, 0.4, 0.4)
        self.sky_horizon_blur = 1.0
        self.sky_night_color = (0.0, 0.0, 0.0)
        self.sky_sun_disk_intensity = 1.0
        self.sky_sun_direction = (0.0, 1.0, 0.5)
        self.sky_sun_disk_scale = 1.0
        self.sky_sun_glow_intensity = 1.0
        self.sky_y_is_up = 1

        # Optional HDR environment map (lat-long, RGBA32F).
        self._env_map: EnvironmentMap | None = None

    def set_environment_hdr(self, hdr_path: str, scaling: float = 1.0):
        """
        Load an HDR environment map from disk.

        The environment map is used for image-based lighting with importance sampling.

        Args:
            hdr_path: Path to HDR file (.hdr format)
            scaling: Intensity multiplier (default 1.0)
        """
        env_map = EnvironmentMap()
        if env_map.load_from_file(hdr_path, scaling=scaling):
            self._env_map = env_map
        else:
            logger.warning("Failed to load HDR environment: %s", hdr_path)

    def set_environment_color(self, color: tuple[float, float, float]):
        """
        Set a uniform color environment (useful for debugging or simple scenes).

        Args:
            color: RGB color values
        """
        env_map = EnvironmentMap()
        if env_map.load_from_color(color):
            self._env_map = env_map

    def clear_environment_map(self):
        """Clear the HDR environment map and use procedural sky only."""
        self._env_map = None

    @property
    def tonemapped_output(self):
        """Return the tonemapped output buffer used for display/extraction."""
        return self._tonemapper.output

    def build(self):
        """Build the OptiX pipeline and scene."""
        logger.info("Initializing OptiX path tracing viewer.")

        # Import optix
        try:
            import optix
        except ImportError:
            logger.error("Could not import optix module. Ensure warp.pyoptix and OptiX SDK are installed.")
            return False

        self._optix = optix

        # Get OptiX include directory
        optix_include = get_optix_include_dir(optix)
        if not optix_include:
            logger.error("Could not find OptiX SDK include directory.")
            return False

        logger.info("Using OptiX SDK include directory: %s", optix_include)

        # Create OptiX context
        wp_device = wp.get_device("cuda")
        cu_context = wp_device.context.value if hasattr(wp_device.context, "value") else int(wp_device.context)
        self._ctx, self._optix_logger = _create_optix_context(optix, int(cu_context))

        # Build PTX
        headers_dir = Path(__file__).parent / "cpp"
        self._ptx = build_ptx(optix_include, headers_dir)

        # Create scene
        self._scene = Scene(self._ctx)
        if self._scene_setup is not None:
            self._scene_setup(self._scene)
        else:
            self._scene.create_cornell_box()
        self._scene.build(optix)

        # Create output buffers
        self._create_buffers()
        self._init_dlss_rr()

        # Create pipeline
        self._create_pipeline()

        # Create SBT
        self._create_sbt()

        logger.info("OptiX path tracing viewer build complete.")
        return True

    @staticmethod
    def _create_cuda_texture_2d(
        height: int, width: int, channels: int, *, surface_access: bool = False
    ) -> wp.Texture2D:
        if channels == 1:
            data = np.zeros((height, width), dtype=np.float32)
        else:
            data = np.zeros((height, width, channels), dtype=np.float32)
        return wp.Texture2D(
            data,
            filter_mode=wp.TextureFilterMode.CLOSEST,
            address_mode=wp.TextureAddressMode.CLAMP,
            device="cuda",
            surface_access=surface_access,
        )

    @staticmethod
    def _half_res(value: int) -> int:
        # Keep dimensions even (where possible) to match common DLSS input expectations.
        v = max(1, int(value) // 2)
        if v > 1 and (v % 2) != 0:
            v -= 1
        return max(1, v)

    def _set_render_resolution(self, render_width: int, render_height: int):
        rw = max(1, int(render_width))
        rh = max(1, int(render_height))
        if rw == self._render_width and rh == self._render_height:
            return
        self._render_width = rw
        self._render_height = rh
        self._create_buffers()
        self.frame_index = 0
        self.sample_index = 0
        self._dlss_reset_history = True

    def _sync_prev_camera_matrices_to_current(self):
        """Initialize previous-frame camera transforms from the current camera pose.

        Mirrors reference first-frame behavior where prevMVP is set to currentMVP to avoid
        spurious large motion vectors after resets/resizes.
        """
        view = self.camera.get_view_matrix().copy()
        proj = self.camera.get_projection_matrix().copy()
        self._prev_view = view
        self._prev_proj = proj
        self._prev_mvp = (view @ proj).astype(np.float32)
        self._last_accum_view = view.copy()
        self._last_accum_proj = proj.copy()

    def _destroy_dlss_rr(self):
        # Surface object lifetime is owned by the Texture2D instance.
        # Clearing references lets texture cleanup release CUDA resources.
        self._dlss_output_surface = 0

        self._dlss_color_in_tex = None
        self._dlss_normal_roughness_tex = None
        self._dlss_motion_tex = None
        self._dlss_depth_tex = None
        self._dlss_diffuse_tex = None
        self._dlss_specular_tex = None
        self._dlss_spec_hit_dist_tex = None
        self._dlss_color_out_tex = None
        self._dlss_output_buffer = None

        if self._dlss_denoiser is not None:
            try:
                self._dlss_denoiser.deinit()
            except Exception as exc:
                logger.warning("Failed to deinitialize DLSS denoiser: %s", exc)
        self._dlss_denoiser = None

        if self._dlss_context is not None:
            try:
                self._dlss_context.deinit()
            except Exception as exc:
                logger.warning("Failed to deinitialize DLSS context: %s", exc)
        self._dlss_context = None
        self._dlss_enabled = False
        # If DLSS gets disabled at runtime, restore full-resolution rendering.
        self._set_render_resolution(self.width, self.height)

    def _init_dlss_rr(self):
        self._destroy_dlss_rr()
        if not self.enable_dlss_rr or self._optix is None:
            return

        required = ("DlssRRContext", "DlssRRInitInfo", "DlssRRResource", "DlssPerfQuality")
        if not all(hasattr(self._optix, name) for name in required):
            logger.info("DLSS RR bindings not present in optix module.")
            return

        try:
            context = self._optix.DlssRRContext()
            context.init()
            if not context.isDlssRRAvailable():
                logger.info("DLSS RR not available on this system.")
                return

            init_info = self._optix.DlssRRInitInfo()
            render_width = self._half_res(self.width)
            render_height = self._half_res(self.height)
            init_info.inputWidth = int(render_width)
            init_info.inputHeight = int(render_height)
            init_info.outputWidth = int(self.width)
            init_info.outputHeight = int(self.height)
            # Prefer an upscaling profile; fallback if binding enum names differ.
            quality_enum = self._optix.DlssPerfQuality
            quality_name = "MAX_QUALITY"
            if not hasattr(quality_enum, quality_name):
                quality_name = "BALANCED" if hasattr(quality_enum, "BALANCED") else "DLAA"
            init_info.quality = getattr(quality_enum, quality_name)
            init_info.preset = self._optix.RayReconstructionHintRenderPreset.DEFAULT
            # Match reference behavior:
            # - MVJittered=false while still passing per-frame jitter to denoise()
            # - lowResolutionMotionVectors=true (motion vectors provided at render resolution)
            init_info.mvJittered = False
            init_info.lowResolutionMotionVectors = True
            init_info.isContentHDR = True
            init_info.depthInverted = False
            init_info.autoExposure = False
            init_info.useHWDepth = False

            # Ask NGX for the optimal input size for the chosen quality mode,
            # for the selected quality mode, and only fallback to half-res on failure.
            if hasattr(context, "querySupportedDlssInputSizes"):
                try:
                    sizes = context.querySupportedDlssInputSizes(int(self.width), int(self.height), init_info.quality)
                    ow = int(getattr(sizes, "optimalWidth", 0))
                    oh = int(getattr(sizes, "optimalHeight", 0))
                    if ow > 0 and oh > 0:
                        render_width = ow
                        render_height = oh
                        init_info.inputWidth = int(render_width)
                        init_info.inputHeight = int(render_height)
                except Exception as exc:
                    logger.warning("Failed to query optimal DLSS input size; using half-res fallback: %s", exc)

            denoiser = context.initDlssRR(init_info)
            self._set_render_resolution(render_width, render_height)

            self._dlss_color_in_tex = self._create_cuda_texture_2d(self._render_height, self._render_width, 4)
            self._dlss_normal_roughness_tex = self._create_cuda_texture_2d(self._render_height, self._render_width, 4)
            self._dlss_motion_tex = self._create_cuda_texture_2d(self._render_height, self._render_width, 2)
            self._dlss_depth_tex = self._create_cuda_texture_2d(self._render_height, self._render_width, 1)
            self._dlss_diffuse_tex = self._create_cuda_texture_2d(self._render_height, self._render_width, 4)
            self._dlss_specular_tex = self._create_cuda_texture_2d(self._render_height, self._render_width, 4)
            self._dlss_spec_hit_dist_tex = self._create_cuda_texture_2d(self._render_height, self._render_width, 1)
            self._dlss_color_out_tex = self._create_cuda_texture_2d(
                self.height, self.width, 4, surface_access=True
            )
            self._dlss_output_buffer = wp.zeros((self.height, self.width), dtype=wp.vec4, device="cuda")
            self._dlss_output_surface = self._dlss_color_out_tex.cuda_surface

            res = self._optix.DlssRRResource
            denoiser.setResource(res.RESOURCE_COLOR_IN, self._dlss_color_in_tex.cuda_texture)
            denoiser.setResource(res.RESOURCE_COLOR_OUT, self._dlss_output_surface)
            denoiser.setResource(res.RESOURCE_NORMALROUGHNESS, self._dlss_normal_roughness_tex.cuda_texture)
            denoiser.setResource(res.RESOURCE_MOTIONVECTOR, self._dlss_motion_tex.cuda_texture)
            denoiser.setResource(res.RESOURCE_LINEARDEPTH, self._dlss_depth_tex.cuda_texture)
            denoiser.setResource(res.RESOURCE_DIFFUSE_ALBEDO, self._dlss_diffuse_tex.cuda_texture)
            denoiser.setResource(res.RESOURCE_SPECULAR_ALBEDO, self._dlss_specular_tex.cuda_texture)
            denoiser.setResource(res.RESOURCE_SPECULAR_HITDISTANCE, self._dlss_spec_hit_dist_tex.cuda_texture)

            self._dlss_context = context
            self._dlss_denoiser = denoiser
            self._dlss_enabled = True
            self._dlss_reset_history = True
            logger.info(
                "DLSS Ray Reconstruction enabled (render=%dx%d, output=%dx%d).",
                self._render_width,
                self._render_height,
                self.width,
                self.height,
            )
        except Exception as exc:
            logger.warning("Failed to initialize DLSS RR: %s", exc)
            self._destroy_dlss_rr()

    def _copy_linear_to_dlss_textures(self):
        if not self._dlss_enabled:
            return
        copies = (
            (self._color_buffer, self._dlss_color_in_tex),
            (self._normal_roughness_buffer, self._dlss_normal_roughness_tex),
            (self._motion_buffer, self._dlss_motion_tex),
            (self._depth_buffer, self._dlss_depth_tex),
            (self._diffuse_buffer, self._dlss_diffuse_tex),
            (self._specular_buffer, self._dlss_specular_tex),
            (self._spec_hit_dist_buffer, self._dlss_spec_hit_dist_tex),
        )
        for src_buffer, dst_tex in copies:
            dst_tex.copy_from_array(src_buffer)

    def _copy_dlss_output_to_color(self):
        if not self._dlss_enabled:
            return
        if self._dlss_output_buffer is None:
            return
        self._dlss_color_out_tex.copy_to_array(self._dlss_output_buffer)

    def _run_dlss_rr(self, reset: bool):
        if not self._dlss_enabled or self._dlss_denoiser is None:
            return False
        try:
            # Match upstream matrix packing in the Vulkan DLSS-RR sample:
            # output in column-major memory order (m11,m21,m31,m41,...).
            view_m = self.camera.get_view_matrix().astype(np.float32)
            proj_m = self.camera.get_projection_matrix().astype(np.float32)
            view = view_m.T.reshape(-1).tolist()
            proj = proj_m.T.reshape(-1).tolist()
            self._dlss_denoiser.denoise(
                int(self._render_width),
                int(self._render_height),
                float(-self._last_jitter[0]),
                float(-self._last_jitter[1]),
                view,
                proj,
                bool(reset or self._dlss_reset_history),
                int(0),
                int(0),
                float(1.0),
                float(1.0),
            )
            self._dlss_reset_history = False
            return True
        except Exception as exc:
            logger.warning("DLSS denoise failed; disabling DLSS RR: %s", exc)
            self._destroy_dlss_rr()
            return False

    def _create_buffers(self):
        """Create output buffers."""
        # HDR color buffer
        self._color_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec4, device="cuda")
        self._accum_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec4, device="cuda")

        # G-buffer outputs
        self._normal_roughness_buffer = wp.zeros(
            (self._render_height, self._render_width), dtype=wp.vec4, device="cuda"
        )
        self._motion_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec2, device="cuda")
        self._depth_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.float32, device="cuda")
        self._diffuse_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec4, device="cuda")
        self._specular_buffer = wp.zeros((self._render_height, self._render_width), dtype=wp.vec4, device="cuda")
        self._spec_hit_dist_buffer = wp.zeros(
            (self._render_height, self._render_width), dtype=wp.float32, device="cuda"
        )

    def _update_instance_transform_buffers(self):
        """Upload current instance transforms for motion vectors.

        The *previous* transforms buffer is populated by
        :meth:`_snapshot_instance_transforms` which copies the current
        buffer to the previous buffer **after** each rendered frame.
        This guarantees that ``prev`` always holds exactly the transforms
        that were used for the last rendered frame.
        """
        if self._scene is None or self._scene.instance_count == 0:
            self._instance_transforms_buffer = None
            self._prev_instance_transforms_buffer = None
            self._instance_transform_count = 0
            return

        instances = getattr(self._scene, "_instances", None)
        if not instances:
            self._instance_transforms_buffer = None
            self._prev_instance_transforms_buffer = None
            self._instance_transform_count = 0
            return

        count = len(instances)

        # (Re)allocate GPU buffers when the instance count changes.
        if count != self._instance_transform_count:
            self._instance_xform_curr_np = np.empty((count, 12), dtype=np.float32)
            self._instance_transforms_buffer = wp.empty(count * 12, dtype=wp.float32, device="cuda")
            self._prev_instance_transforms_buffer = wp.empty(count * 12, dtype=wp.float32, device="cuda")
            self._instance_transform_count = count
            self._prev_instance_transforms_valid = False

        curr = self._instance_xform_curr_np
        for i, inst in enumerate(instances):
            m = np.asarray(inst.transform, dtype=np.float32).reshape(4, 4)
            curr[i, 0:4] = m[0, :]
            curr[i, 4:8] = m[1, :]
            curr[i, 8:12] = m[2, :]

        self._instance_transforms_buffer.assign(curr.reshape(-1))

        # First frame (or after resize): prev == current so motion is zero.
        if not self._prev_instance_transforms_valid:
            wp.copy(self._prev_instance_transforms_buffer, self._instance_transforms_buffer)
            self._prev_instance_transforms_valid = True

    def _snapshot_instance_transforms(self):
        """Copy current instance transforms to the previous-frame buffer on GPU.

        Must be called once per frame **after** the OptiX launch so that the
        next frame sees the correct previous-frame transforms for rigid-body
        motion vectors.
        """
        if (
            self._instance_transforms_buffer is not None
            and self._prev_instance_transforms_buffer is not None
            and self._instance_transforms_buffer.shape == self._prev_instance_transforms_buffer.shape
        ):
            wp.copy(self._prev_instance_transforms_buffer, self._instance_transforms_buffer)

    def _create_pipeline(self):
        """Create the OptiX pipeline."""
        optix = self._optix
        pipeline_kwargs = {
            "usesMotionBlur": False,
            "traversableGraphFlags": int(optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING),
            "numPayloadValues": 19,
            "numAttributeValues": 2,
            "exceptionFlags": int(optix.EXCEPTION_FLAG_NONE),
            "pipelineLaunchParamsVariableName": "params",
        }
        if optix.version()[1] >= 2:
            pipeline_kwargs["usesPrimitiveTypeFlags"] = int(optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE)
        pco = optix.PipelineCompileOptions(**pipeline_kwargs)

        mco = optix.ModuleCompileOptions(
            maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
            debugLevel=optix.COMPILE_DEBUG_LEVEL_DEFAULT,
        )
        module_result = self._ctx.moduleCreate(mco, pco, self._ptx)
        if isinstance(module_result, tuple):
            self._module = module_result[0]
        else:
            self._module = module_result

        # Use shared pyoptix SBT helper framework.
        self._sbt_manager = SbtKernelManager(optix, self._ctx, self._module, num_ray_subtypes=2)
        self._sbt_manager.set_raygen_kernel("__raygen__primary")
        self._sbt_manager.add_miss_kernels(["__miss__primary", "__miss__secondary"])
        self._sbt_manager.register_hit_shader_type(
            HitKernel("__closesthit__primary"),
            HitKernel("__closesthit__secondary", any_hit="__anyhit__secondary"),
        )

        plo = optix.PipelineLinkOptions()
        plo.maxTraceDepth = 2
        groups = self._sbt_manager.get_all_program_groups()
        self._pipeline = self._ctx.pipelineCreate(
            pco,
            plo,
            groups,
            "",
        )
        self._pipeline.setStackSize(2048, 2048, 2048, 2)

    def _create_sbt(self):
        """Create the Shader Binding Table."""
        sbt_resources = self._sbt_manager.build_sbt(device="cuda")
        self._sbt = sbt_resources.sbt
        self._sbt_keepalive = sbt_resources.keepalive

    @staticmethod
    def _halton(index: int, base: int) -> float:
        f = 1.0
        r = 0.0
        i = max(0, int(index))
        b = max(2, int(base))
        while i > 0:
            f /= float(b)
            r += f * float(i % b)
            i //= b
        return r

    def _get_launch_params_dtype(self):
        """Return the cached numpy structured dtype for launch params."""
        if self._launch_params_dtype is not None:
            return self._launch_params_dtype

        sky_dtype = np.dtype(
            [
                ("rgbUnitConversion", np.float32, (3,)),
                ("multiplier", np.float32),
                ("haze", np.float32),
                ("redblueshift", np.float32),
                ("saturation", np.float32),
                ("horizonHeight", np.float32),
                ("groundColor", np.float32, (3,)),
                ("horizonBlur", np.float32),
                ("nightColor", np.float32, (3,)),
                ("sunDiskIntensity", np.float32),
                ("sunDirection", np.float32, (3,)),
                ("sunDiskScale", np.float32),
                ("sunGlowIntensity", np.float32),
                ("yIsUp", np.int32),
            ],
            align=True,
        )

        self._launch_params_dtype = np.dtype(
            [
                ("tlas", np.uint64),
                ("view", np.float32, (16,)),
                ("proj", np.float32, (16,)),
                ("prevView", np.float32, (16,)),
                ("prevProj", np.float32, (16,)),
                ("prevMVP", np.float32, (16,)),
                ("viewInv", np.float32, (16,)),
                ("projInv", np.float32, (16,)),
                ("jitter", np.float32, (2,)),
                ("envIntensity", np.float32, (4,)),
                ("envRotation", np.float32),
                ("flags", np.uint32),
                ("skyInfo", sky_dtype),
                ("materialAddress", np.uint64),
                ("compactMaterialAddress", np.uint64),
                ("instanceMaterialIdAddress", np.uint64),
                ("instanceRenderPrimIdAddress", np.uint64),
                ("renderPrimitiveAddress", np.uint64),
                ("instanceTransformsAddress", np.uint64),
                ("prevInstanceTransformsAddress", np.uint64),
                ("materialCount", np.uint32),
                ("instanceCount", np.uint32),
                ("renderPrimCount", np.uint32),
                ("frameIndex", np.uint32),
                ("maxBounces", np.uint32),
                ("directLightSamples", np.uint32),
                ("textureDescAddress", np.uint64),
                ("textureDataAddress", np.uint64),
                ("textureCount", np.uint32),
                ("_pad0", np.uint32),
                ("envMapAddress", np.uint64),
                ("envMapWidth", np.uint32),
                ("envMapHeight", np.uint32),
                ("envMapFormat", np.uint32),
                ("_pad1", np.uint32),
                ("envAccelAddress", np.uint64),
                ("envMapIntegral", np.float32),
                ("envMapAverage", np.float32),
                ("colorOutput", np.uint64),
                ("normalRoughnessOutput", np.uint64),
                ("motionVectorOutput", np.uint64),
                ("depthOutput", np.uint64),
                ("diffuseAlbedoOutput", np.uint64),
                ("specularAlbedoOutput", np.uint64),
                ("specHitDistOutput", np.uint64),
                ("outputMode", np.int32),
                ("_pad2", np.int32),
            ],
            align=True,
        )
        return self._launch_params_dtype

    @staticmethod
    def _addr_u64(value) -> np.uint64:
        return np.uint64(0 if value is None else value)

    def _update_launch_params(self, frame_index_override: int | None = None):
        """Update launch parameters for the current frame."""
        self._update_instance_transform_buffers()

        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix()
        view_inv = np.linalg.inv(view)
        proj_inv = np.linalg.inv(proj)

        dt = self._get_launch_params_dtype()
        params_size = dt.itemsize

        if self._launch_params_np is None:
            self._launch_params_np = np.zeros(1, dtype=dt)
        p = self._launch_params_np[0]

        p["tlas"] = self._scene.tlas_handle
        p["view"] = view.reshape(-1)
        p["proj"] = proj.reshape(-1)
        p["prevView"] = self._prev_view.reshape(-1)
        p["prevProj"] = self._prev_proj.reshape(-1)
        p["prevMVP"] = self._prev_mvp.reshape(-1)
        p["viewInv"] = view_inv.reshape(-1)
        p["projInv"] = proj_inv.reshape(-1)
        frame_index_value = self.sample_index if frame_index_override is None else int(frame_index_override)
        if self.use_halton_jitter:
            jitter_x = self._halton(frame_index_value, 2) - 0.5
            jitter_y = self._halton(frame_index_value, 3) - 0.5
            p["jitter"] = (np.float32(jitter_x), np.float32(jitter_y))
            self._last_jitter = (float(jitter_x), float(jitter_y))
        else:
            p["jitter"] = (0.0, 0.0)
            self._last_jitter = (0.0, 0.0)
        p["envIntensity"] = (1.0, 1.0, 1.0, 1.0)
        p["envRotation"] = np.float32(0.0)
        flags = 2
        if self._env_map is None:
            flags |= 1
        p["flags"] = np.uint32(flags)
        sky_dir = np.asarray(self.sky_sun_direction, dtype=np.float32)
        sky_dir_norm = np.linalg.norm(sky_dir)
        if sky_dir_norm > 1.0e-8:
            sky_dir = sky_dir / sky_dir_norm
        p["skyInfo"]["rgbUnitConversion"] = np.asarray(self.sky_rgb_unit_conversion, dtype=np.float32)
        p["skyInfo"]["multiplier"] = np.float32(self.sky_multiplier)
        p["skyInfo"]["haze"] = np.float32(self.sky_haze)
        p["skyInfo"]["redblueshift"] = np.float32(self.sky_redblueshift)
        p["skyInfo"]["saturation"] = np.float32(self.sky_saturation)
        p["skyInfo"]["horizonHeight"] = np.float32(self.sky_horizon_height)
        p["skyInfo"]["groundColor"] = np.asarray(self.sky_ground_color, dtype=np.float32)
        p["skyInfo"]["horizonBlur"] = np.float32(self.sky_horizon_blur)
        p["skyInfo"]["nightColor"] = np.asarray(self.sky_night_color, dtype=np.float32)
        p["skyInfo"]["sunDiskIntensity"] = np.float32(self.sky_sun_disk_intensity)
        p["skyInfo"]["sunDirection"] = sky_dir
        p["skyInfo"]["sunDiskScale"] = np.float32(self.sky_sun_disk_scale)
        p["skyInfo"]["sunGlowIntensity"] = np.float32(self.sky_sun_glow_intensity)
        p["skyInfo"]["yIsUp"] = np.int32(self.sky_y_is_up)

        _a = self._addr_u64
        p["materialAddress"] = _a(self._scene.materials.gpu_address)
        p["compactMaterialAddress"] = _a(self._scene.compact_materials_address)
        p["instanceMaterialIdAddress"] = _a(self._scene.instance_material_ids_address)
        p["instanceRenderPrimIdAddress"] = _a(self._scene.instance_render_prim_ids_address)
        p["renderPrimitiveAddress"] = _a(self._scene.render_primitives_address)
        p["instanceTransformsAddress"] = np.uint64(
            0 if self._instance_transforms_buffer is None else self._instance_transforms_buffer.ptr
        )
        p["prevInstanceTransformsAddress"] = np.uint64(
            0 if self._prev_instance_transforms_buffer is None else self._prev_instance_transforms_buffer.ptr
        )
        p["materialCount"] = np.uint32(self._scene.materials.count)
        p["instanceCount"] = np.uint32(self._scene.instance_count)
        p["renderPrimCount"] = np.uint32(self._scene.mesh_count)
        p["frameIndex"] = np.uint32(frame_index_value)
        p["maxBounces"] = np.uint32(self.max_bounces)
        p["directLightSamples"] = np.uint32(self.direct_light_samples)
        p["textureDescAddress"] = _a(self._scene.texture_descs_address)
        p["textureDataAddress"] = _a(self._scene.texture_data_address)
        p["textureCount"] = np.uint32(self._scene.texture_count)
        if self._env_map is not None:
            p["envMapAddress"] = _a(self._env_map.env_map_address)
            p["envMapWidth"] = np.uint32(self._env_map.width)
            p["envMapHeight"] = np.uint32(self._env_map.height)
            p["envMapFormat"] = np.uint32(0)
            p["envAccelAddress"] = _a(self._env_map.env_accel_address)
            p["envMapIntegral"] = np.float32(self._env_map.integral)
            p["envMapAverage"] = np.float32(self._env_map.average)
        else:
            p["envMapAddress"] = np.uint64(0)
            p["envMapWidth"] = np.uint32(0)
            p["envMapHeight"] = np.uint32(0)
            p["envMapFormat"] = np.uint32(0)
            p["envAccelAddress"] = np.uint64(0)
            p["envMapIntegral"] = np.float32(0.0)
            p["envMapAverage"] = np.float32(0.0)

        p["colorOutput"] = self._color_buffer.ptr
        p["normalRoughnessOutput"] = self._normal_roughness_buffer.ptr
        p["motionVectorOutput"] = self._motion_buffer.ptr
        p["depthOutput"] = self._depth_buffer.ptr
        p["diffuseAlbedoOutput"] = self._diffuse_buffer.ptr
        p["specularAlbedoOutput"] = self._specular_buffer.ptr
        p["specHitDistOutput"] = self._spec_hit_dist_buffer.ptr
        p["outputMode"] = self.OUTPUT_FINAL

        # Reuse GPU buffer when size matches; only reallocate on resize.
        params_bytes = self._launch_params_np.view(np.uint8).reshape(-1)
        if self._launch_params_buffer is not None and self._launch_params_size == params_size:
            self._launch_params_buffer.assign(params_bytes)
        else:
            self._launch_params_buffer = wp.array(params_bytes, dtype=wp.uint8, device="cuda")
            self._launch_params_size = params_size

    def _update_temporal_state(self, current_view: np.ndarray, current_proj: np.ndarray, use_external_accum: bool) -> bool:
        """Update accumulation state and return whether temporal history resets."""
        if self._dlss_enabled:
            return False

        reset_temporal = (
            self.output_mode != self._last_output_mode
            or (not np.allclose(current_view, self._last_accum_view))
            or (not np.allclose(current_proj, self._last_accum_proj))
        )

        if use_external_accum:
            if reset_temporal:
                wp.launch(
                    _reset_accum_buffer,
                    dim=(self._render_width, self._render_height),
                    inputs=[self._accum_buffer],
                    device="cuda",
                )
                self.frame_index = 0
            return bool(reset_temporal)

        # No persistent external accumulation for this mode.
        wp.launch(
            _reset_accum_buffer,
            dim=(self._render_width, self._render_height),
            inputs=[self._accum_buffer],
            device="cuda",
        )
        return bool(reset_temporal)

    def _launch_samples(self, samples_this_frame: int, use_external_accum: bool):
        """Launch OptiX path tracing and optional external accumulation kernels."""
        for s in range(samples_this_frame):
            launch_frame_index = self.sample_index + s
            self._update_launch_params(frame_index_override=launch_frame_index)

            self._optix.launch(
                self._pipeline,
                0,  # stream
                self._launch_params_buffer.ptr,
                self._launch_params_buffer.shape[0],
                self._sbt,
                self._render_width,
                self._render_height,
                1,  # depth
            )

            if not self._dlss_enabled:
                accum_sample_index = int(self.frame_index if use_external_accum else s)
                wp.launch(
                    _accumulate_sample,
                    dim=(self._render_width, self._render_height),
                    inputs=[self._color_buffer, self._accum_buffer, accum_sample_index],
                    device="cuda",
                )

                if use_external_accum:
                    self.frame_index += 1

    def _process_debug_output(self):
        self._tonemapper.resize(self.width, self.height)
        self._tonemapper.process_debug(
            self.output_mode,
            self._color_buffer,
            self._depth_buffer,
            self._motion_buffer,
            self._normal_roughness_buffer,
            self._diffuse_buffer,
            self._specular_buffer,
            self._spec_hit_dist_buffer,
            self._render_width,
            self._render_height,
        )

    def _process_final_output(self, source_buffer, *, resize_to_render: bool):
        if resize_to_render:
            self._tonemapper.resize(self._render_width, self._render_height)
        self._tonemapper.process(source_buffer)

    def _process_output(self, source_buffer, *, resize_final_to_render: bool):
        if self.output_mode == self.OUTPUT_FINAL:
            self._process_final_output(source_buffer, resize_to_render=resize_final_to_render)
            return
        self._process_debug_output()

    def render(self):
        """Render a frame."""
        if self._pipeline is None:
            logger.error("Pipeline not built. Call build() first.")
            return

        current_view = self.camera.get_view_matrix().copy()
        current_proj = self.camera.get_projection_matrix().copy()
        use_external_accum = self.accumulate_samples and not self._dlss_enabled
        samples_this_frame = 1 if self._dlss_enabled else self.samples_per_frame
        reset_temporal = self._update_temporal_state(current_view, current_proj, use_external_accum)
        self._launch_samples(samples_this_frame, use_external_accum)

        # Snapshot current instance transforms -> previous for next frame's
        # rigid-body motion vectors.  This is a GPU-side copy so it is cheap.
        self._snapshot_instance_transforms()

        # Keep previous matrices for next frame's motion-vector calculation.
        self._prev_view = current_view.copy()
        self._prev_proj = current_proj.copy()
        self._prev_mvp = (current_view @ current_proj).astype(np.float32)
        self._last_accum_view = current_view.copy()
        self._last_accum_proj = current_proj.copy()
        self._last_output_mode = self.output_mode

        if self._dlss_enabled:
            # Single sync: ensure OptiX launch + Warp kernel writes are visible
            # before copying into DLSS texture resources and running DLSS.
            wp.synchronize_device("cuda")
            self._copy_linear_to_dlss_textures()
            if self._run_dlss_rr(reset_temporal):
                # Single sync: ensure DLSS writes are complete before reading output.
                wp.synchronize_device("cuda")
                self._copy_dlss_output_to_color()
                if self._dlss_output_buffer is not None:
                    self._process_output(self._dlss_output_buffer, resize_final_to_render=False)
                else:
                    self._process_output(self._color_buffer, resize_final_to_render=False)
            else:
                self._process_output(self._color_buffer, resize_final_to_render=True)
        else:
            self._process_output(self._accum_buffer, resize_final_to_render=True)
        self.sample_index += samples_this_frame

    def get_output(self) -> np.ndarray:
        """Get the current output as a numpy array."""
        wp.synchronize_device("cuda")
        return self._tonemapper.get_numpy()

    def resize(self, width: int, height: int):
        """Resize the render buffers."""
        if width != self.width or height != self.height:
            self.width = width
            self.height = height
            self.camera.set_aspect_ratio(width, height)
            self._sync_prev_camera_matrices_to_current()
            self._set_render_resolution(width, height)
            self._tonemapper.resize(width, height)
            self._init_dlss_rr()
            self.frame_index = 0

    def __del__(self):
        self._destroy_dlss_rr()


def main():
    """Run the path tracing viewer."""
    logger.info("%s", "=" * 60)
    logger.info("OptiX Path Tracing Viewer")
    logger.info("%s", "=" * 60)

    viewer = PathTracingViewer(width=800, height=600)

    if not viewer.build():
        logger.error("Failed to build viewer.")
        return 1

    # Render a few frames
    logger.info("Rendering frames.")
    for i in range(10):
        viewer.render()
        logger.info("Frame %d", i + 1)

    # Get final output
    output = viewer.get_output()
    logger.info("Output shape: %s", output.shape)
    logger.info("Output range: [%.3f, %.3f]", float(output.min()), float(output.max()))

    # Save to file if possible
    try:
        from PIL import Image

        # Convert to uint8
        img_data = (output[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(img_data)
        img.save("pathtracing_output.png")
        logger.info("Saved output to pathtracing_output.png")
    except ImportError:
        logger.info("Pillow not installed; skipping image save.")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
