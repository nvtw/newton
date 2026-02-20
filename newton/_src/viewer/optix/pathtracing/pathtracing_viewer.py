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
A Python/OptiX port of the C# Vulkan path tracer.

This viewer renders a scene using OptiX ray tracing with PBR materials,
displaying raw buffers (radiance, normals, depth, etc.) for debugging.
"""

import hashlib
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import warp as wp
import warp._src.build as wp_build

# Initialize warp
wp.init()

# Import local modules
from ..optix_context import _create_optix_context
from ..hit_kernels import HitKernel
from ..sbt_helpers import SbtKernelManager
from .camera import Camera
from .environment_map import EnvironmentMap
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


def _get_optix_include_dir():
    """Get the OptiX SDK include directory."""

    def _has_optix_device_header(path: str) -> bool:
        return os.path.isfile(os.path.join(path, "optix_device.h"))

    def _parse_version_from_path(path: str) -> tuple[int, ...]:
        # Windows example: ".../OptiX SDK 9.0.0/include"
        m = re.search(r"OptiX SDK (\d+(?:\.\d+)*)", path)
        if not m:
            return (0,)
        return tuple(int(p) for p in m.group(1).split("."))

    # Check environment variable first.
    optix_dir = os.environ.get("OPTIX_SDK_INCLUDE_DIR")
    if optix_dir and os.path.isdir(optix_dir) and _has_optix_device_header(optix_dir):
        return optix_dir

    # Prefer highest installed OptiX SDK version on Windows.
    discovered: list[str] = []
    sdk_root = Path("C:/ProgramData/NVIDIA Corporation")
    if sdk_root.is_dir():
        for p in sdk_root.glob("OptiX SDK */include"):
            if p.is_dir() and _has_optix_device_header(str(p)):
                discovered.append(str(p))
    discovered.sort(key=_parse_version_from_path, reverse=True)

    # Common fallback locations (keep 9.0.0 first).
    candidates = [
        *discovered,
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/include",
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0/include",
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0/include",
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0/include",
        "/opt/optix/include",
        os.path.expanduser("~/optix/include"),
    ]

    for path in candidates:
        if os.path.isdir(path) and _has_optix_device_header(path):
            return path

    return None


def _load_header(path: Path) -> str:
    """Load a header file as text."""
    return path.read_text(encoding="utf-8")


def _load_header_with_resolved_includes(path: Path, include_dirs: list[Path]) -> str:
    """
    Load a header and recursively inline local includes.

    Supports ``#include "relative/path.h"`` and resolves first relative to the
    including file, then against ``include_dirs``.
    """
    include_re = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')
    visited: set[Path] = set()

    def _resolve_include(include_name: str, current_dir: Path) -> Path | None:
        candidates = [current_dir / include_name, *[inc_dir / include_name for inc_dir in include_dirs]]
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.is_file():
                return candidate
        return None

    def _inline(file_path: Path) -> str:
        file_path = file_path.resolve()
        if file_path in visited:
            return ""
        visited.add(file_path)

        out_lines = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            match = include_re.match(line)
            if not match:
                out_lines.append(line)
                continue

            include_name = match.group(1)
            resolved = _resolve_include(include_name, file_path.parent)
            if resolved is None:
                # Keep unknown include lines untouched (e.g., non-local includes).
                out_lines.append(line)
                continue

            out_lines.append(f"// === begin include: {include_name} ===")
            out_lines.append(_inline(resolved))
            out_lines.append(f"// === end include: {include_name} ===")

        return "\n".join(out_lines)

    return _inline(path)


def _round_up(val: int, mult_of: int) -> int:
    return val if val % mult_of == 0 else val + mult_of - val % mult_of


def _get_aligned_itemsize(formats: list[str], alignment: int) -> int:
    names = [f"x{i}" for i in range(len(formats))]
    temp_dtype = np.dtype({"names": names, "formats": formats, "align": True})
    return _round_up(temp_dtype.itemsize, alignment)


def _get_ptx_cache_root() -> Path:
    """Return OS-appropriate cache directory for OptiX PTX artifacts.

    Resolution order:
      1) user cache dir via Warp's bundled appdirs
      2) fallback to ~/.cache/newton/optix/ptx
    """
    try:
        from warp.thirdparty import appdirs  # noqa: PLC0415

        cache_dir = appdirs.user_cache_dir(appname="newton", appauthor="Newton", version="optix-ptx")
        if cache_dir:
            return Path(cache_dir)
    except Exception:
        pass

    return Path.home() / ".cache" / "newton" / "optix" / "ptx"


def _build_ptx(optix_include_dir: str, headers_dir: Path) -> bytes:
    """
    Build PTX from the path tracing headers.

    Args:
        optix_include_dir: Path to OptiX SDK include directory
        headers_dir: Path to the `cpp/` headers directory

    Returns:
        PTX string
    """

    def _compile_cuda_source_to_ptx(cuda_source: str, module_tag: str, device: str = "cuda") -> bytes:
        """Compile in-memory CUDA source into PTX bytes."""
        device_obj = wp.get_device(device)
        if not device_obj.is_cuda:
            raise RuntimeError(f"PTX can only be generated for CUDA devices, got '{device_obj}'")

        digest = hashlib.sha256(cuda_source.encode("utf-8")).hexdigest()[:16]
        local_cache_root = _get_ptx_cache_root()
        # Keep cache by default for fast startup. If build fails (stale/corrupt
        # cache), clear and retry once.
        local_cache_root.mkdir(parents=True, exist_ok=True)
        module_dir = os.path.join(str(local_cache_root), f"wp_pathtracing_{module_tag}_{digest}")
        os.makedirs(module_dir, exist_ok=True)

        cu_path = os.path.join(module_dir, f"wp_pathtracing_{module_tag}_{digest}.cu")
        ptx_path = os.path.join(module_dir, f"wp_pathtracing_{module_tag}_{digest}.ptx")

        with open(cu_path, "w", encoding="utf-8") as f:
            f.write(cuda_source)

        # Avoid stale/incompatible NVRTC PCH reuse across sessions. We only
        # disable PCH for this local PTX build path.
        old_use_pch = wp.config.use_precompiled_headers
        try:
            wp.config.use_precompiled_headers = False
            try:
                wp_build.build_cuda(cu_path, arch=device_obj.arch, output_path=ptx_path)
            except Exception:
                # Automatic recovery path for stale local PTX cache issues.
                shutil.rmtree(local_cache_root, ignore_errors=True)
                local_cache_root.mkdir(parents=True, exist_ok=True)
                os.makedirs(module_dir, exist_ok=True)
                wp_build.build_cuda(cu_path, arch=device_obj.arch, output_path=ptx_path)
        finally:
            wp.config.use_precompiled_headers = old_use_pch
        with open(ptx_path, "rb") as f:
            return f.read()

    # Load func_common.h first (defines float4x4, float3, etc.)
    func_common_path = headers_dir / "func_common.h"
    func_common_content = _load_header(func_common_path) if func_common_path.exists() else ""

    # Load constants.h
    constants_path = headers_dir / "constants.h"
    constants_content = _load_header(constants_path) if constants_path.exists() else ""

    # Runtime launch params for the currently enabled OptiX programs.
    # Keep this struct in strict sync with _update_launch_params().
    runtime_launch_params = """
// Display/output modes (mirrors pathtracing launch_params naming)
#define OUTPUT_MODE_FINAL       0
#define OUTPUT_MODE_RADIANCE    1
#define OUTPUT_MODE_DEPTH       2
#define OUTPUT_MODE_MOTION      3
#define OUTPUT_MODE_NORMAL      4
#define OUTPUT_MODE_ROUGHNESS   5
#define OUTPUT_MODE_DIFFUSE     6
#define OUTPUT_MODE_SPECULAR    7
#define OUTPUT_MODE_SPEC_HITDIST 8

struct RuntimeFrameInfo
{
    float view[16];
    float proj[16];
    float prevView[16];
    float prevProj[16];
    float prevMVP[16];
    float viewInv[16];
    float projInv[16];
    float jitter[2];
    float envIntensity[4];
    float envRotation;
    unsigned int flags;
};

struct RuntimeSkyInfo
{
    float rgbUnitConversion[3];
    float multiplier;
    float haze;
    float redblueshift;
    float saturation;
    float horizonHeight;
    float groundColor[3];
    float horizonBlur;
    float nightColor[3];
    float sunDiskIntensity;
    float sunDirection[3];
    float sunDiskScale;
    float sunGlowIntensity;
    int   yIsUp;
};

struct LaunchParams
{
    OptixTraversableHandle tlas;
    RuntimeFrameInfo       frameInfo;
    RuntimeSkyInfo         skyInfo;
    unsigned long long     materialAddress;
    unsigned long long     compactMaterialAddress;
    unsigned long long     instanceMaterialIdAddress;
    unsigned long long     instanceRenderPrimIdAddress;
    unsigned long long     renderPrimitiveAddress;
    unsigned long long     instanceTransformsAddress;
    unsigned long long     prevInstanceTransformsAddress;
    unsigned int           materialCount;
    unsigned int           instanceCount;
    unsigned int           renderPrimCount;
    unsigned int           frameIndex;
    unsigned int           maxBounces;
    unsigned int           directLightSamples;
    unsigned long long     textureDescAddress;
    unsigned long long     textureDataAddress;
    unsigned int           textureCount;
    unsigned int           _pad0;
    unsigned long long     envMapAddress;
    unsigned int           envMapWidth;
    unsigned int           envMapHeight;
    unsigned int           envMapFormat;
    unsigned int           _pad1;
    unsigned long long     envAccelAddress;
    float                  envMapIntegral;
    float                  envMapAverage;
    unsigned long long     colorOutput;
    unsigned long long     normalRoughnessOutput;
    unsigned long long     motionVectorOutput;
    unsigned long long     depthOutput;
    unsigned long long     diffuseAlbedoOutput;
    unsigned long long     specularAlbedoOutput;
    unsigned long long     specHitDistOutput;
    int                    outputMode;
    int                    _pad2;
};

extern "C" __constant__ LaunchParams params;
"""

    # Entry header for runtime programs. Local includes are resolved recursively
    # to avoid manual copy/paste concatenation of helper headers.
    entry_header = headers_dir / "optix_programs.h"
    if not entry_header.exists():
        raise RuntimeError(f"Entry OptiX header not found: {entry_header}")
    rt_content = _load_header_with_resolved_includes(entry_header, [headers_dir])

    def _load_optix_device_header_text() -> str:
        root = os.path.normpath(optix_include_dir)
        optix_device_h = os.path.join(root, "optix_device.h")
        if not os.path.isfile(optix_device_h):
            raise RuntimeError(f"OptiX device header not found: {optix_device_h}")

        include_pattern = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$', re.MULTILINE)
        visited = set()

        def inline_local_includes(file_path: str) -> str:
            norm_path = os.path.normpath(file_path)
            if norm_path in visited:
                return ""
            visited.add(norm_path)

            with open(norm_path, encoding="utf-8") as f:
                src = f.read()

            def replace_include(match):
                rel = match.group(1)
                include_path = os.path.normpath(os.path.join(os.path.dirname(norm_path), rel))
                if not os.path.isfile(include_path):
                    include_path = os.path.normpath(os.path.join(root, rel))
                if include_path.startswith(root) and os.path.isfile(include_path):
                    return inline_local_includes(include_path)
                return match.group(0)

            return include_pattern.sub(replace_include, src)

        return "#define __OPTIX_INCLUDE_INTERNAL_HEADERS__\n" + inline_local_includes(optix_device_h)

    # Build complete CUDA source
    optix_header_text = _load_optix_device_header_text()
    cuda_source = f"""
// OptiX Path Tracing Kernels
// Auto-generated from header files

{optix_header_text}

// CUDA texture/surface opaque handle types
typedef unsigned long long cudaTextureObject_t;
typedef unsigned long long cudaSurfaceObject_t;

// === constants.h ===
{constants_content}

// === func_common.h ===
{func_common_content}

// === runtime launch params ===
{runtime_launch_params}

// RT headers (resolved from cpp/optix_programs.h includes)
{rt_content}
"""

    # Compile to PTX
    print("[PTX] Compiling path tracing kernels...")
    ptx = _compile_cuda_source_to_ptx(cuda_source, module_tag="kernels", device="cuda")
    print(f"[PTX] Compiled {len(ptx)} bytes")

    return ptx


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

        # Physical sky defaults matched to C# PhysicalSkyParameters.Default.
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
            print(f"[Viewer] Failed to load HDR environment: {hdr_path}")

    def set_environment_color(self, color: tuple[float, float, float]):
        """
        Set a uniform color environment (useful for debugging or simple scenes).

        Args:
            color: RGB color values
        """
        env_map = EnvironmentMap()
        if env_map.load_from_color(color):
            self._env_map = env_map

    def build(self):
        """Build the OptiX pipeline and scene."""
        print("[Viewer] Initializing OptiX...")

        # Import optix
        try:
            import optix
        except ImportError:
            print("ERROR: Could not import optix module.")
            print("Make sure warp.pyoptix is available and OptiX SDK is installed.")
            return False

        self._optix = optix

        # Get OptiX include directory
        optix_include = _get_optix_include_dir()
        if not optix_include:
            print("ERROR: Could not find OptiX SDK include directory.")
            print("Set OPTIX_SDK_INCLUDE_DIR environment variable.")
            return False

        print(f"[Viewer] Using OptiX SDK: {optix_include}")

        # Create OptiX context
        wp_device = wp.get_device("cuda")
        cu_context = wp_device.context.value if hasattr(wp_device.context, "value") else int(wp_device.context)
        self._ctx, self._optix_logger = _create_optix_context(optix, int(cu_context))

        # Build PTX
        headers_dir = Path(__file__).parent / "cpp"
        self._ptx = _build_ptx(optix_include, headers_dir)

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

        print("[Viewer] Build complete!")
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

        Mirrors C# first-frame behavior where prevMVP is set to currentMVP to avoid
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
                print(f"[DLSS] Warning: failed to deinit denoiser: {exc}")
        self._dlss_denoiser = None

        if self._dlss_context is not None:
            try:
                self._dlss_context.deinit()
            except Exception as exc:
                print(f"[DLSS] Warning: failed to deinit context: {exc}")
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
            print("[DLSS] DLSS RR bindings not present in optix module.")
            return

        try:
            context = self._optix.DlssRRContext()
            context.init()
            if not context.isDlssRRAvailable():
                print("[DLSS] DLSS RR not available on this system.")
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
            # Match C#/optix-subd reference:
            # - MVJittered=false while still passing per-frame jitter to denoise()
            # - lowResolutionMotionVectors=true (motion vectors provided at render resolution)
            init_info.mvJittered = False
            init_info.lowResolutionMotionVectors = True
            init_info.isContentHDR = True
            init_info.depthInverted = False
            init_info.autoExposure = False
            init_info.useHWDepth = False

            # Match the C# reference path: ask NGX for the optimal input size
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
                    print(f"[DLSS] Warning: failed to query optimal input size, using half-res fallback: {exc}")

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
            print(
                f"[DLSS] Ray Reconstruction enabled "
                f"(render={self._render_width}x{self._render_height}, output={self.width}x{self.height})."
            )
        except Exception as exc:
            print(f"[DLSS] Failed to initialize DLSS RR: {exc}")
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
            # Match C# MatrixToArray() packing used by MinimalDlssRRViewer:
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
            print(f"[DLSS] Denoise failed, disabling DLSS RR: {exc}")
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
        """Upload current/previous instance transforms for motion vectors."""
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

        # Reuse numpy staging buffers when instance count is stable.
        if count != self._instance_transform_count:
            self._instance_xform_curr_np = np.empty((count, 12), dtype=np.float32)
            self._instance_xform_prev_np = np.empty((count, 12), dtype=np.float32)
            self._instance_transforms_buffer = wp.empty(count * 12, dtype=wp.float32, device="cuda")
            self._prev_instance_transforms_buffer = wp.empty(count * 12, dtype=wp.float32, device="cuda")
            self._instance_transform_count = count

        curr = self._instance_xform_curr_np
        prev = self._instance_xform_prev_np
        for i, inst in enumerate(instances):
            m = np.asarray(inst.transform, dtype=np.float32).reshape(4, 4)
            pm = np.asarray(inst.prev_transform, dtype=np.float32).reshape(4, 4)
            curr[i, 0:4] = m[0, :]
            curr[i, 4:8] = m[1, :]
            curr[i, 8:12] = m[2, :]
            prev[i, 0:4] = pm[0, :]
            prev[i, 4:8] = pm[1, :]
            prev[i, 8:12] = pm[2, :]

        self._instance_transforms_buffer.assign(curr.reshape(-1))
        self._prev_instance_transforms_buffer.assign(prev.reshape(-1))

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
        p["directLightSamples"] = np.uint32(1)
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

    def render(self):
        """Render a frame."""
        if self._pipeline is None:
            print("ERROR: Pipeline not built. Call build() first.")
            return

        current_view = self.camera.get_view_matrix().copy()
        current_proj = self.camera.get_projection_matrix().copy()
        use_external_accum = self.accumulate_samples and not self._dlss_enabled
        samples_this_frame = 1 if self._dlss_enabled else self.samples_per_frame
        reset_temporal = False
        if self._dlss_enabled:
            # Match C# behavior: do NOT reset DLSS on camera motion.
            # History reset is handled only by explicit lifecycle events
            # (first frame/init/resize) via _dlss_reset_history.
            reset_temporal = False
        elif use_external_accum:
            reset_accum = (
                self.output_mode != self._last_output_mode
                or (not np.allclose(current_view, self._last_accum_view))
                or (not np.allclose(current_proj, self._last_accum_proj))
            )
            reset_temporal = bool(reset_accum)
            if reset_accum:
                wp.launch(
                    _reset_accum_buffer,
                    dim=(self._render_width, self._render_height),
                    inputs=[self._accum_buffer],
                    device="cuda",
                )
                self.frame_index = 0
        else:
            reset_temporal = (
                self.output_mode != self._last_output_mode
                or (not np.allclose(current_view, self._last_accum_view))
                or (not np.allclose(current_proj, self._last_accum_proj))
            )
            # No persistent external accumulation for this mode.
            wp.launch(
                _reset_accum_buffer,
                dim=(self._render_width, self._render_height),
                inputs=[self._accum_buffer],
                device="cuda",
            )

        # With DLSS enabled, keep one fresh sample per frame and let DLSS own temporal history.
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
                    if self.output_mode == self.OUTPUT_FINAL:
                        self._tonemapper.process(self._dlss_output_buffer)
                    else:
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
                else:
                    if self.output_mode == self.OUTPUT_FINAL:
                        self._tonemapper.process(self._color_buffer)
                    else:
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
            else:
                if self.output_mode == self.OUTPUT_FINAL:
                    self._tonemapper.resize(self._render_width, self._render_height)
                    self._tonemapper.process(self._color_buffer)
                else:
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
        else:
            if self.output_mode == self.OUTPUT_FINAL:
                self._tonemapper.resize(self._render_width, self._render_height)
                self._tonemapper.process(self._accum_buffer)
            else:
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
    print("=" * 60)
    print("OptiX Path Tracing Viewer")
    print("=" * 60)

    viewer = PathTracingViewer(width=800, height=600)

    if not viewer.build():
        print("Failed to build viewer")
        return 1

    # Render a few frames
    print("\nRendering frames...")
    for i in range(10):
        viewer.render()
        print(f"  Frame {i + 1}")

    # Get final output
    output = viewer.get_output()
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Save to file if possible
    try:
        from PIL import Image

        # Convert to uint8
        img_data = (output[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(img_data)
        img.save("pathtracing_output.png")
        print("\nSaved output to pathtracing_output.png")
    except ImportError:
        print("\nPillow not installed, skipping image save")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
