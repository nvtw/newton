# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""OptiX header discovery and PTX compilation helpers."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

import warp as wp
import warp._src.build as wp_build


def get_optix_include_dir(optix_module=None) -> str | None:
    """Get the OptiX include directory used for downstream compilation."""

    def _has_optix_device_header(path: str) -> bool:
        return os.path.isfile(os.path.join(path, "optix_device.h"))

    def _parse_version_from_path(path: str) -> tuple[int, ...]:
        # Windows example: ".../OptiX SDK 9.0.0/include"
        m = re.search(r"OptiX SDK (\d+(?:\.\d+)*)", path)
        if not m:
            return (0,)
        return tuple(int(p) for p in m.group(1).split("."))

    # Preferred path: query the include directory from the installed wrapper.
    if optix_module is not None and hasattr(optix_module, "get_optix_include_dir"):
        try:
            optix_dir = optix_module.get_optix_include_dir()
        except Exception:
            optix_dir = None
        if optix_dir and os.path.isdir(optix_dir) and _has_optix_device_header(optix_dir):
            print(f"[Viewer] OptiX header found: {os.path.join(optix_dir, 'optix_device.h')}")
            return optix_dir
        print("BIG")
        print(
            "[Viewer] optix.get_optix_include_dir() did not provide a directory "
            "with optix_device.h; falling back to SDK discovery."
        )

    discovered: list[str] = []
    sdk_root = Path("C:/ProgramData/NVIDIA Corporation")
    if sdk_root.is_dir():
        for include_dir in sdk_root.glob("OptiX SDK */include"):
            if include_dir.is_dir() and _has_optix_device_header(str(include_dir)):
                discovered.append(str(include_dir))
    discovered.sort(key=_parse_version_from_path, reverse=True)

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
    return path.read_text(encoding="utf-8")


def _load_header_with_resolved_includes(path: Path, include_dirs: list[Path]) -> str:
    """Load a header and recursively inline local includes."""
    include_re = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')
    visited: set[Path] = set()

    def _resolve_include(include_name: str, current_dir: Path) -> Path | None:
        candidates = [current_dir / include_name, *[inc_dir / include_name for inc_dir in include_dirs]]
        for candidate in candidates:
            resolved_candidate = candidate.resolve()
            if resolved_candidate.is_file():
                return resolved_candidate
        return None

    def _inline(file_path: Path) -> str:
        resolved_file_path = file_path.resolve()
        if resolved_file_path in visited:
            return ""
        visited.add(resolved_file_path)

        out_lines = []
        for line in resolved_file_path.read_text(encoding="utf-8").splitlines():
            match = include_re.match(line)
            if not match:
                out_lines.append(line)
                continue

            include_name = match.group(1)
            resolved = _resolve_include(include_name, resolved_file_path.parent)
            if resolved is None:
                out_lines.append(line)
                continue

            out_lines.append(f"// === begin include: {include_name} ===")
            out_lines.append(_inline(resolved))
            out_lines.append(f"// === end include: {include_name} ===")

        return "\n".join(out_lines)

    return _inline(path)


def _get_ptx_cache_root() -> Path:
    """Return OS-appropriate cache directory for OptiX PTX artifacts."""
    try:
        from warp.thirdparty import appdirs  # noqa: PLC0415

        cache_dir = appdirs.user_cache_dir(appname="newton", appauthor="Newton", version="optix-ptx")
        if cache_dir:
            return Path(cache_dir)
    except Exception:
        pass

    return Path.home() / ".cache" / "newton" / "optix" / "ptx"


def build_ptx(optix_include_dir: str, headers_dir: Path) -> bytes:
    """Build PTX from path tracing headers."""

    def _compile_cuda_source_to_ptx(cuda_source: str, module_tag: str, device: str = "cuda") -> bytes:
        device_obj = wp.get_device(device)
        if not device_obj.is_cuda:
            raise RuntimeError(f"PTX can only be generated for CUDA devices, got '{device_obj}'")

        digest = hashlib.sha256(cuda_source.encode("utf-8")).hexdigest()[:16]
        local_cache_root = _get_ptx_cache_root()
        local_cache_root.mkdir(parents=True, exist_ok=True)
        module_dir = os.path.join(str(local_cache_root), f"wp_pathtracing_{module_tag}_{digest}")
        os.makedirs(module_dir, exist_ok=True)

        cu_path = os.path.join(module_dir, f"wp_pathtracing_{module_tag}_{digest}.cu")
        ptx_path = os.path.join(module_dir, f"wp_pathtracing_{module_tag}_{digest}.ptx")

        with open(cu_path, "w", encoding="utf-8") as f:
            f.write(cuda_source)

        old_use_pch = wp.config.use_precompiled_headers
        try:
            wp.config.use_precompiled_headers = False
            try:
                wp_build.build_cuda(cu_path, arch=device_obj.arch, output_path=ptx_path)
            except Exception:
                shutil.rmtree(local_cache_root, ignore_errors=True)
                local_cache_root.mkdir(parents=True, exist_ok=True)
                os.makedirs(module_dir, exist_ok=True)
                wp_build.build_cuda(cu_path, arch=device_obj.arch, output_path=ptx_path)
        finally:
            wp.config.use_precompiled_headers = old_use_pch
        with open(ptx_path, "rb") as f:
            return f.read()

    func_common_path = headers_dir / "func_common.h"
    func_common_content = _load_header(func_common_path) if func_common_path.exists() else ""

    constants_path = headers_dir / "constants.h"
    constants_content = _load_header(constants_path) if constants_path.exists() else ""

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

    print("[PTX] Compiling path tracing kernels...")
    ptx = _compile_cuda_source_to_ptx(cuda_source, module_tag="kernels", device="cuda")
    print(f"[PTX] Compiled {len(ptx)} bytes")
    return ptx
