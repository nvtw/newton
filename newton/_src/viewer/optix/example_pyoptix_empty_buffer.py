# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

###########################################################################
# Warp + PyOptiX cube ray tracing (triangle mesh) + CUDA/GL interop
###########################################################################

import argparse
import ctypes
import hashlib
import os
import re
import time

import numpy as np

import warp as wp
import warp._src.build as wp_build
import warp._src.codegen as wp_codegen

OPTIX_SDK_INCLUDE_DIR = "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/include"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYOPTIX_KERNELS_PATH = os.path.join(SCRIPT_DIR, "example_pyoptix_kernels.h")


def _require_optix():
    try:
        import optix  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "Failed to import 'optix'. Install otk-pyoptix first (https://github.com/NVIDIA/otk-pyoptix)."
        ) from e

    return optix


def _round_up(val: int, mult_of: int) -> int:
    return val if val % mult_of == 0 else val + mult_of - val % mult_of


def _get_aligned_itemsize(formats: list[str], alignment: int) -> int:
    names = [f"x{i}" for i in range(len(formats))]
    temp_dtype = np.dtype({"names": names, "formats": formats, "align": True})
    return _round_up(temp_dtype.itemsize, alignment)


def _numpy_to_warp_device_bytes(numpy_array: np.ndarray) -> wp.array:
    host_bytes = np.ascontiguousarray(numpy_array).view(np.uint8).reshape(-1)
    return wp.array(host_bytes, dtype=wp.uint8, device="cuda")


class _Logger:
    def __init__(self):
        self.num_messages = 0

    def __call__(self, level, tag, message):
        print(f"[{level:>2}][{tag:>12}]: {message}")
        self.num_messages += 1


def _load_optix_device_header_text() -> str:
    root = os.path.normpath(OPTIX_SDK_INCLUDE_DIR)
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


def _load_native_section(section_name: str) -> str:
    if not os.path.isfile(PYOPTIX_KERNELS_PATH):
        raise RuntimeError(f"Missing native source file: {PYOPTIX_KERNELS_PATH}")

    with open(PYOPTIX_KERNELS_PATH, encoding="utf-8") as f:
        src = f.read()

    begin = f"// -- section: {section_name} -- begin"
    end = f"// -- section: {section_name} -- end"
    pattern = re.compile(rf"{re.escape(begin)}\n(.*?){re.escape(end)}", re.DOTALL)
    match = pattern.search(src)
    if not match:
        raise RuntimeError(f"Section '{section_name}' not found in {PYOPTIX_KERNELS_PATH}")
    return match.group(1).rstrip() + "\n"


def _compile_cuda_source_to_ptx(cuda_source: str, module_tag: str, device: str = "cuda") -> bytes:
    device_obj = wp.get_device(device)
    if not device_obj.is_cuda:
        raise RuntimeError(f"PTX can only be generated for CUDA devices, got '{device_obj}'")

    digest = hashlib.sha256(cuda_source.encode("utf-8")).hexdigest()[:16]
    module_dir = os.path.join(wp.config.kernel_cache_dir, f"wp_optix_example_{module_tag}_{digest}")
    os.makedirs(module_dir, exist_ok=True)

    cu_path = os.path.join(module_dir, f"wp_optix_example_{module_tag}_{digest}.cu")
    ptx_path = os.path.join(module_dir, f"wp_optix_example_{module_tag}_{digest}.ptx")

    if not os.path.exists(ptx_path) or not wp.config.cache_kernels:
        with open(cu_path, "w", encoding="utf-8") as f:
            f.write(cuda_source)
        wp_build.build_cuda(cu_path, arch=device_obj.arch, output_path=ptx_path)

    with open(ptx_path, "rb") as f:
        return f.read()


def _build_optix_ptx_from_warp(trace_mode: str, device: str = "cuda") -> bytes:
    launch_params = _load_native_section("launch_params")

    if trace_mode == "notrace":
        common_header = _load_native_section("common_header_notrace")
        raygen_and_programs = _load_native_section("trace_mode_notrace")
    elif trace_mode == "trace0":
        common_header = _load_optix_device_header_text()
        raygen_and_programs = _load_native_section("trace_mode_trace0")
    else:
        raise ValueError(f"Unsupported trace mode: {trace_mode}")

    header_text_cuda = common_header + launch_params + raygen_and_programs
    cuda_source = f"{header_text_cuda}\n{wp_codegen.cuda_module_header.format(block_dim=256)}"
    return _compile_cuda_source_to_ptx(cuda_source, module_tag=trace_mode, device=device)


def _create_optix_context(optix, cuda_context):
    logger = _Logger()
    ctx_options = optix.DeviceContextOptions(logCallbackFunction=logger, logCallbackLevel=4)
    if optix.version()[1] >= 2:
        ctx_options.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL
    return optix.deviceContextCreate(cuda_context, ctx_options), logger


def _create_cube_gas(optix, ctx):
    vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [3, 2, 6],
            [3, 6, 7],
            [0, 3, 7],
            [0, 7, 4],
            [1, 5, 6],
            [1, 6, 2],
        ],
        dtype=np.uint32,
    )

    d_vertices = wp.array(vertices, dtype=wp.float32, device="cuda")
    d_indices = wp.array(indices, dtype=wp.uint32, device="cuda")

    accel_options = optix.AccelBuildOptions(
        buildFlags=int(optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS),
        operation=optix.BUILD_OPERATION_BUILD,
    )

    tri = optix.BuildInputTriangleArray()
    tri.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
    tri.numVertices = vertices.shape[0]
    tri.vertexStrideInBytes = 12
    tri.vertexBuffers = [d_vertices.ptr]
    tri.indexFormat = optix.INDICES_FORMAT_UNSIGNED_INT3
    tri.numIndexTriplets = indices.shape[0]
    tri.indexStrideInBytes = 12
    tri.indexBuffer = d_indices.ptr
    tri.flags = [optix.GEOMETRY_FLAG_NONE]
    tri.numSbtRecords = 1

    sizes = ctx.accelComputeMemoryUsage([accel_options], [tri])
    d_temp = wp.empty(sizes.tempSizeInBytes, dtype=wp.uint8, device="cuda")
    d_gas = wp.empty(sizes.outputSizeInBytes, dtype=wp.uint8, device="cuda")

    handle = ctx.accelBuild(
        0,
        [accel_options],
        [tri],
        d_temp.ptr,
        sizes.tempSizeInBytes,
        d_gas.ptr,
        sizes.outputSizeInBytes,
        [],
    )
    # Build happens on stream 0; make completion explicit before first trace.
    wp.synchronize_device("cuda")
    keepalive = {"d_vertices": d_vertices, "d_indices": d_indices, "d_temp": d_temp, "d_gas": d_gas}
    return int(handle), keepalive


def _create_optix_pipeline(optix, ctx, ptx: bytes, trace_mode: str):
    trace_enabled = trace_mode != "notrace"
    kwargs = {
        "usesMotionBlur": False,
        "traversableGraphFlags": int(optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS),
        "numPayloadValues": 0 if trace_mode == "notrace" else 1,
        "numAttributeValues": 2,
        "exceptionFlags": int(optix.EXCEPTION_FLAG_NONE),
        "pipelineLaunchParamsVariableName": "params",
    }
    if optix.version()[1] >= 2:
        kwargs["usesPrimitiveTypeFlags"] = int(optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE)
    pipeline_options = optix.PipelineCompileOptions(**kwargs)

    module_options = optix.ModuleCompileOptions(
        maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
        debugLevel=optix.COMPILE_DEBUG_LEVEL_DEFAULT,
    )
    module, log = ctx.moduleCreate(module_options, pipeline_options, ptx)
    if log:
        print(f"Module create log:\n{log}")

    raygen_desc = optix.ProgramGroupDesc()
    raygen_desc.raygenModule = module
    raygen_desc.raygenEntryFunctionName = "__raygen__cube"

    miss_desc = optix.ProgramGroupDesc()
    miss_desc.missModule = module
    miss_desc.missEntryFunctionName = "__miss__cube"

    hit_group = None
    if trace_enabled:
        hit_desc = optix.ProgramGroupDesc()
        hit_desc.hitgroupModuleCH = module
        hit_desc.hitgroupEntryFunctionNameCH = "__closesthit__cube"

    if optix.version()[1] >= 4:
        pg_options = optix.ProgramGroupOptions()
        raygen_group = ctx.programGroupCreate([raygen_desc], pg_options)[0][0]
        miss_group = ctx.programGroupCreate([miss_desc], pg_options)[0][0]
        if trace_enabled:
            hit_group = ctx.programGroupCreate([hit_desc], pg_options)[0][0]
    else:
        raygen_group = ctx.programGroupCreate([raygen_desc])[0][0]
        miss_group = ctx.programGroupCreate([miss_desc])[0][0]
        if trace_enabled:
            hit_group = ctx.programGroupCreate([hit_desc])[0][0]

    link_options = optix.PipelineLinkOptions()
    link_options.maxTraceDepth = 1
    groups = [raygen_group, miss_group] + ([hit_group] if hit_group is not None else [])
    pipeline = ctx.pipelineCreate(pipeline_options, link_options, groups, "")

    # Use a conservative fixed stack setup (common in many OptiX samples)
    # to avoid subtle stack-size underestimation issues with custom codegen.
    pipeline.setStackSize(2 * 1024, 2 * 1024, 2 * 1024, 1)

    header_format = f"{optix.SBT_RECORD_HEADER_SIZE}B"
    record_dtype = np.dtype(
        {
            "names": ["header"],
            "formats": [header_format],
            "itemsize": _get_aligned_itemsize([header_format], optix.SBT_RECORD_ALIGNMENT),
            "align": True,
        }
    )

    h_rg = np.array([(0,)], dtype=record_dtype)
    h_ms = np.array([(0,)], dtype=record_dtype)
    h_hg = np.array([(0,)], dtype=record_dtype)
    optix.sbtRecordPackHeader(raygen_group, h_rg)
    optix.sbtRecordPackHeader(miss_group, h_ms)
    if hit_group is not None:
        optix.sbtRecordPackHeader(hit_group, h_hg)

    d_rg = _numpy_to_warp_device_bytes(h_rg)
    d_ms = _numpy_to_warp_device_bytes(h_ms)
    d_hg = _numpy_to_warp_device_bytes(h_hg)

    sbt = optix.ShaderBindingTable()
    sbt.raygenRecord = d_rg.ptr
    sbt.missRecordBase = d_ms.ptr
    sbt.missRecordStrideInBytes = h_ms.dtype.itemsize
    sbt.missRecordCount = 1
    if hit_group is not None:
        sbt.hitgroupRecordBase = d_hg.ptr
        sbt.hitgroupRecordStrideInBytes = h_hg.dtype.itemsize
        sbt.hitgroupRecordCount = 1
    else:
        sbt.hitgroupRecordBase = 0
        sbt.hitgroupRecordStrideInBytes = 0
        sbt.hitgroupRecordCount = 0

    keepalive = {
        "module": module,
        "raygen_group": raygen_group,
        "miss_group": miss_group,
        "hit_group": hit_group,
        "d_rg": d_rg,
        "d_ms": d_ms,
        "d_hg": d_hg,
    }
    return pipeline, sbt, keepalive


class _RealtimeViewer:
    def __init__(
        self, optix, pipeline, sbt, gas_handle: int, width: int, height: int, fps: int = 60, max_frames: int = 0
    ):
        import pyglet  # noqa: PLC0415
        from pyglet import gl  # noqa: PLC0415

        self.optix = optix
        self.pipeline = pipeline
        self.sbt = sbt
        self.gas_handle = gas_handle
        self.width = width
        self.height = height
        self.fps = fps
        self.max_frames = max_frames
        self.frame_count = 0
        self.pyglet = pyglet
        self.gl = gl

        self.window = pyglet.window.Window(width=width, height=height, caption="Warp + PyOptiX Cube", vsync=False)
        self.window.push_handlers(on_draw=self.on_draw, on_close=self.on_close)

        self.texture = pyglet.image.Texture.create(width=width, height=height, rectangle=False)
        self.texture.min_filter = gl.GL_NEAREST
        self.texture.mag_filter = gl.GL_NEAREST
        self.sprite = pyglet.sprite.Sprite(self.texture, x=0, y=0)

        self.pbo = gl.GLuint()
        gl.glGenBuffers(1, self.pbo)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, width * height * 4, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

        self.cuda_gl = wp.RegisteredGLBuffer(
            int(self.pbo.value), device="cuda", flags=wp.RegisteredGLBuffer.WRITE_DISCARD, fallback_to_copy=False
        )

        self.params_dtype = np.dtype(
            {
                "names": ["image", "width", "height", "time_sec", "trav_handle"],
                "formats": ["u8", "u4", "u4", "f4", "u8"],
                "offsets": [0, 8, 12, 16, 24],
                "itemsize": 32,
            }
        )
        self.params_host = np.zeros(1, dtype=self.params_dtype)
        self.params_device = wp.empty(self.params_dtype.itemsize, dtype=wp.uint8, device="cuda")
        self.start_time = time.perf_counter()

        pyglet.clock.schedule_interval(self.update, 1.0 / float(max(self.fps, 1)))

    def _upload_params(self):
        params_bytes = self.params_host.view(np.uint8).reshape(-1)
        wp.copy(self.params_device, wp.array(params_bytes, dtype=wp.uint8, device="cpu", copy=False))

    def update(self, _dt):
        # Keep Warp's CUDA context current in the GUI callback thread.
        with wp.ScopedDevice("cuda"):
            mapped_pbo = self.cuda_gl.map(dtype=wp.uint32, shape=(self.width * self.height,))
            self.params_host["image"][0] = mapped_pbo.ptr
            self.params_host["width"][0] = self.width
            self.params_host["height"][0] = self.height
            self.params_host["time_sec"][0] = np.float32(time.perf_counter() - self.start_time)
            self.params_host["trav_handle"][0] = int(self.gas_handle)
            self._upload_params()

            self.optix.launch(
                self.pipeline,
                0,
                self.params_device.ptr,
                self.params_dtype.itemsize,
                self.sbt,
                self.width,
                self.height,
                1,
            )
            wp.synchronize_device("cuda")
            self.cuda_gl.unmap()
            self.frame_count += 1
            if self.max_frames > 0 and self.frame_count >= self.max_frames:
                self.pyglet.app.exit()

    def on_draw(self):
        gl = self.gl
        self.window.clear()
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        gl.glBindTexture(self.texture.target, self.texture.id)
        gl.glTexSubImage2D(
            self.texture.target,
            0,
            0,
            0,
            self.width,
            self.height,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        gl.glBindTexture(self.texture.target, 0)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        self.sprite.draw()

    def on_close(self):
        self.gl.glDeleteBuffers(1, self.pbo)
        self.window.close()

    def run(self):
        self.pyglet.app.run()


def _run_headless(optix, pipeline, sbt, gas_handle: int, width: int, height: int, frames: int):
    params_dtype = np.dtype(
        {
            "names": ["image", "width", "height", "time_sec", "trav_handle"],
            "formats": ["u8", "u4", "u4", "f4", "u8"],
            "offsets": [0, 8, 12, 16, 24],
            "itemsize": 32,
        }
    )
    params_host = np.zeros(1, dtype=params_dtype)
    params_device = wp.empty(params_dtype.itemsize, dtype=wp.uint8, device="cuda")
    image = wp.empty(width * height, dtype=wp.uint32, device="cuda")
    start_time = time.perf_counter()

    for _ in range(max(frames, 1)):
        params_host["image"][0] = image.ptr
        params_host["width"][0] = width
        params_host["height"][0] = height
        params_host["time_sec"][0] = np.float32(time.perf_counter() - start_time)
        params_host["trav_handle"][0] = int(gas_handle)
        params_bytes = params_host.view(np.uint8).reshape(-1)
        wp.copy(params_device, wp.array(params_bytes, dtype=wp.uint8, device="cpu", copy=False))

        optix.launch(
            pipeline,
            0,
            params_device.ptr,
            params_dtype.itemsize,
            sbt,
            width,
            height,
            1,
        )
        wp.synchronize_device("cuda")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--trace-mode", choices=["notrace", "trace0"], default="trace0")
    parser.add_argument("--max-frames", type=int, default=0, help="Auto-exit after N frames (0 = run forever)")
    parser.add_argument("--headless", action="store_true", help="Run OptiX launches without OpenGL interop")
    args = parser.parse_args()

    optix = _require_optix()
    wp.init()

    with wp.ScopedDevice(args.device):
        if not wp.get_device().is_cuda:
            raise RuntimeError("This example requires CUDA.")

        print(f"PyOptiX: OptiX SDK {optix.version()}")

        ptx = _build_optix_ptx_from_warp(trace_mode=args.trace_mode, device=args.device)
        wp_device = wp.get_device(args.device)
        cu_context = wp_device.context
        if hasattr(cu_context, "value"):
            cu_context = cu_context.value
        ctx, logger = _create_optix_context(optix, int(cu_context))
        if args.trace_mode == "notrace":
            gas_handle = 0
            gas_keepalive = {}
        else:
            gas_handle, gas_keepalive = _create_cube_gas(optix, ctx)
        pipeline, sbt, pipe_keepalive = _create_optix_pipeline(optix, ctx, ptx, trace_mode=args.trace_mode)
        print(f"OptiX log messages: {logger.num_messages}")

        if args.headless:
            frames = args.max_frames if args.max_frames > 0 else 1
            _run_headless(optix, pipeline, sbt, gas_handle, args.width, args.height, frames)
        else:
            viewer = _RealtimeViewer(
                optix, pipeline, sbt, gas_handle, args.width, args.height, fps=args.fps, max_frames=args.max_frames
            )
            viewer._keepalive = {"pipe": pipe_keepalive, "gas": gas_keepalive}
            viewer.run()


if __name__ == "__main__":
    main()
