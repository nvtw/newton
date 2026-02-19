#!/usr/bin/env python3
"""
Test compilation of compute shader headers (tonemap.h and debug_visualize.h).
"""

import hashlib
import os
import sys

# Add warp to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

import warp as wp
import warp._src.build as wp_build

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def compile_cuda_source_to_ptx(cuda_source: str, module_tag: str, device: str = "cuda") -> bytes:
    """Compile CUDA source to PTX."""
    device_obj = wp.get_device(device)
    if not device_obj.is_cuda:
        raise RuntimeError(f"PTX can only be generated for CUDA devices, got '{device_obj}'")

    digest = hashlib.sha256(cuda_source.encode("utf-8")).hexdigest()[:16]
    module_dir = os.path.join(wp.config.kernel_cache_dir, f"wp_compute_{module_tag}_{digest}")
    os.makedirs(module_dir, exist_ok=True)

    cu_path = os.path.join(module_dir, f"wp_compute_{module_tag}_{digest}.cu")
    ptx_path = os.path.join(module_dir, f"wp_compute_{module_tag}_{digest}.ptx")

    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(cuda_source)

    try:
        wp_build.build_cuda(cu_path, arch=device_obj.arch, output_path=ptx_path)
        with open(ptx_path, "rb") as f:
            return f.read()
    except Exception:
        print(f"Compilation failed. Source saved to: {cu_path}")
        raise


def read_header(filename: str) -> str:
    """Read a header file."""
    path = os.path.join(SCRIPT_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    wp.init()

    device = "cuda"
    with wp.ScopedDevice(device):
        if not wp.get_device().is_cuda:
            raise RuntimeError("This test requires CUDA.")

        print("Testing compilation of compute shader headers")
        print(f"Headers directory: {SCRIPT_DIR}")

        # Read headers
        tonemap_h = read_header("tonemap.h")
        debug_vis_h = read_header("debug_visualize.h")

        # Build CUDA source
        cuda_source = (
            """
// CUDA texture types
typedef unsigned long long cudaTextureObject_t;
typedef unsigned long long cudaSurfaceObject_t;

// Basic math
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// === Inlined from tonemap.h ===
"""
            + tonemap_h
            + """

// === Inlined from debug_visualize.h ===
"""
            + debug_vis_h
            + """

// Test kernel for tonemap
extern "C" __global__ void testTonemapKernel()
{
    TonemapperParams tm;
    tm.method = eTonemapFilmic;  // Use the defined constant
    tm.isActive = 1;
    tm.exposure = 1.0f;
    tm.brightness = 0.0f;
    tm.contrast = 1.0f;
    tm.saturation = 1.0f;
    tm.vignette = 0.0f;
    tm._padding = 0.0f;
    
    float3 color = make_float3(1.0f, 0.5f, 0.25f);
    float2 uv = make_float2(0.5f, 0.5f);
    float3 result = applyTonemap(color, uv, tm);
}

// Test kernel for debug visualize
extern "C" __global__ void testDebugVisKernel()
{
    DebugVisualizeParams params;
    params.mode = DEBUG_MODE_FINAL;  // Use the correct constant name
    params.maxDepth = 100.0f;
    
    float3 hdr = make_float3(2.0f, 1.0f, 0.5f);
    float3 result = debugTonemap(hdr);
}
"""
        )

        print(f"\nCompiling compute shader source ({len(cuda_source)} chars)...")

        try:
            ptx = compile_cuda_source_to_ptx(cuda_source, "compute_shaders", device)
            print(f"\nSUCCESS: Compute shader headers compiled ({len(ptx)} bytes PTX)")
            return True
        except Exception as e:
            print("\nFAILED: Compute shader compilation")
            print(f"Error: {e}")
            return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
