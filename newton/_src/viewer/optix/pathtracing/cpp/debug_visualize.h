/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Debug visualization compute kernel - CUDA version
 * Converted from Vulkan GLSL debug_visualize.comp
 *
 * NOTE: This file is designed to be used as a standalone CUDA kernel,
 *       not as part of the OptiX path tracing pipeline.
 */

#ifndef DEBUG_VISUALIZE_H
#define DEBUG_VISUALIZE_H

#define DEBUG_VIS_WORKGROUP_SIZE 8

//-----------------------------------------------------------------------------
// Debug visualization modes
//-----------------------------------------------------------------------------
#define DEBUG_MODE_FINAL           1  // Diffuse/Specular radiance - tonemapped HDR
#define DEBUG_MODE_VIEWZ           2  // ViewZ (linear depth) - normalized
#define DEBUG_MODE_MOTION          3  // Motion vectors - visualize XY as RG
#define DEBUG_MODE_NORMAL          4  // Normal/Roughness - normals remapped
#define DEBUG_MODE_ROUGHNESS       5  // Roughness channel only
#define DEBUG_MODE_BASECOLOR       6  // Diffuse albedo (baseColor)
#define DEBUG_MODE_SPECALBEDO      7  // Specular albedo
#define DEBUG_MODE_SPECHITDIST     8  // Specular hit distance - normalized

//-----------------------------------------------------------------------------
// Debug visualization parameters
//-----------------------------------------------------------------------------
struct DebugVisualizeParams {
    int mode;  // Visualization mode (see DEBUG_MODE_* constants)
    float maxDepth;  // Max depth for normalization
};

//-----------------------------------------------------------------------------
// Vector helper functions (if not already defined)
//-----------------------------------------------------------------------------
#ifndef DEBUG_VIS_VECTOR_HELPERS
#define DEBUG_VIS_VECTOR_HELPERS

static __forceinline__ __device__ float3 debug_clamp(float3 v, float lo, float hi)
{
    return make_float3(fminf(fmaxf(v.x, lo), hi), fminf(fmaxf(v.y, lo), hi), fminf(fmaxf(v.z, lo), hi));
}

static __forceinline__ __device__ float2 debug_clamp2(float2 v, float lo, float hi)
{
    return make_float2(fminf(fmaxf(v.x, lo), hi), fminf(fmaxf(v.y, lo), hi));
}

#endif  // DEBUG_VIS_VECTOR_HELPERS

//-----------------------------------------------------------------------------
// Reinhard tonemapping for HDR -> LDR (matches C++ nvvkhl::TonemapperPostProcess)
//-----------------------------------------------------------------------------
static __forceinline__ __device__ float3 debugTonemap(float3 hdr)
{
    return make_float3(hdr.x / (hdr.x + 1.0f), hdr.y / (hdr.y + 1.0f), hdr.z / (hdr.z + 1.0f));
}

//-----------------------------------------------------------------------------
// Debug visualization launch parameters
//-----------------------------------------------------------------------------
struct DebugVisualizeLaunchParams {
    // Input buffers (matching RTBindings order)
    cudaSurfaceObject_t viewZ;  // R16F depth
    cudaSurfaceObject_t motionVectors;  // RG16F motion
    cudaSurfaceObject_t normalRoughness;  // RGBA16F normal.xyz + roughness
    cudaSurfaceObject_t baseColorMetallicity;  // RGBA8 color + metallicity
    cudaSurfaceObject_t specAlbedo;  // RGBA8 spec albedo
    cudaSurfaceObject_t color;  // RGBA16F HDR color
    cudaSurfaceObject_t specHitDist;  // R16F hit distance

    // Output
    cudaSurfaceObject_t outputImage;  // RGBA8 output

    // Dimensions
    unsigned int outputWidth;
    unsigned int outputHeight;
    unsigned int colorWidth;
    unsigned int colorHeight;

    // Parameters
    DebugVisualizeParams params;
};

//-----------------------------------------------------------------------------
// Debug visualization CUDA kernel
// Note: This is a template - actual kernel should be defined in your main file
//-----------------------------------------------------------------------------
/*
extern "C" __global__ void debugVisualizeKernel(DebugVisualizeLaunchParams params)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= params.outputWidth || y >= params.outputHeight)
        return;

    // Scale input position if buffers have different size (render vs display)
    int inputX = int(float(x) * float(params.colorWidth) / float(params.outputWidth));
    int inputY = int(float(y) * float(params.colorHeight) / float(params.outputHeight));

    float4 result;

    // C++ nvvkhl::GBuffer ImGui display compatible mode:
    // - uiImageViews use identity swizzle for RGB, alpha swizzled to 1.0
    // - R16F formats display as (R, 0, 0, 1) - RED channel only
    // - RG16F formats display as (R, G, 0, 1)
    // - RGBA formats display as (R, G, B, 1) with alpha forced to 1
    // - Float values are clamped to [0, 1] by the sampler/display

    switch (params.params.mode)
    {
        case DEBUG_MODE_FINAL: // Diffuse/Specular radiance - tonemapped HDR input
        {
            float4 hdrColor;
            surf2Dread(&hdrColor, params.color, inputX * sizeof(float4), inputY);
            float3 hdr = make_float3(hdrColor.x, hdrColor.y, hdrColor.z);
            float3 ldr = debugTonemap(hdr);
            result = make_float4(ldr.x, ldr.y, ldr.z, 1.0f);
            break;
        }

        case DEBUG_MODE_VIEWZ: // ViewZ (linear depth) - normalized for visualization
        {
            float depth;
            surf2Dread(&depth, params.viewZ, inputX * sizeof(float), inputY);
            // Normalize depth: divide by maxDepth for better visualization
            float normalized = depth / params.params.maxDepth;
            normalized = fminf(fmaxf(normalized, 0.0f), 1.0f);
            result = make_float4(normalized, normalized, normalized, 1.0f);
            break;
        }

        case DEBUG_MODE_MOTION: // Motion vectors - visualize XY motion as RG
        {
            float2 motion;
            surf2Dread(&motion, params.motionVectors, inputX * sizeof(float2), inputY);
            // Motion vectors are in pixels, scale and shift to [0,1] for visualization
            // Map [-0.5, 0.5] to [0, 1] for visibility of small motions
            float2 vis = make_float2(motion.x * 10.0f + 0.5f, motion.y * 10.0f + 0.5f);
            vis = debug_clamp2(vis, 0.0f, 1.0f);
            result = make_float4(vis.x, vis.y, 0.0f, 1.0f);
            break;
        }

        case DEBUG_MODE_NORMAL: // Normal/Roughness - normals remapped from [-1,1] to [0,1]
        {
            float4 nr;
            surf2Dread(&nr, params.normalRoughness, inputX * sizeof(float4), inputY);
            // Remap normals from [-1,1] to [0,1] for display
            float3 normal = make_float3(nr.x * 0.5f + 0.5f, nr.y * 0.5f + 0.5f, nr.z * 0.5f + 0.5f);
            result = make_float4(normal.x, normal.y, normal.z, 1.0f);
            break;
        }

        case DEBUG_MODE_ROUGHNESS: // Roughness channel only (from normal/roughness alpha)
        {
            float4 nr;
            surf2Dread(&nr, params.normalRoughness, inputX * sizeof(float4), inputY);
            float roughness = nr.w;
            result = make_float4(roughness, roughness, roughness, 1.0f);
            break;
        }

        case DEBUG_MODE_BASECOLOR: // Diffuse albedo (baseColor)
        {
            float4 bcm;
            surf2Dread(&bcm, params.baseColorMetallicity, inputX * sizeof(float4), inputY);
            result = make_float4(bcm.x, bcm.y, bcm.z, 1.0f);
            break;
        }

        case DEBUG_MODE_SPECALBEDO: // Specular albedo
        {
            float4 sa;
            surf2Dread(&sa, params.specAlbedo, inputX * sizeof(float4), inputY);
            result = make_float4(sa.x, sa.y, sa.z, 1.0f);
            break;
        }

        case DEBUG_MODE_SPECHITDIST: // Specular hit distance - normalized for visualization
        {
            float hitDist;
            surf2Dread(&hitDist, params.specHitDist, inputX * sizeof(float), inputY);
            // Normalize by maxDepth for visualization
            float normalized = hitDist / params.params.maxDepth;
            normalized = fminf(fmaxf(normalized, 0.0f), 1.0f);
            result = make_float4(normalized, normalized, normalized, 1.0f);
            break;
        }

        default:
            result = make_float4(1.0f, 0.0f, 1.0f, 1.0f); // Magenta for invalid mode
    }

    // Write to output surface (convert to uchar4 for RGBA8)
    uchar4 output = make_uchar4(
        (unsigned char)(result.x * 255.0f),
        (unsigned char)(result.y * 255.0f),
        (unsigned char)(result.z * 255.0f),
        (unsigned char)(result.w * 255.0f)
    );
    surf2Dwrite(output, params.outputImage, x * sizeof(uchar4), y);
}
*/

#endif  // DEBUG_VISUALIZE_H
