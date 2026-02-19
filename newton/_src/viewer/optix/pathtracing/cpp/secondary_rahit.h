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

// Secondary ray any-hit shader - CUDA/OptiX version
// Converted from Vulkan GLSL secondary.rahit
//
// This shader handles alpha testing and stochastic transparency for
// secondary rays (shadow rays and bounce rays).
// NOTE: This file is designed to be inlined into a single CUDA source.
//       Include all dependency headers before this file:
//       - shared_types.h, scene_common.h, pbr_common.h, get_hit.h
//       OptiX device headers should be included before this file.

#ifndef SECONDARY_RAHIT_H
#define SECONDARY_RAHIT_H

//-----------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------
static __forceinline__ __device__ unsigned int float_as_uint_ahit(float f) { return __float_as_uint(f); }

static __forceinline__ __device__ float uint_as_float_ahit(unsigned int u) { return __uint_as_float(u); }

//-----------------------------------------------------------------------------
// Get seed from payload (for stochastic transparency)
//-----------------------------------------------------------------------------
static __forceinline__ __device__ unsigned int getSeedFromPayload() { return optixGetPayload_0(); }

static __forceinline__ __device__ void setSeedInPayload(unsigned int seed) { optixSetPayload_0(seed); }

//-----------------------------------------------------------------------------
// Alpha test function
// Returns true if the hit should be ignored (transparent)
//-----------------------------------------------------------------------------
static __forceinline__ __device__ bool
alphaTest(const GltfShadeMaterial& mat, float2 uv, unsigned int& seed, cudaTextureObject_t* textureObjects)
{
    // Opaque materials never get ignored
    if (mat.alphaMode == ALPHA_OPAQUE)
        return false;

    // Get base alpha
    float alpha = mat.pbrBaseColorFactor.w;

    // Sample texture if present
    if (mat.pbrBaseColorTexture.index > -1 && textureObjects != nullptr) {
        float4 texColor = tex2D<float4>(textureObjects[mat.pbrBaseColorTexture.index], uv.x, uv.y);
        alpha *= texColor.w;
    }

    // Alpha mask mode: binary cutoff
    if (mat.alphaMode == ALPHA_MASK) {
        return alpha <= mat.alphaCutoff;
    }

    // Alpha blend mode: stochastic transparency
    // Generate random value and compare to alpha
    float r = rand(seed);
    return alpha < r;
}

//-----------------------------------------------------------------------------
// Any-hit shader for shadow rays
//
// This shader is called for every potential hit during shadow ray traversal.
// It performs alpha testing and ignores transparent hits.
//
// For shadow rays, we use a simplified payload:
//   payload_0: seed (uint) for stochastic transparency
//
// Launch params must provide:
// - renderPrimitiveAddress: pointer to RenderPrimitive array
// - gpuMaterialAddress: pointer to GltfShadeMaterial array
// - textureObjects: cudaTextureObject_t array (optional)
//-----------------------------------------------------------------------------
/*
extern "C" __global__ void __anyhit__shadow()
{
    // Get RenderPrimitive
    const RenderPrimitive* renderPrims = reinterpret_cast<const RenderPrimitive*>(params.renderPrimitiveAddress);
    RenderPrimitive renderPrim = renderPrims[optixGetInstanceId()];

    // Get material ID
    const unsigned int* materialIds = reinterpret_cast<const unsigned int*>(renderPrim.materialIdAddress);
    unsigned int matIndex = materialIds[optixGetPrimitiveIndex()];

    // Get material
    const GltfShadeMaterial* materials = reinterpret_cast<const GltfShadeMaterial*>(params.gpuMaterialAddress);
    GltfShadeMaterial mat = materials[matIndex];

    // Quick exit for opaque materials
    if(mat.alphaMode == ALPHA_OPAQUE)
        return;  // Accept hit

    // Get UV coordinates
    float2 attribs = optixGetTriangleBarycentrics();
    float3 barycentrics = make_float3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    uint3 triangleIndex = getTriangleIndices(renderPrim, optixGetPrimitiveIndex());
    float2 uv = getInterpolatedVertexTexCoord0(renderPrim, triangleIndex, barycentrics);

    // Get seed for stochastic transparency
    unsigned int seed = getSeedFromPayload();

    // Perform alpha test
    bool isTransparent = alphaTest(mat, uv, seed, params.textureObjects);

    // Update seed in payload
    setSeedInPayload(seed);

    if(isTransparent)
    {
        // Ignore this hit, continue traversal
        optixIgnoreIntersection();
    }
    // Otherwise, accept the hit (shadow is cast)
}
*/

//-----------------------------------------------------------------------------
// Any-hit shader for secondary (bounce) rays
//
// Similar to shadow any-hit, but may need to handle more complex
// transparency scenarios for path tracing.
//-----------------------------------------------------------------------------
/*
extern "C" __global__ void __anyhit__secondary()
{
    // Get RenderPrimitive
    const RenderPrimitive* renderPrims = reinterpret_cast<const RenderPrimitive*>(params.renderPrimitiveAddress);
    RenderPrimitive renderPrim = renderPrims[optixGetInstanceId()];

    // Get material ID
    const unsigned int* materialIds = reinterpret_cast<const unsigned int*>(renderPrim.materialIdAddress);
    unsigned int matIndex = materialIds[optixGetPrimitiveIndex()];

    // Get material
    const GltfShadeMaterial* materials = reinterpret_cast<const GltfShadeMaterial*>(params.gpuMaterialAddress);
    GltfShadeMaterial mat = materials[matIndex];

    // Quick exit for opaque materials
    if(mat.alphaMode == ALPHA_OPAQUE)
        return;  // Accept hit

    // Get UV coordinates
    float2 attribs = optixGetTriangleBarycentrics();
    float3 barycentrics = make_float3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    uint3 triangleIndex = getTriangleIndices(renderPrim, optixGetPrimitiveIndex());
    float2 uv = getInterpolatedVertexTexCoord0(renderPrim, triangleIndex, barycentrics);

    // Get seed from payload (payload_0 for secondary rays)
    unsigned int seed = optixGetPayload_0();

    // Perform alpha test
    bool isTransparent = alphaTest(mat, uv, seed, params.textureObjects);

    // Update seed in payload
    optixSetPayload_0(seed);

    if(isTransparent)
    {
        // Ignore this hit, continue traversal
        optixIgnoreIntersection();
    }
}
*/

#endif  // SECONDARY_RAHIT_H
