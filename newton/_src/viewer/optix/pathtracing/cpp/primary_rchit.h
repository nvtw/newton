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

// Primary ray closest-hit shader - CUDA/OptiX version
// Converted from Vulkan GLSL primary.rchit
//
// This file contains the __closesthit__primary function.
// It returns geometry data only; material evaluation is done in raygen.
// NOTE: This file is designed to be inlined into a single CUDA source.
//       Include all dependency headers before this file:
//       - shared_types.h, ray_common.h, scene_common.h, get_hit.h
//       OptiX device headers should be included before this file.

#ifndef PRIMARY_RCHIT_H
#define PRIMARY_RCHIT_H

//-----------------------------------------------------------------------------
// SBT data structure for hit groups
// This should match your SBT record layout
//-----------------------------------------------------------------------------
struct HitGroupSbtData {
    // Add any per-instance data here if needed
};

//-----------------------------------------------------------------------------
// Primary closest-hit shader implementation
//
// This shader computes geometry information at the hit point and returns it
// via the payload. Material evaluation is deferred to the raygen shader.
//
// Payload layout (using optixGetPayload_* / optixSetPayload_*):
//   0: hitT (float, as uint bits)
//   1-3: normal_envmapRadiance (float3, as uint bits)
//   4-6: tangent (float3, as uint bits)
//   7-8: uv (float2, as uint bits)
//   9: materialId (uint)
//   10: bitangentSign (float, as uint bits)
//   11: instanceId (int, as uint bits)
//   12: meshId (uint)
//   13: primitiveId (uint)
//   14-16: barycentrics (float3, as uint bits)
//   17-18: uv1 (float2, as uint bits)
//-----------------------------------------------------------------------------

// Helper to convert float to uint bits
static __forceinline__ __device__ unsigned int float_as_uint(float f) { return __float_as_uint(f); }

// Helper to convert uint bits to float
static __forceinline__ __device__ float uint_as_float(unsigned int u) { return __uint_as_float(u); }

//-----------------------------------------------------------------------------
// Set payload from RayPayload structure
//-----------------------------------------------------------------------------
static __forceinline__ __device__ void setRayPayload(const RayPayload& payload)
{
    optixSetPayload_0(float_as_uint(payload.hitT));
    optixSetPayload_1(float_as_uint(payload.normal_envmapRadiance.x));
    optixSetPayload_2(float_as_uint(payload.normal_envmapRadiance.y));
    optixSetPayload_3(float_as_uint(payload.normal_envmapRadiance.z));
    optixSetPayload_4(float_as_uint(payload.tangent.x));
    optixSetPayload_5(float_as_uint(payload.tangent.y));
    optixSetPayload_6(float_as_uint(payload.tangent.z));
    optixSetPayload_7(float_as_uint(payload.uv.x));
    optixSetPayload_8(float_as_uint(payload.uv.y));
    optixSetPayload_9(payload.materialId);
    optixSetPayload_10(float_as_uint(payload.bitangentSign));
    optixSetPayload_11(static_cast<unsigned int>(payload.instanceId));
    optixSetPayload_12(payload.meshId);
    optixSetPayload_13(payload.primitiveId);
    optixSetPayload_14(float_as_uint(payload.barycentrics.x));
    optixSetPayload_15(float_as_uint(payload.barycentrics.y));
    optixSetPayload_16(float_as_uint(payload.barycentrics.z));
    optixSetPayload_17(float_as_uint(payload.uv1.x));
    optixSetPayload_18(float_as_uint(payload.uv1.y));
}

//-----------------------------------------------------------------------------
// Get payload into RayPayload structure
//-----------------------------------------------------------------------------
static __forceinline__ __device__ RayPayload getRayPayload()
{
    RayPayload payload;
    payload.hitT = uint_as_float(optixGetPayload_0());
    payload.normal_envmapRadiance.x = uint_as_float(optixGetPayload_1());
    payload.normal_envmapRadiance.y = uint_as_float(optixGetPayload_2());
    payload.normal_envmapRadiance.z = uint_as_float(optixGetPayload_3());
    payload.tangent.x = uint_as_float(optixGetPayload_4());
    payload.tangent.y = uint_as_float(optixGetPayload_5());
    payload.tangent.z = uint_as_float(optixGetPayload_6());
    payload.uv.x = uint_as_float(optixGetPayload_7());
    payload.uv.y = uint_as_float(optixGetPayload_8());
    payload.materialId = optixGetPayload_9();
    payload.bitangentSign = uint_as_float(optixGetPayload_10());
    payload.instanceId = static_cast<int>(optixGetPayload_11());
    payload.meshId = optixGetPayload_12();
    payload.primitiveId = optixGetPayload_13();
    payload.barycentrics.x = uint_as_float(optixGetPayload_14());
    payload.barycentrics.y = uint_as_float(optixGetPayload_15());
    payload.barycentrics.z = uint_as_float(optixGetPayload_16());
    payload.uv1.x = uint_as_float(optixGetPayload_17());
    payload.uv1.y = uint_as_float(optixGetPayload_18());
    return payload;
}

//-----------------------------------------------------------------------------
// Primary closest-hit shader
//
// Launch params must provide:
// - renderPrimitiveAddress: pointer to RenderPrimitive array
// - materialIdBufferAddress: pointer to per-primitive material IDs (or in RenderPrimitive)
// - bitangentFlip: float for bitangent direction
//-----------------------------------------------------------------------------
/*
extern "C" __global__ void __closesthit__primary()
{
    // Get RenderPrimitive using instance custom index
    const RenderPrimitive* renderPrims = reinterpret_cast<const RenderPrimitive*>(params.renderPrimitiveAddress);
    RenderPrimitive renderPrim = renderPrims[optixGetInstanceId()];  // or optixGetInstanceCustomIndex()

    // Get hit state using OptiX built-ins
    HitState hit = GetHitStateOptiX(renderPrim, params.pc.bitangentFlip);

    // Get material ID from per-mesh material ID buffer
    const unsigned int* materialIds = reinterpret_cast<const unsigned int*>(renderPrim.materialIdAddress);
    unsigned int materialId = materialIds[optixGetPrimitiveIndex()];

    // Get barycentrics for deformable mesh motion vectors
    float2 attribs = optixGetTriangleBarycentrics();
    float3 bary = make_float3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);

    // Set payload with geometry data
    RayPayload payload;
    payload.hitT = optixGetRayTmax();
    payload.normal_envmapRadiance = hit.nrm;
    payload.tangent = hit.tangent;
    payload.uv = hit.uv;
    payload.materialId = materialId;
    payload.bitangentSign = hit.bitangentSign;
    payload.instanceId = optixGetInstanceId();
    payload.meshId = optixGetInstanceId();  // or custom index
    payload.primitiveId = optixGetPrimitiveIndex();
    payload.barycentrics = bary;

    setRayPayload(payload);
}
*/

#endif  // PRIMARY_RCHIT_H
