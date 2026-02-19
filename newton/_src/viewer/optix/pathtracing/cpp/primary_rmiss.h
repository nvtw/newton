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

// Primary ray miss shader - CUDA/OptiX version
// Converted from Vulkan GLSL primary.rmiss
//
// This shader is executed when primary rays miss all geometry and hit the
// background environment map. It returns the environment color in the payload.
// NOTE: This file is designed to be inlined into a single CUDA source.
//       Include all dependency headers before this file:
//       - shared_types.h, sky_common.h, dlss_helper.h, ray_common.h, func_common.h
//       OptiX device headers should be included before this file.

#ifndef PRIMARY_RMISS_H
#define PRIMARY_RMISS_H

//-----------------------------------------------------------------------------
// Helper to convert float to uint bits
//-----------------------------------------------------------------------------
static __forceinline__ __device__ unsigned int float_as_uint_miss(float f) { return __float_as_uint(f); }

//-----------------------------------------------------------------------------
// Set miss payload
// For miss, we set:
//   - hitT = DLSS_INF_DISTANCE (indicates miss)
//   - normal_envmapRadiance = environment color
//-----------------------------------------------------------------------------
static __forceinline__ __device__ void setMissPayload(float hitT, float3 envColor)
{
    optixSetPayload_0(float_as_uint_miss(hitT));
    optixSetPayload_1(float_as_uint_miss(envColor.x));
    optixSetPayload_2(float_as_uint_miss(envColor.y));
    optixSetPayload_3(float_as_uint_miss(envColor.z));
}

//-----------------------------------------------------------------------------
// Evaluate environment color for a given direction
//-----------------------------------------------------------------------------
static __forceinline__ __device__ float3 evaluateEnvironment(
    float3 rayDirection,
    const FrameInfo& frameInfo,
    const PhysicalSkyParameters& skyInfo,
    cudaTextureObject_t hdrTexture
)
{
    float3 envColor;

    if (TEST_FLAG(frameInfo.flags, FLAGS_ENVMAP_SKY)) {
        envColor = evalPhysicalSky(skyInfo, rayDirection);
    } else {
        // Rotate direction for HDR environment
        float3 dir = rotate(rayDirection, make_float3(0.0f, 1.0f, 0.0f), -frameInfo.envRotation);

        // Convert direction to spherical UV coordinates
        float2 uv = getSphericalUv(dir);

        // Sample HDR texture
        float4 hdrColor = tex2D<float4>(hdrTexture, uv.x, uv.y);
        envColor = make_float3(hdrColor.x, hdrColor.y, hdrColor.z);
    }

    // Apply environment intensity
    envColor = envColor * make_float3(frameInfo.envIntensity.x, frameInfo.envIntensity.y, frameInfo.envIntensity.z);

    return envColor;
}

//-----------------------------------------------------------------------------
// Primary miss shader implementation
//
// Launch params must provide:
// - frameInfo: FrameInfo structure with flags, envRotation, envIntensity
// - skyInfo: PhysicalSkyParameters for procedural sky
// - hdrTexture: cudaTextureObject_t for HDR environment map
//-----------------------------------------------------------------------------
/*
extern "C" __global__ void __miss__primary()
{
    float3 rayDirection = optixGetWorldRayDirection();

    float3 envColor = evaluateEnvironment(
        rayDirection,
        params.frameInfo,
        params.skyInfo,
        params.hdrTexture
    );

    // Set payload: hitT = DLSS_INF_DISTANCE indicates miss
    // normal_envmapRadiance contains the environment color
    setMissPayload(DLSS_INF_DISTANCE, envColor);
}
*/

#endif  // PRIMARY_RMISS_H
