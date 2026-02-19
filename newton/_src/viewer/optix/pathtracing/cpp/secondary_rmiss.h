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

// Secondary ray miss shader - CUDA/OptiX version
// Converted from Vulkan GLSL secondary.rmiss
//
// This shader handles secondary rays that miss geometry and hit the environment.
// It computes the environment contribution with MIS weighting.
// NOTE: This file is designed to be inlined into a single CUDA source.
//       Include all dependency headers before this file:
//       - shared_types.h, sky_common.h, func_common.h, ray_common.h
//       OptiX device headers should be included before this file.

#ifndef SECONDARY_RMISS_H
#define SECONDARY_RMISS_H

//-----------------------------------------------------------------------------
// Helper functions for payload access
//-----------------------------------------------------------------------------
static __forceinline__ __device__ unsigned int float_as_uint_miss_sec(float f) { return __float_as_uint(f); }

static __forceinline__ __device__ float uint_as_float_miss_sec(unsigned int u) { return __uint_as_float(u); }

//-----------------------------------------------------------------------------
// PayloadSecondary access functions for miss shader
//-----------------------------------------------------------------------------
static __forceinline__ __device__ PayloadSecondary getSecondaryPayloadMiss()
{
    PayloadSecondary payload;
    payload.seed = optixGetPayload_0();
    payload.hitT = uint_as_float_miss_sec(optixGetPayload_1());
    payload.contrib.x = uint_as_float_miss_sec(optixGetPayload_2());
    payload.contrib.y = uint_as_float_miss_sec(optixGetPayload_3());
    payload.contrib.z = uint_as_float_miss_sec(optixGetPayload_4());
    payload.weight.x = uint_as_float_miss_sec(optixGetPayload_5());
    payload.weight.y = uint_as_float_miss_sec(optixGetPayload_6());
    payload.weight.z = uint_as_float_miss_sec(optixGetPayload_7());
    payload.rayOrigin.x = uint_as_float_miss_sec(optixGetPayload_8());
    payload.rayOrigin.y = uint_as_float_miss_sec(optixGetPayload_9());
    payload.rayOrigin.z = uint_as_float_miss_sec(optixGetPayload_10());
    payload.rayDirection.x = uint_as_float_miss_sec(optixGetPayload_11());
    payload.rayDirection.y = uint_as_float_miss_sec(optixGetPayload_12());
    payload.rayDirection.z = uint_as_float_miss_sec(optixGetPayload_13());
    payload.bsdfPDF = uint_as_float_miss_sec(optixGetPayload_14());
    payload.maxRoughness.x = uint_as_float_miss_sec(optixGetPayload_15());
    payload.maxRoughness.y = uint_as_float_miss_sec(optixGetPayload_16());
    return payload;
}

static __forceinline__ __device__ void setSecondaryPayloadMiss(const PayloadSecondary& payload)
{
    optixSetPayload_0(payload.seed);
    optixSetPayload_1(float_as_uint_miss_sec(payload.hitT));
    optixSetPayload_2(float_as_uint_miss_sec(payload.contrib.x));
    optixSetPayload_3(float_as_uint_miss_sec(payload.contrib.y));
    optixSetPayload_4(float_as_uint_miss_sec(payload.contrib.z));
    optixSetPayload_5(float_as_uint_miss_sec(payload.weight.x));
    optixSetPayload_6(float_as_uint_miss_sec(payload.weight.y));
    optixSetPayload_7(float_as_uint_miss_sec(payload.weight.z));
    optixSetPayload_8(float_as_uint_miss_sec(payload.rayOrigin.x));
    optixSetPayload_9(float_as_uint_miss_sec(payload.rayOrigin.y));
    optixSetPayload_10(float_as_uint_miss_sec(payload.rayOrigin.z));
    optixSetPayload_11(float_as_uint_miss_sec(payload.rayDirection.x));
    optixSetPayload_12(float_as_uint_miss_sec(payload.rayDirection.y));
    optixSetPayload_13(float_as_uint_miss_sec(payload.rayDirection.z));
    optixSetPayload_14(float_as_uint_miss_sec(payload.bsdfPDF));
    optixSetPayload_15(float_as_uint_miss_sec(payload.maxRoughness.x));
    optixSetPayload_16(float_as_uint_miss_sec(payload.maxRoughness.y));
}

//-----------------------------------------------------------------------------
// Evaluate environment with MIS weighting
//-----------------------------------------------------------------------------
static __forceinline__ __device__ float3 evaluateEnvironmentMIS(
    float3 rayDirection,
    float bsdfPDF,
    const FrameInfo& frameInfo,
    const PhysicalSkyParameters& skyInfo,
    cudaTextureObject_t hdrTexture,
    unsigned int hdrWidth,
    unsigned int hdrHeight
)
{
    float3 envColor;
    float envPDF;

    if (TEST_FLAG(frameInfo.flags, FLAGS_ENVMAP_SKY)) {
        envColor = evalPhysicalSky(skyInfo, rayDirection);
        envPDF = samplePhysicalSkyPDF(skyInfo, rayDirection);
    } else {
        // Rotate direction for HDR environment
        float3 dir = rotate(rayDirection, make_float3(0.0f, 1.0f, 0.0f), -frameInfo.envRotation);

        // Convert direction to spherical UV coordinates
        float2 uv = getSphericalUv(dir);

        // Sample HDR texture
        float4 hdrColor = tex2D<float4>(hdrTexture, uv.x, uv.y);
        envColor = make_float3(hdrColor.x, hdrColor.y, hdrColor.z);

        // Compute PDF for HDR sampling
        // PDF = 1 / (2 * PI * sin(theta)) for uniform spherical sampling
        // For importance-sampled HDR, the PDF is stored in the alpha channel or computed separately
        float theta = acosf(fminf(fmaxf(dir.y, -1.0f), 1.0f));
        float sinTheta = sinf(theta);
        envPDF = (sinTheta > 0.0f) ? 1.0f / (2.0f * M_PI * M_PI * sinTheta) : 0.0f;
    }

    // Apply environment intensity
    envColor = envColor * make_float3(frameInfo.envIntensity.x, frameInfo.envIntensity.y, frameInfo.envIntensity.z);

    // MIS weight
    float misWeight = powerHeuristic(bsdfPDF, envPDF);

    return envColor * misWeight;
}

//-----------------------------------------------------------------------------
// Secondary miss shader implementation
//
// This shader is called when secondary (bounce) rays miss all geometry.
// It evaluates the environment contribution with MIS weighting based on
// the BSDF PDF from the previous bounce.
//
// Launch params must provide:
// - frameInfo: FrameInfo
// - skyInfo: PhysicalSkyParameters
// - hdrTexture: cudaTextureObject_t
// - hdrWidth, hdrHeight: unsigned int
//-----------------------------------------------------------------------------
/*
extern "C" __global__ void __miss__secondary()
{
    PayloadSecondary payload = getSecondaryPayloadMiss();

    float3 rayDirection = optixGetWorldRayDirection();

    // Evaluate environment with MIS
    float3 envContrib = evaluateEnvironmentMIS(
        rayDirection,
        payload.bsdfPDF,
        params.frameInfo,
        params.skyInfo,
        params.hdrTexture,
        params.hdrWidth,
        params.hdrHeight
    );

    // Add environment contribution
    payload.contrib = payload.contrib + envContrib;

    // Signal termination (no more bounces)
    payload.weight = make_float3(0.0f, 0.0f, 0.0f);
    payload.hitT = DLSS_INF_DISTANCE;

    setSecondaryPayloadMiss(payload);
}
*/

//-----------------------------------------------------------------------------
// Shadow miss shader
// Used for shadow rays to determine if a point is visible to the light
//-----------------------------------------------------------------------------
/*
extern "C" __global__ void __miss__shadow()
{
    // Shadow ray missed all geometry, light is visible
    // Set payload to 1 to indicate visibility
    optixSetPayload_0(1);
}
*/

#endif  // SECONDARY_RMISS_H
