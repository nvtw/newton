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

// Secondary ray closest-hit shader - CUDA/OptiX version
// Converted from Vulkan GLSL secondary.rchit
//
// This shader handles path tracing bounces, evaluating materials and
// computing direct lighting at each hit point.
// NOTE: This file is designed to be inlined into a single CUDA source.
//       Include all dependency headers before this file:
//       - shared_types.h, scene_common.h, sky_common.h, pbr_common.h
//       - dlss_helper.h, ray_common.h, get_hit.h, mat_eval_common.h
//       OptiX device headers should be included before this file.

#ifndef SECONDARY_RCHIT_H
#define SECONDARY_RCHIT_H

//-----------------------------------------------------------------------------
// Helper functions for payload access
//-----------------------------------------------------------------------------
static __forceinline__ __device__ unsigned int float_as_uint_sec(float f) { return __float_as_uint(f); }

static __forceinline__ __device__ float uint_as_float_sec(unsigned int u) { return __uint_as_float(u); }

//-----------------------------------------------------------------------------
// PayloadSecondary access functions
// Payload layout:
//   0: seed (uint)
//   1: hitT (float)
//   2-4: contrib (float3)
//   5-7: weight (float3)
//   8-10: rayOrigin (float3)
//   11-13: rayDirection (float3)
//   14: bsdfPDF (float)
//   15-16: maxRoughness (float2)
//-----------------------------------------------------------------------------
static __forceinline__ __device__ PayloadSecondary getSecondaryPayload()
{
    PayloadSecondary payload;
    payload.seed = optixGetPayload_0();
    payload.hitT = uint_as_float_sec(optixGetPayload_1());
    payload.contrib.x = uint_as_float_sec(optixGetPayload_2());
    payload.contrib.y = uint_as_float_sec(optixGetPayload_3());
    payload.contrib.z = uint_as_float_sec(optixGetPayload_4());
    payload.weight.x = uint_as_float_sec(optixGetPayload_5());
    payload.weight.y = uint_as_float_sec(optixGetPayload_6());
    payload.weight.z = uint_as_float_sec(optixGetPayload_7());
    payload.rayOrigin.x = uint_as_float_sec(optixGetPayload_8());
    payload.rayOrigin.y = uint_as_float_sec(optixGetPayload_9());
    payload.rayOrigin.z = uint_as_float_sec(optixGetPayload_10());
    payload.rayDirection.x = uint_as_float_sec(optixGetPayload_11());
    payload.rayDirection.y = uint_as_float_sec(optixGetPayload_12());
    payload.rayDirection.z = uint_as_float_sec(optixGetPayload_13());
    payload.bsdfPDF = uint_as_float_sec(optixGetPayload_14());
    payload.maxRoughness.x = uint_as_float_sec(optixGetPayload_15());
    payload.maxRoughness.y = uint_as_float_sec(optixGetPayload_16());
    return payload;
}

static __forceinline__ __device__ void setSecondaryPayload(const PayloadSecondary& payload)
{
    optixSetPayload_0(payload.seed);
    optixSetPayload_1(float_as_uint_sec(payload.hitT));
    optixSetPayload_2(float_as_uint_sec(payload.contrib.x));
    optixSetPayload_3(float_as_uint_sec(payload.contrib.y));
    optixSetPayload_4(float_as_uint_sec(payload.contrib.z));
    optixSetPayload_5(float_as_uint_sec(payload.weight.x));
    optixSetPayload_6(float_as_uint_sec(payload.weight.y));
    optixSetPayload_7(float_as_uint_sec(payload.weight.z));
    optixSetPayload_8(float_as_uint_sec(payload.rayOrigin.x));
    optixSetPayload_9(float_as_uint_sec(payload.rayOrigin.y));
    optixSetPayload_10(float_as_uint_sec(payload.rayOrigin.z));
    optixSetPayload_11(float_as_uint_sec(payload.rayDirection.x));
    optixSetPayload_12(float_as_uint_sec(payload.rayDirection.y));
    optixSetPayload_13(float_as_uint_sec(payload.rayDirection.z));
    optixSetPayload_14(float_as_uint_sec(payload.bsdfPDF));
    optixSetPayload_15(float_as_uint_sec(payload.maxRoughness.x));
    optixSetPayload_16(float_as_uint_sec(payload.maxRoughness.y));
}

//-----------------------------------------------------------------------------
// Shading result structure
//-----------------------------------------------------------------------------
struct ShadingResult {
    float3 weight;
    float3 contrib;
    float3 rayOrigin;
    float3 rayDirection;
    float bsdfPDF;
};

//-----------------------------------------------------------------------------
// Sample lights (HDR environment or physical sky)
//-----------------------------------------------------------------------------
static __forceinline__ __device__ float3 sampleLights(
    const HitState& state,
    unsigned int& seed,
    float3& dirToLight,
    float& lightPdf,
    const FrameInfo& frameInfo,
    const PhysicalSkyParameters& skyInfo,
    cudaTextureObject_t hdrTexture,
    const EnvAccel* envSamplingData,
    unsigned int hdrWidth,
    unsigned int hdrHeight
)
{
    float3 randVal = make_float3(rand(seed), rand(seed), rand(seed));
    float3 lightContrib;

    if (TEST_FLAG(frameInfo.flags, FLAGS_ENVMAP_SKY)) {
        SkySamplingResult skySample = samplePhysicalSky(skyInfo, make_float2(randVal.x, randVal.y));
        dirToLight = skySample.direction;
        lightPdf = skySample.pdf;
        lightContrib = skySample.radiance;
    } else {
        float4 radiance_pdf = environmentSample(hdrTexture, envSamplingData, hdrWidth, hdrHeight, randVal, dirToLight);
        dirToLight = rotate(dirToLight, make_float3(0.0f, 1.0f, 0.0f), frameInfo.envRotation);
        lightContrib = make_float3(radiance_pdf.x, radiance_pdf.y, radiance_pdf.z);
        lightPdf = radiance_pdf.w;
    }

    lightContrib
        = lightContrib * make_float3(frameInfo.envIntensity.x, frameInfo.envIntensity.y, frameInfo.envIntensity.z);

    return lightContrib / fmaxf(lightPdf, 0.0001f);
}

//-----------------------------------------------------------------------------
// Evaluate shading at hit point
//-----------------------------------------------------------------------------
static __forceinline__ __device__ ShadingResult shading(
    const PbrMaterial& pbrMat,
    const HitState& hit,
    unsigned int& seed,
    OptixTraversableHandle traversable,
    const FrameInfo& frameInfo,
    const PhysicalSkyParameters& skyInfo,
    cudaTextureObject_t hdrTexture,
    const EnvAccel* envSamplingData,
    unsigned int hdrWidth,
    unsigned int hdrHeight,
    float currentHitT
)
{
    ShadingResult result;
    result.contrib = pbrMat.emissive;
    result.weight = make_float3(0.0f, 0.0f, 0.0f);
    result.rayOrigin = make_float3(0.0f, 0.0f, 0.0f);
    result.rayDirection = make_float3(0.0f, 0.0f, 0.0f);
    result.bsdfPDF = 0.0f;

    // Sample light
    float3 dirToLight;
    float lightPdf;
    float3 lightRadianceOverPdf = sampleLights(
        hit, seed, dirToLight, lightPdf, frameInfo, skyInfo, hdrTexture, envSamplingData, hdrWidth, hdrHeight
    );

    const bool lightValid = (dot(dirToLight, pbrMat.N) > 0.0f) && lightPdf > 0.0f;

    if (lightValid) {
        BsdfEvaluateData evalData;
        evalData.k1 = -optixGetWorldRayDirection();
        evalData.k2 = dirToLight;
        evalData.xi = make_float3(rand(seed), rand(seed), rand(seed));

        bsdfEvaluate(evalData, pbrMat);

        if (evalData.pdf > 0.0f) {
            const float misWeight = powerHeuristic(lightPdf, evalData.pdf);
            const float3 w = lightRadianceOverPdf * misWeight;
            float3 contribution = w * (evalData.bsdf_diffuse + evalData.bsdf_glossy);

            // Shadow ray
            float3 shadowRayOrigin = offsetRay(hit.pos, hit.geonrm);
            unsigned int shadowPayload = 0;

            optixTrace(
                traversable, shadowRayOrigin, dirToLight, 0.001f, DLSS_INF_DISTANCE, 0.0f, 0xFF,
                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, SBTOFFSET_SECONDARY, 1,
                MISSINDEX_SECONDARY, shadowPayload
            );

            if (shadowPayload == 1)  // Light visible
            {
                result.contrib = result.contrib + contribution;
            }
        }
    }

    // Sample BSDF for next ray
    {
        BsdfSampleData sampleData;
        sampleData.k1 = -optixGetWorldRayDirection();
        sampleData.xi = make_float3(rand(seed), rand(seed), rand(seed));
        bsdfSample(sampleData, pbrMat);

        if (sampleData.event_type == BSDF_EVENT_ABSORB || any_isnan(sampleData.bsdf_over_pdf)
            || any_isinf(sampleData.bsdf_over_pdf)) {
            result.weight = make_float3(0.0f, 0.0f, 0.0f);
        } else {
            result.weight = sampleData.bsdf_over_pdf;
            result.rayDirection = sampleData.k2;
            result.bsdfPDF = sampleData.pdf;
            float3 offsetDir = dot(result.rayDirection, pbrMat.N) > 0 ? hit.geonrm : -hit.geonrm;
            result.rayOrigin = offsetRay(hit.pos, offsetDir);
        }
    }

    return result;
}

//-----------------------------------------------------------------------------
// Secondary closest-hit shader implementation
//
// Launch params must provide:
// - traversable: OptixTraversableHandle
// - frameInfo: FrameInfo
// - renderPrimitiveAddress: pointer to RenderPrimitive array
// - gpuMaterialAddress: pointer to GltfShadeMaterial array
// - textureObjects: cudaTextureObject_t array
// - skyInfo: PhysicalSkyParameters
// - hdrTexture: cudaTextureObject_t
// - envSamplingData: EnvAccel*
// - hdrWidth, hdrHeight: unsigned int
// - pc: RtxPushConstant
//-----------------------------------------------------------------------------
/*
extern "C" __global__ void __closesthit__secondary()
{
    // Get payload
    PayloadSecondary payload = getSecondaryPayload();

    // Get RenderPrimitive
    const RenderPrimitive* renderPrims = reinterpret_cast<const RenderPrimitive*>(params.renderPrimitiveAddress);
    RenderPrimitive renderPrim = renderPrims[optixGetInstanceId()];

    // Get hit state
    HitState hit = GetHitStateOptiX(renderPrim, params.pc.bitangentFlip);

    // Get material
    const unsigned int* materialIds = reinterpret_cast<const unsigned int*>(renderPrim.materialIdAddress);
    unsigned int matIndex = materialIds[optixGetPrimitiveIndex()];
    const GltfShadeMaterial* materials = reinterpret_cast<const GltfShadeMaterial*>(params.gpuMaterialAddress);
    GltfShadeMaterial mat = materials[matIndex];

    // Evaluate material
    PbrMaterial pbrMat = evaluateMaterial(mat, hit.nrm, hit.tangent, hit.bitangent, hit.uv);

    // Override material properties if requested
    if(params.pc.overrideRoughness > 0)
    {
        float r = fminf(fmaxf(params.pc.overrideRoughness, 0.001f), 1.0f);
        pbrMat.roughness = make_float2(r * r, r * r);
    }
    if(params.pc.overrideMetallic > 0)
    {
        pbrMat.metallic = params.pc.overrideMetallic;
    }

    // Path regularization
    if(TEST_FLAG(params.frameInfo.flags, FLAGS_USE_PATH_REGULARIZATION))
    {
        payload.maxRoughness = fmaxf(pbrMat.roughness, payload.maxRoughness);
        pbrMat.roughness = payload.maxRoughness;
    }

    payload.hitT = optixGetRayTmax();

    // Compute shading
    ShadingResult result = shading(pbrMat, hit, payload.seed,
                                    params.traversable, params.frameInfo, params.skyInfo,
                                    params.hdrTexture, params.envSamplingData,
                                    params.hdrWidth, params.hdrHeight, payload.hitT);

    // Guard against NaN/Inf
    if(any_isnan(result.weight) || any_isinf(result.weight))
        result.weight = make_float3(0.0f, 0.0f, 0.0f);
    if(any_isnan(result.contrib) || any_isinf(result.contrib))
        result.contrib = make_float3(0.0f, 0.0f, 0.0f);

    payload.weight = result.weight;
    payload.contrib = result.contrib;
    payload.rayOrigin = result.rayOrigin;
    payload.rayDirection = result.rayDirection;
    payload.bsdfPDF = fmaxf(result.bsdfPDF, 0.0001f);

    // If weight is zero, signal to stop tracing
    if(length(payload.weight) < 0.0001f)
    {
        payload.hitT = -payload.hitT;  // Negative hitT signals termination
    }

    setSecondaryPayload(payload);
}
*/

#endif  // SECONDARY_RCHIT_H
