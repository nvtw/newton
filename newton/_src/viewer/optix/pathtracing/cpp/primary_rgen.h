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

// Primary ray generation shader - CUDA/OptiX version
// Converted from Vulkan GLSL primary.rgen
//
// This file contains the __raygen__primary function for the path tracer.
// NOTE: This file is designed to be inlined into a single CUDA source.
//       Include all dependency headers before this file:
//       - shared_types.h, scene_common.h, sky_common.h, pbr_common.h
//       - dlss_helper.h, ray_common.h, mat_eval_common.h
//       OptiX device headers should be included before this file.

#ifndef PRIMARY_RGEN_H
#define PRIMARY_RGEN_H

//-----------------------------------------------------------------------------
// Launch parameters structure - must be defined by the host application
// This is a template showing the expected structure
//-----------------------------------------------------------------------------
/*
struct LaunchParams
{
    // Acceleration structure
    OptixTraversableHandle traversable;

    // Frame info
    FrameInfo frameInfo;

    // Scene description
    unsigned long long gpuMaterialAddress;
    unsigned long long renderNodeAddress;
    unsigned long long renderPrimitiveAddress;
    unsigned long long materialIdBufferAddress;

    // Instance transforms
    TransformMatrix3x4* instanceTransforms;
    TransformMatrix3x4* prevInstanceTransforms;

    // Pick result
    PickResult* pickResult;

    // Output images (surface objects or pointers)
    cudaSurfaceObject_t dlssColor;
    cudaSurfaceObject_t dlssObjectMotion;
    cudaSurfaceObject_t dlssNormalRoughness;
    cudaSurfaceObject_t dlssViewZ;
    cudaSurfaceObject_t dlssSpecAlbedo;
    cudaSurfaceObject_t dlssSpecHitDistance;
    cudaSurfaceObject_t dlssBaseColorMetallicity;

    // Environment
    EnvAccel* envSamplingData;
    cudaTextureObject_t hdrTexture;
    unsigned int hdrWidth;
    unsigned int hdrHeight;

    // Sky parameters
    PhysicalSkyParameters skyInfo;

    // Textures
    cudaTextureObject_t* textureObjects;

    // Push constants
    RtxPushConstant pc;
};

extern "C" __constant__ LaunchParams params;
*/

//-----------------------------------------------------------------------------
// Hit state structure for primary ray
//-----------------------------------------------------------------------------
struct PrimaryHitState {
    float3 pos;
    float3 nrm;
    float3 geonrm;
    float2 uv;
    float3 tangent;
    float3 bitangent;
    float bitangentSign;
};

//-----------------------------------------------------------------------------
// Alpha transparency check
//-----------------------------------------------------------------------------
static __forceinline__ __device__ bool isAlphaTransparent(
    unsigned int materialId,
    float2 uv,
    unsigned int& seed,
    const GltfShadeMaterial* materials,
    cudaTextureObject_t* textureObjects
)
{
    GltfShadeMaterial mat = materials[materialId];

    if (mat.alphaMode == ALPHA_OPAQUE)
        return false;

    float alpha = mat.pbrBaseColorFactor.w;
    if (mat.pbrBaseColorTexture.index > -1 && textureObjects != nullptr) {
        float4 texColor = tex2D<float4>(textureObjects[mat.pbrBaseColorTexture.index], uv.x, uv.y);
        alpha *= texColor.w;
    }

    if (mat.alphaMode == ALPHA_MASK) {
        return alpha <= mat.alphaCutoff;
    } else  // ALPHA_BLEND - stochastic transparency
    {
        return alpha < rand(seed);
    }
}

//-----------------------------------------------------------------------------
// Build hit info from payload
//-----------------------------------------------------------------------------
static __forceinline__ __device__ void buildHitInfo(
    const RayPayload& pload,
    float3 rayOrigin,
    float3 rayDirection,
    PbrMaterial& pbrMat,
    PrimaryHitState& hitState,
    const GltfShadeMaterial* materials,
    cudaTextureObject_t* textureObjects,
    float overrideRoughness,
    float overrideMetallic
)
{
    hitState.pos = rayOrigin + pload.hitT * rayDirection;
    hitState.nrm = pload.normal_envmapRadiance;
    hitState.geonrm = hitState.nrm;
    hitState.uv = pload.uv;
    hitState.tangent = pload.tangent;
    hitState.bitangent = cross(hitState.nrm, hitState.tangent) * pload.bitangentSign;

    GltfShadeMaterial mat = materials[pload.materialId];
    pbrMat = evaluateMaterial(mat, hitState.nrm, hitState.tangent, hitState.bitangent, hitState.uv);

    if (overrideRoughness > 0) {
        float r = fminf(fmaxf(overrideRoughness, MICROFACET_MIN_ROUGHNESS), 1.0f);
        pbrMat.roughness = make_float2(r * r, r * r);
    }
    if (overrideMetallic > 0) {
        pbrMat.metallic = overrideMetallic;
    }
}

//-----------------------------------------------------------------------------
// Compute motion vectors
//-----------------------------------------------------------------------------
static __forceinline__ __device__ float2
computeCameraMotionVector(float2 pixelCenter, float4 motionOrigin, const float4x4& prevMVP, uint2 launchSize)
{
    // Project to previous frame using matrix-vector multiplication
    float4 oldPos = mul(prevMVP, motionOrigin);

    oldPos.x /= oldPos.w;
    oldPos.y /= oldPos.w;
    oldPos.x = (oldPos.x * 0.5f + 0.5f) * float(launchSize.x);
    oldPos.y = (oldPos.y * 0.5f + 0.5f) * float(launchSize.y);

    return make_float2(oldPos.x - pixelCenter.x, oldPos.y - pixelCenter.y);
}

//-----------------------------------------------------------------------------
// Direct light contribution from HDR environment
//-----------------------------------------------------------------------------
static __forceinline__ __device__ void HdrContrib(
    const PbrMaterial& pbrMat,
    float3 startPos,
    float3 toEye,
    float3& outRadiance,
    unsigned int& seed,
    OptixTraversableHandle traversable,
    const FrameInfo& frameInfo,
    const PhysicalSkyParameters& skyInfo,
    cudaTextureObject_t hdrTexture,
    const EnvAccel* envSamplingData,
    unsigned int hdrWidth,
    unsigned int hdrHeight
)
{
    outRadiance = make_float3(0.0f, 0.0f, 0.0f);

    float3 lightDir;
    float3 lightContrib;
    float lightPdf;

    float3 randVal = make_float3(rand(seed), rand(seed), rand(seed));

    if (TEST_FLAG(frameInfo.flags, FLAGS_ENVMAP_SKY)) {
        SkySamplingResult skySample = samplePhysicalSky(skyInfo, make_float2(randVal.x, randVal.y));
        lightDir = skySample.direction;
        lightContrib = skySample.radiance;
        lightPdf = skySample.pdf;
    } else {
        float4 radiance_pdf = environmentSample(hdrTexture, envSamplingData, hdrWidth, hdrHeight, randVal, lightDir);
        lightDir = rotate(lightDir, make_float3(0.0f, 1.0f, 0.0f), frameInfo.envRotation);
        lightContrib = make_float3(radiance_pdf.x, radiance_pdf.y, radiance_pdf.z);
        lightPdf = radiance_pdf.w;
    }

    lightContrib
        = lightContrib * make_float3(frameInfo.envIntensity.x, frameInfo.envIntensity.y, frameInfo.envIntensity.z);

    float dotNL = dot(lightDir, pbrMat.N);

    if (dotNL > 0.0f && lightPdf > 0.0f) {
        BsdfEvaluateData bsdfEval;
        bsdfEval.k1 = toEye;
        bsdfEval.k2 = lightDir;
        bsdfEval.xi = randVal;

        bsdfEvaluate(bsdfEval, pbrMat);

        if (bsdfEval.pdf > 0.0f) {
            const float mis_weight = powerHeuristic(lightPdf, bsdfEval.pdf);
            float3 lightRadiance = mis_weight * lightContrib / fmaxf(lightPdf, 0.0001f);
            float3 radiance = (bsdfEval.bsdf_diffuse + bsdfEval.bsdf_glossy) * lightRadiance;

            // Shadow ray
            unsigned int shadowPayload = 0;
            optixTrace(
                traversable, startPos, lightDir, 0.001f, DLSS_INF_DISTANCE, 0.0f, 0xFF,
                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, SBTOFFSET_SECONDARY, 1,
                MISSINDEX_SECONDARY, shadowPayload
            );

            // If shadow ray missed (light visible), add contribution
            if (shadowPayload == 1) {
                outRadiance = radiance;
            }
        }
    }
}

//-----------------------------------------------------------------------------
// Note: The actual __raygen__primary function should be implemented in your
// main CUDA file, using the helper functions above. Here's a template:
//-----------------------------------------------------------------------------
/*
extern "C" __global__ void __raygen__primary()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Initialize random seed
    unsigned int seed = xxhash32(make_uint3(idx.x, idx.y, params.pc.frame));

    // Compute ray direction from camera
    float2 pixelCenter = make_float2(float(idx.x) + 0.5f, float(idx.y) + 0.5f);
    pixelCenter = pixelCenter + params.frameInfo.jitter;

    float2 inUV = pixelCenter / make_float2(float(dim.x), float(dim.y));
    float2 d = inUV * 2.0f - make_float2(1.0f, 1.0f);

    // Get camera origin and direction from view matrices
    // ... (implement based on your camera setup)

    // Trace primary ray
    RayPayload payload;
    optixTrace(
        params.traversable,
        origin,
        direction,
        0.01f,
        1e32f,
        0.0f,
        0xFF,
        OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
        SBTOFFSET_PRIMARY,
        1,
        MISSINDEX_PRIMARY,
        // ... payload registers
    );

    // Process hit and compute shading
    // ... (use helper functions above)

    // Write output to DLSS buffers
    // surf2Dwrite(color, params.dlssColor, idx.x * sizeof(float4), idx.y);
}
*/

#endif  // PRIMARY_RGEN_H
