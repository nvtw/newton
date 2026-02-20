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

// OptiX entry points used by the Python path tracing viewer.
// This keeps required programs under pathtracing/rt as requested.
//
// NOTE:
// - These are currently baseline entry points used by the Python viewer.
// - The converted files (primary_rgen.h, primary_rchit.h, etc.) contain the
//   fuller translation helper logic and can be wired in incrementally.

#ifndef OPTIX_PROGRAMS_H
#define OPTIX_PROGRAMS_H

// Runtime helper dependencies for payload packing/unpacking.
// Keep dependency order strict:
// - shared/scene/pbr types before material evaluation
// - ray payload types before primary_rchit payload helpers
#include "shared_types.h"
#include "scene_common.h"
#include "pbr_common.h"
#include "ray_common.h"
#include "pathtrace_common.h"
#include "mat_eval_common.h"
#include "dlss_helper.h"
#include "sky_common.h"
#include "viewer_rt_common.h"
#include "primary_rchit.h"

// Compute motion vector for camera motion only (static objects, or sky).
// motionOrigin is the world-space position to be projected into previous frame's screen.
// Use w=0 and a direction vector for points at infinity (sky).
// Matches C# computeCameraMotionVector: uses jittered pixelCenter with MVJittered=false.
// All matrices use GLM/NumPy column-major-in-memory convention; use mul_cm.
static __forceinline__ __device__ float2 compute_camera_motion_vector(
    const float2 pixelCenter, const float4 motionOrigin, const float4x4 prevMVP, const uint2 dim
)
{
    float4 oldPos = mul_cm(prevMVP, motionOrigin);
    oldPos.x /= oldPos.w;
    oldPos.y /= oldPos.w;
    oldPos.x = (oldPos.x * 0.5f + 0.5f) * float(dim.x);
    oldPos.y = (oldPos.y * 0.5f + 0.5f) * float(dim.y);
    return make_float2(oldPos.x - pixelCenter.x, oldPos.y - pixelCenter.y);
}

static __forceinline__ __device__ float3 inverse_transform_point(const TransformMatrix3x4& m, const float3 worldPos)
{
    const float a00 = m.row0.x, a01 = m.row0.y, a02 = m.row0.z;
    const float a10 = m.row1.x, a11 = m.row1.y, a12 = m.row1.z;
    const float a20 = m.row2.x, a21 = m.row2.y, a22 = m.row2.z;

    const float det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20);
    if (fabsf(det) < 1.0e-12f)
        return worldPos;

    const float invDet = 1.0f / det;
    const float i00 = (a11 * a22 - a12 * a21) * invDet;
    const float i01 = (a02 * a21 - a01 * a22) * invDet;
    const float i02 = (a01 * a12 - a02 * a11) * invDet;
    const float i10 = (a12 * a20 - a10 * a22) * invDet;
    const float i11 = (a00 * a22 - a02 * a20) * invDet;
    const float i12 = (a02 * a10 - a00 * a12) * invDet;
    const float i20 = (a10 * a21 - a11 * a20) * invDet;
    const float i21 = (a01 * a20 - a00 * a21) * invDet;
    const float i22 = (a00 * a11 - a01 * a10) * invDet;

    const float3 d = make_float3(worldPos.x - m.row0.w, worldPos.y - m.row1.w, worldPos.z - m.row2.w);
    return make_float3(
        i00 * d.x + i01 * d.y + i02 * d.z, i10 * d.x + i11 * d.y + i12 * d.z, i20 * d.x + i21 * d.y + i22 * d.z
    );
}

static __forceinline__ __device__ bool transforms_equal(const TransformMatrix3x4& a, const TransformMatrix3x4& b)
{
    return a.row0.x == b.row0.x && a.row0.y == b.row0.y && a.row0.z == b.row0.z && a.row0.w == b.row0.w
        && a.row1.x == b.row1.x && a.row1.y == b.row1.y && a.row1.z == b.row1.z && a.row1.w == b.row1.w
        && a.row2.x == b.row2.x && a.row2.y == b.row2.y && a.row2.z == b.row2.z && a.row2.w == b.row2.w;
}

static __forceinline__ __device__ float2 compute_object_motion_vector(
    const float2 pixelCenter, const float3 worldPos, const int instanceId, const float4x4 prevMVP, const uint2 dim
)
{
    if (instanceId < 0 || params.instanceTransformsAddress == 0ull || params.prevInstanceTransformsAddress == 0ull)
        return compute_camera_motion_vector(pixelCenter, make_float4(worldPos.x, worldPos.y, worldPos.z, 1.0f), prevMVP, dim);

    const unsigned int iid = static_cast<unsigned int>(instanceId);
    if (iid >= params.instanceCount)
        return compute_camera_motion_vector(pixelCenter, make_float4(worldPos.x, worldPos.y, worldPos.z, 1.0f), prevMVP, dim);

    const TransformMatrix3x4* instanceTransforms
        = reinterpret_cast<const TransformMatrix3x4*>(params.instanceTransformsAddress);
    const TransformMatrix3x4* prevInstanceTransforms
        = reinterpret_cast<const TransformMatrix3x4*>(params.prevInstanceTransformsAddress);
    const TransformMatrix3x4 currT = instanceTransforms[iid];
    const TransformMatrix3x4 prevT = prevInstanceTransforms[iid];

    float3 prevWorldPos;
    if (transforms_equal(currT, prevT))
        prevWorldPos = worldPos;
    else
        prevWorldPos = transformPoint(prevT, inverse_transform_point(currT, worldPos));

    float4 oldPos = mul_cm(prevMVP, make_float4(prevWorldPos.x, prevWorldPos.y, prevWorldPos.z, 1.0f));
    oldPos.x /= oldPos.w;
    oldPos.y /= oldPos.w;
    oldPos.x = (oldPos.x * 0.5f + 0.5f) * float(dim.x);
    oldPos.y = (oldPos.y * 0.5f + 0.5f) * float(dim.y);
    return make_float2(oldPos.x - pixelCenter.x, oldPos.y - pixelCenter.y);
}

static __forceinline__ __device__ float2 compute_deformable_motion_vector(
    const float2 pixelCenter, const float3 prevLocalPos, const int instanceId, const float4x4 prevMVP, const uint2 dim
)
{
    if (instanceId < 0 || params.prevInstanceTransformsAddress == 0ull)
        return make_float2(0.0f, 0.0f);

    const unsigned int iid = static_cast<unsigned int>(instanceId);
    if (iid >= params.instanceCount)
        return make_float2(0.0f, 0.0f);

    const TransformMatrix3x4* prevInstanceTransforms
        = reinterpret_cast<const TransformMatrix3x4*>(params.prevInstanceTransformsAddress);
    const TransformMatrix3x4 prevT = prevInstanceTransforms[iid];
    const float3 prevWorldPos = transformPoint(prevT, prevLocalPos);

    float4 oldPos = mul_cm(prevMVP, make_float4(prevWorldPos.x, prevWorldPos.y, prevWorldPos.z, 1.0f));
    oldPos.x /= oldPos.w;
    oldPos.y /= oldPos.w;
    oldPos.x = (oldPos.x * 0.5f + 0.5f) * float(dim.x);
    oldPos.y = (oldPos.y * 0.5f + 0.5f) * float(dim.y);
    return make_float2(oldPos.x - pixelCenter.x, oldPos.y - pixelCenter.y);
}

extern "C" __global__ void __raygen__primary()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Camera jitter from frame info - matches C# primary.rgen exactly.
    const float2 pixel = make_float2(float(idx.x), float(idx.y));
    const float2 jitter = make_float2(params.frameInfo.jitter[0], params.frameInfo.jitter[1]);
    const float2 pixelCenter = make_float2(pixel.x + jitter.x + 0.5f, pixel.y + jitter.y + 0.5f);
    const float2 inUV = make_float2(pixelCenter.x / float(dim.x), pixelCenter.y / float(dim.y));
    const float2 d = make_float2(inUV.x * 2.0f - 1.0f, inUV.y * 2.0f - 1.0f);

    // Load matrices.  All are stored in GLM/NumPy memory order (column-major-in-memory).
    // Use mul_cm for matrix-vector products to match GLSL `mat4 * vec4`.
    const float4x4 view = load_mat4_from_array(params.frameInfo.view);
    const float4x4 viewInv = load_mat4_from_array(params.frameInfo.viewInv);
    const float4x4 projInv = load_mat4_from_array(params.frameInfo.projInv);
    const float4x4 prevMVP = load_mat4_from_array(params.frameInfo.prevMVP);

    // Ray origin = camera position = (viewInv * vec4(0,0,0,1)).xyz
    const float4 eyePos4 = mul_cm(viewInv, make_float4(0.0f, 0.0f, 0.0f, 1.0f));
    const float3 eyePos = make_float3(eyePos4.x, eyePos4.y, eyePos4.z);
    float3 origin = eyePos;

    // Ray direction: projInv * ndc -> view space target, then viewInv rotation to world.
    // Matches C#: target = projInv * vec4(d.x, d.y, 0.01, 1.0)
    const float4 target = mul_cm(projInv, make_float4(d.x, d.y, 0.01f, 1.0f));
    const float3x3 viewInv3 = make_float3x3(viewInv);
    float3 direction = mul_cm(viewInv3, normalize(make_float3(target.x, target.y, target.z)));
    const float3 orgDirection = direction;

    const unsigned int pixelIndex = idx.y * dim.x + idx.x;
    const unsigned int rayFlags = OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    unsigned int rng = xxhash32(make_uint3(idx.x, idx.y, params.frameIndex));

    PbrMaterial pbrMat;
    bool hitSky = false;
    bool isPsr = false;
    float psrHitDist = 0.0f;
    float3 psrThroughput = make_float3(1.0f, 1.0f, 1.0f);
    float3 psrDirectRadiance = make_float3(0.0f, 0.0f, 0.0f);
    float psrMirror[9];
    identityMatrix3x3(psrMirror);

    //====================================================================
    // STEP 1 - Find first non-mirror primary hit (PSR loop).
    // Matches C# primary.rgen STEP 1 exactly.
    //====================================================================
    int psrDepth = 0;
    const int MAX_PSR_DEPTH = 5;
    bool foundOpaqueHit = false;
    float3 hitPos = make_float3(0.0f, 0.0f, 0.0f);
    int hitInstanceId = -1;
    unsigned int hitPrimitiveId = 0u;
    float3 hitBarycentrics = make_float3(0.0f, 0.0f, 0.0f);

    do {
        unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0;
        unsigned int p7 = 0, p8 = 0, p9 = 0, p10 = 0, p11 = 0, p12 = 0;
        unsigned int p13 = 0, p14 = 0, p15 = 0, p16 = 0, p17 = 0, p18 = 0;

        optixTrace(
            params.tlas, origin, direction, 0.01f, 1e32f, 0.0f, 0xFF, rayFlags, 0, 1, 0, p0, p1, p2, p3, p4, p5, p6, p7,
            p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18
        );

        const float hitT = __uint_as_float(p0);
        hitSky = (hitT == DLSS_INF_DISTANCE);

        if (hitSky) {
            const float3 skyColor = make_float3(__uint_as_float(p1), __uint_as_float(p2), __uint_as_float(p3));
            psrDirectRadiance = psrDirectRadiance + psrThroughput * skyColor;
            break;
        }

        psrHitDist += hitT;

        // Unpack payload.
        const float3 hitNormal = make_float3(__uint_as_float(p1), __uint_as_float(p2), __uint_as_float(p3));
        const float3 hitTangent = make_float3(__uint_as_float(p4), __uint_as_float(p5), __uint_as_float(p6));
        const float2 hitUv = make_float2(__uint_as_float(p7), __uint_as_float(p8));
        const unsigned int hitMaterialId = p9;
        const float hitBitangentSign = __uint_as_float(p10);
        const float2 hitUv1 = make_float2(__uint_as_float(p17), __uint_as_float(p18));
        const int payloadInstanceId = static_cast<int>(p11);
        const unsigned int payloadPrimitiveId = p13;
        const float3 payloadBarycentrics = make_float3(__uint_as_float(p14), __uint_as_float(p15), __uint_as_float(p16));
        hitPos = origin + direction * hitT;
        hitInstanceId = payloadInstanceId;
        hitPrimitiveId = payloadPrimitiveId;
        hitBarycentrics = payloadBarycentrics;

        // Evaluate material at hit.
        const bool hasPbr = evaluate_pbr_from_payload(
            hitMaterialId, normalize(hitNormal), normalize(hitTangent), hitBitangentSign, hitUv, hitUv1, pbrMat
        );

        origin = offsetRay(hitPos, pbrMat.Ng);

        // Match C# alpha traversal intent: skip transparent hits and keep tracing.
        // We only have resolved opacity here (not full alpha mode), so use a robust
        // stochastic alpha test for partial coverage and a hard reject for near-zero alpha.
        if (hasPbr)
        {
            if (pbrMat.opacity <= 1.0e-4f)
            {
                ++psrDepth;
                continue;
            }
            if (pbrMat.opacity < 1.0f && rand01(rng) > pbrMat.opacity)
            {
                ++psrDepth;
                continue;
            }
        }

        // Non-mirror surface?
        if (!hasPbr || (pbrMat.roughness.x > ((MICROFACET_MIN_ROUGHNESS * MICROFACET_MIN_ROUGHNESS) + 0.001f))
            || pbrMat.metallic < 1.0f || !TEST_FLAG(params.frameInfo.flags, FLAGS_USE_PSR)) {
            foundOpaqueHit = true;
            break;
        }

        // Mirror hit - accumulate PSR chain.
        isPsr = true;
        foundOpaqueHit = true;
        psrDirectRadiance = psrDirectRadiance + psrThroughput * pbrMat.emissive;

        {
            BsdfSampleData specBsdfSample;
            specBsdfSample.xi = make_float3(rand01(rng), rand01(rng), rand01(rng));
            specBsdfSample.k1 = -direction;
            specBsdfSample.pdf = 0.0f;
            specBsdfSample.bsdf_over_pdf = make_float3(0.0f, 0.0f, 0.0f);
            specBsdfSample.event_type = BSDF_EVENT_ABSORB;
            bsdfSample(specBsdfSample, pbrMat);

            if (specBsdfSample.event_type != BSDF_EVENT_GLOSSY_REFLECTION)
                break;
            if (any_isnan(specBsdfSample.bsdf_over_pdf) || any_isinf(specBsdfSample.bsdf_over_pdf))
                break;

            psrThroughput = psrThroughput * specBsdfSample.bsdf_over_pdf;

            float mirrorMat[9];
            buildMirrorMatrix(pbrMat.N, mirrorMat);
            float tmp[9];
            multiplyMatrix3x3(psrMirror, mirrorMat, tmp);
            for (int i = 0; i < 9; ++i)
                psrMirror[i] = tmp[i];

            direction = normalize(specBsdfSample.k2);
        }

        ++psrDepth;
    } while (psrDepth < MAX_PSR_DEPTH);

    // Virtual origin for PSR depth computation (matches C#).
    const float3 virtualOrigin = eyePos + orgDirection * psrHitDist;
    const float viewDepth = -mul_cm(view, make_float4(virtualOrigin.x, virtualOrigin.y, virtualOrigin.z, 1.0f)).z;

    //====================================================================
    // Auxiliary DLSS-style outputs.
    //====================================================================
    float auxViewZ = DLSS_INF_DISTANCE;
    float2 auxMotion = make_float2(0.0f, 0.0f);
    float4 auxNormalRoughness = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 auxDiffuseAlbedo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 auxSpecularAlbedo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float auxSpecHitDist = 0.0f;

    // Early out when hitting sky (even via mirrors) - matches C# exactly.
    if (hitSky) {
        const float3 skyGuide = reinhardMax(psrDirectRadiance);
        auxDiffuseAlbedo = make_float4(skyGuide.x, skyGuide.y, skyGuide.z, 0.0f);

        float4 motionOrigin;
        if (!isPsr) {
            auxViewZ = DLSS_INF_DISTANCE;
            motionOrigin = make_float4(orgDirection.x, orgDirection.y, orgDirection.z, 0.0f);
        } else {
            auxViewZ = viewDepth;
            motionOrigin = make_float4(virtualOrigin.x, virtualOrigin.y, virtualOrigin.z, 1.0f);
        }
        auxMotion = compute_camera_motion_vector(pixelCenter, motionOrigin, prevMVP, make_uint2(dim.x, dim.y));

        float4* colorBuffer = reinterpret_cast<float4*>(params.colorOutput);
        colorBuffer[pixelIndex] = make_float4(psrDirectRadiance.x, psrDirectRadiance.y, psrDirectRadiance.z, 1.0f);
        if (params.normalRoughnessOutput != 0ull)
            reinterpret_cast<float4*>(params.normalRoughnessOutput)[pixelIndex] = auxNormalRoughness;
        if (params.motionVectorOutput != 0ull)
            reinterpret_cast<float2*>(params.motionVectorOutput)[pixelIndex] = auxMotion;
        if (params.depthOutput != 0ull)
            reinterpret_cast<float*>(params.depthOutput)[pixelIndex] = auxViewZ;
        if (params.diffuseAlbedoOutput != 0ull)
            reinterpret_cast<float4*>(params.diffuseAlbedoOutput)[pixelIndex] = auxDiffuseAlbedo;
        if (params.specularAlbedoOutput != 0ull)
            reinterpret_cast<float4*>(params.specularAlbedoOutput)[pixelIndex] = auxSpecularAlbedo;
        if (params.specHitDistOutput != 0ull)
            reinterpret_cast<float*>(params.specHitDistOutput)[pixelIndex] = auxSpecHitDist;
        return;
    }

    // Handle case where no opaque surface was found.
    if (!foundOpaqueHit) {
        float4* colorBuffer = reinterpret_cast<float4*>(params.colorOutput);
        colorBuffer[pixelIndex] = make_float4(psrDirectRadiance.x, psrDirectRadiance.y, psrDirectRadiance.z, 1.0f);
        if (params.normalRoughnessOutput != 0ull)
            reinterpret_cast<float4*>(params.normalRoughnessOutput)[pixelIndex] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (params.motionVectorOutput != 0ull)
            reinterpret_cast<float2*>(params.motionVectorOutput)[pixelIndex] = make_float2(0.0f, 0.0f);
        if (params.depthOutput != 0ull)
            reinterpret_cast<float*>(params.depthOutput)[pixelIndex] = DLSS_INF_DISTANCE;
        if (params.diffuseAlbedoOutput != 0ull)
            reinterpret_cast<float4*>(params.diffuseAlbedoOutput)[pixelIndex] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (params.specularAlbedoOutput != 0ull)
            reinterpret_cast<float4*>(params.specularAlbedoOutput)[pixelIndex] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (params.specHitDistOutput != 0ull)
            reinterpret_cast<float*>(params.specHitDistOutput)[pixelIndex] = 0.0f;
        return;
    }

    // ViewZ buffer.
    auxViewZ = viewDepth;

    // Normal/Roughness buffer - transform through PSR mirror chain.
    {
        const float3 worldNormal = applyMatrix3x3(psrMirror, pbrMat.N);
        auxNormalRoughness
            = make_float4(worldNormal.x, worldNormal.y, worldNormal.z, sqrtf(fmaxf(pbrMat.roughness.x, 0.0f)));
    }

    // Tint material by accumulated PSR mirror throughput.
    pbrMat.baseColor = pbrMat.baseColor * psrThroughput;
    pbrMat.specularColor = pbrMat.specularColor * psrThroughput;
    pbrMat.emissive = pbrMat.emissive * psrThroughput + psrDirectRadiance;

    // Motion Vector Buffer - matches C# exactly.
    {
        float2 motionVec;
        if (isPsr) {
            motionVec = compute_camera_motion_vector(
                pixelCenter, make_float4(virtualOrigin.x, virtualOrigin.y, virtualOrigin.z, 1.0f), prevMVP,
                make_uint2(dim.x, dim.y)
            );
        } else {
            bool usedObjectMotion = false;
            if (hitInstanceId >= 0 && params.instanceRenderPrimIdAddress != 0ull && params.renderPrimitiveAddress != 0ull
                && static_cast<unsigned int>(hitInstanceId) < params.instanceCount)
            {
                const unsigned int* instanceRenderPrimIds
                    = reinterpret_cast<const unsigned int*>(params.instanceRenderPrimIdAddress);
                const unsigned int renderPrimId = instanceRenderPrimIds[static_cast<unsigned int>(hitInstanceId)];
                if (renderPrimId < params.renderPrimCount)
                {
                    const RenderPrimitive* renderPrims
                        = reinterpret_cast<const RenderPrimitive*>(params.renderPrimitiveAddress);
                    const RenderPrimitive renderPrim = renderPrims[renderPrimId];

                    if (hasVertexPrevPosition(renderPrim))
                    {
                        const uint3 tri = getTriangleIndices(renderPrim, hitPrimitiveId);
                        const float3 prevLocalPos = getInterpolatedVertexPrevPosition(renderPrim, tri, hitBarycentrics);
                        motionVec = compute_deformable_motion_vector(
                            pixelCenter, prevLocalPos, hitInstanceId, prevMVP, make_uint2(dim.x, dim.y)
                        );
                    }
                    else
                    {
                        motionVec = compute_object_motion_vector(
                            pixelCenter, hitPos, hitInstanceId, prevMVP, make_uint2(dim.x, dim.y)
                        );
                    }
                    usedObjectMotion = true;
                }
            }

            if (!usedObjectMotion)
            {
                motionVec = compute_camera_motion_vector(
                    pixelCenter, make_float4(hitPos.x, hitPos.y, hitPos.z, 1.0f), prevMVP, make_uint2(dim.x, dim.y)
                );
            }
        }
        auxMotion = motionVec;
    }

    // BaseColor/Metalness buffer.
    auxDiffuseAlbedo = make_float4(pbrMat.baseColor.x, pbrMat.baseColor.y, pbrMat.baseColor.z, pbrMat.metallic);

    // Transform eye vector into "virtual world" for PSR surfaces.
    const float3 toEye = -direction;

    //====================================================================
    // STEP 2 - Direct light contribution at hit position.
    // Matches C# primary.rgen HdrContrib.
    //====================================================================
    float3 hdrRadiance = make_float3(0.0f, 0.0f, 0.0f);
    {
        float3 dirToLight = make_float3(0.0f, 1.0f, 0.0f);
        float3 lightRadiance = make_float3(0.0f, 0.0f, 0.0f);
        float lightPdf = 0.0f;

        if (TEST_FLAG(params.frameInfo.flags, FLAGS_ENVMAP_SKY)) {
            const PhysicalSkyParameters sky = sky_params_from_launch();
            const float2 lightXi = make_float2(rand01(rng), rand01(rng));
            const SkySamplingResult skySample = samplePhysicalSky(sky, lightXi);
            dirToLight = skySample.direction;
            lightRadiance = skySample.radiance
                * make_float3(params.frameInfo.envIntensity[0], params.frameInfo.envIntensity[1],
                              params.frameInfo.envIntensity[2]);
            lightPdf = skySample.pdf;
        } else {
            if (!sample_environment_importance(rng, dirToLight, lightRadiance, lightPdf)) {
                dirToLight = sample_uniform_sphere(rng);
                lightRadiance = eval_environment(dirToLight);
                lightPdf = 1.0f / (4.0f * M_PI);
            }
        }

        if (lightPdf > 1.0e-6f && dot(dirToLight, pbrMat.N) > 0.0f) {
            BsdfEvaluateData evalData;
            evalData.k1 = toEye;
            evalData.k2 = dirToLight;
            evalData.xi = make_float3(rand01(rng), rand01(rng), rand01(rng));
            evalData.bsdf_diffuse = make_float3(0.0f, 0.0f, 0.0f);
            evalData.bsdf_glossy = make_float3(0.0f, 0.0f, 0.0f);
            evalData.pdf = 0.0f;
            bsdfEvaluate(evalData, pbrMat);

            if (evalData.pdf > 1.0e-6f) {
                unsigned int visibility = 0u;
                const float3 shadowOrigin = offsetRay(hitPos, pbrMat.Ng);
                optixTrace(
                    params.tlas, shadowOrigin, dirToLight, 0.001f, DLSS_INF_DISTANCE, 0.0f, 0xFF,
                    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
                        | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                    SBTOFFSET_SECONDARY, 1, MISSINDEX_SECONDARY, visibility
                );
                if (visibility == 1u) {
                    const float misWeight = powerHeuristic(lightPdf, evalData.pdf);
                    const float3 bsdfSum = evalData.bsdf_diffuse + evalData.bsdf_glossy;
                    hdrRadiance = bsdfSum * (lightRadiance / lightPdf) * misWeight;
                }
            }
        }

        if (any_isnan(hdrRadiance) || any_isinf(hdrRadiance))
            hdrRadiance = make_float3(0.0f, 0.0f, 0.0f);
    }

    float3 directLum = psrDirectRadiance + pbrMat.emissive;

    //====================================================================
    // STEP 3 - Indirect contribution (path tracing from PSR surface).
    // Matches C# primary.rgen STEP 3.
    //====================================================================
    float3 radiance = hdrRadiance;
    float pathLength = 0.0f;

    // STEP 3.1 - Sample BSDF direction.
    BsdfSampleData sampleData;
    sampleData.xi = make_float3(rand01(rng), rand01(rng), rand01(rng));
    sampleData.k1 = toEye;
    sampleData.pdf = 0.0f;
    sampleData.bsdf_over_pdf = make_float3(0.0f, 0.0f, 0.0f);
    sampleData.event_type = BSDF_EVENT_ABSORB;
    bsdfSample(sampleData, pbrMat);

    if (any_isnan(sampleData.bsdf_over_pdf) || any_isinf(sampleData.bsdf_over_pdf))
        sampleData.event_type = BSDF_EVENT_ABSORB;

    if (sampleData.event_type != BSDF_EVENT_ABSORB) {
        // STEP 3.2 - Set up secondary ray.
        float3 secOrigin = origin;
        float3 secDirection = sampleData.k2;
        float3 throughput = sampleData.bsdf_over_pdf;
        float bsdfPdf = fmaxf(sampleData.pdf, 0.0001f);

        // STEP 3.3 - Trace secondary bounces.
        const int maxDepth = int((params.maxBounces > 0u) ? params.maxBounces : 1u);
        for (int depth = 1; depth < maxDepth; ++depth) {
            unsigned int q0 = 0, q1 = 0, q2 = 0, q3 = 0, q4 = 0, q5 = 0, q6 = 0;
            unsigned int q7 = 0, q8 = 0, q9 = 0, q10 = 0, q11 = 0, q12 = 0;
            unsigned int q13 = 0, q14 = 0, q15 = 0, q16 = 0, q17 = 0, q18 = 0;

            optixTrace(
                params.tlas, secOrigin, secDirection, 0.001f, 1e16f, 0.0f, 0xFF, rayFlags, 0, 1, 0, q0, q1, q2, q3, q4,
                q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18
            );

            const float t = __uint_as_float(q0);
            const bool miss = (t >= DLSS_INF_DISTANCE * 0.99f);

            // Specular hit distance (first secondary segment, glossy reflection).
            if (depth == 1 && sampleData.event_type == BSDF_EVENT_GLOSSY_REFLECTION)
                pathLength = fabsf(t);

            if (miss) {
                float3 envColor;
                float envPdf;
                if (TEST_FLAG(params.frameInfo.flags, FLAGS_ENVMAP_SKY)) {
                    const PhysicalSkyParameters sky = sky_params_from_launch();
                    envColor = evalPhysicalSky(sky, secDirection)
                        * make_float3(params.frameInfo.envIntensity[0], params.frameInfo.envIntensity[1],
                                      params.frameInfo.envIntensity[2]);
                    envPdf = samplePhysicalSkyPDF(sky, secDirection);
                } else {
                    envColor = eval_environment(secDirection);
                    envPdf = 1.0f / (4.0f * M_PIf);
                }
                float misWeight = powerHeuristic(bsdfPdf, fmaxf(envPdf, 0.0001f));
                if (isnan(misWeight) || isinf(misWeight))
                    misWeight = 0.0f;
                radiance = radiance + throughput * envColor * misWeight;
                break;
            }

            // Unpack secondary hit payload.
            const float3 secNormal
                = normalize(make_float3(__uint_as_float(q1), __uint_as_float(q2), __uint_as_float(q3)));
            const float3 secTangent = make_float3(__uint_as_float(q4), __uint_as_float(q5), __uint_as_float(q6));
            const float2 secUv = make_float2(__uint_as_float(q7), __uint_as_float(q8));
            const float2 secUv1 = make_float2(__uint_as_float(q17), __uint_as_float(q18));
            const unsigned int secMatId = q9;
            const float secBitangentSign = __uint_as_float(q10);

            PbrMaterial secPbrMat;
            const bool secHasPbr = evaluate_pbr_from_payload(
                secMatId, secNormal, secTangent, secBitangentSign, secUv, secUv1, secPbrMat
            );

            const float3 secHitPos = secOrigin + secDirection * t;
            radiance = radiance + throughput * secPbrMat.emissive;

            // Direct lighting at secondary hit.
            {
                float3 secDirToLight = make_float3(0.0f, 1.0f, 0.0f);
                float3 secLightRadiance = make_float3(0.0f, 0.0f, 0.0f);
                float secLightPdf = 0.0f;

                if (TEST_FLAG(params.frameInfo.flags, FLAGS_ENVMAP_SKY)) {
                    const PhysicalSkyParameters sky = sky_params_from_launch();
                    const float2 xi2 = make_float2(rand01(rng), rand01(rng));
                    const SkySamplingResult skySample = samplePhysicalSky(sky, xi2);
                    secDirToLight = skySample.direction;
                    secLightRadiance = skySample.radiance
                        * make_float3(params.frameInfo.envIntensity[0], params.frameInfo.envIntensity[1],
                                      params.frameInfo.envIntensity[2]);
                    secLightPdf = skySample.pdf;
                } else {
                    if (!sample_environment_importance(rng, secDirToLight, secLightRadiance, secLightPdf)) {
                        secDirToLight = sample_uniform_sphere(rng);
                        secLightRadiance = eval_environment(secDirToLight);
                        secLightPdf = 1.0f / (4.0f * M_PI);
                    }
                }

                if (secLightPdf > 1.0e-6f && dot(secDirToLight, secPbrMat.N) > 0.0f) {
                    BsdfEvaluateData secEval;
                    secEval.k1 = -secDirection;
                    secEval.k2 = secDirToLight;
                    secEval.xi = make_float3(rand01(rng), rand01(rng), rand01(rng));
                    secEval.bsdf_diffuse = make_float3(0.0f, 0.0f, 0.0f);
                    secEval.bsdf_glossy = make_float3(0.0f, 0.0f, 0.0f);
                    secEval.pdf = 0.0f;
                    bsdfEvaluate(secEval, secPbrMat);

                    if (secEval.pdf > 1.0e-6f) {
                        unsigned int vis = 0u;
                        optixTrace(
                            params.tlas, secHitPos + secPbrMat.Ng * 0.001f, secDirToLight, 0.001f, DLSS_INF_DISTANCE,
                            0.0f, 0xFF,
                            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
                                | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                            SBTOFFSET_SECONDARY, 1, MISSINDEX_SECONDARY, vis
                        );
                        if (vis == 1u) {
                            const float mw = powerHeuristic(secLightPdf, secEval.pdf);
                            radiance = radiance
                                + throughput * (secEval.bsdf_diffuse + secEval.bsdf_glossy)
                                    * (secLightRadiance / secLightPdf) * mw;
                        }
                    }
                }
            }

            // Sample next bounce direction.
            BsdfSampleData secSample;
            secSample.k1 = -secDirection;
            secSample.xi = make_float3(rand01(rng), rand01(rng), rand01(rng));
            secSample.pdf = 0.0f;
            secSample.bsdf_over_pdf = make_float3(0.0f, 0.0f, 0.0f);
            secSample.event_type = BSDF_EVENT_ABSORB;
            bsdfSample(secSample, secPbrMat);

            if (secSample.event_type == BSDF_EVENT_ABSORB || any_isnan(secSample.bsdf_over_pdf)
                || any_isinf(secSample.bsdf_over_pdf))
                break;

            throughput = throughput * secSample.bsdf_over_pdf;
            if (any_isnan(throughput) || any_isinf(throughput))
                break;

            // Russian roulette.
            if (depth >= 2) {
                const float maxComp = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
                const float rrProb = fminf(fmaxf(maxComp, 0.05f), 0.99f);
                if (rand01(rng) > rrProb)
                    break;
                throughput = throughput / rrProb;
            }

            secOrigin = offsetRay(secHitPos, secPbrMat.Ng);
            secDirection = normalize(secSample.k2);
            bsdfPdf = fmaxf(secSample.pdf, 0.0001f);
        }
    }

    // Specular albedo (pre-integrated environment term).
    float3 Fenv = make_float3(0.0f, 0.0f, 0.0f);
    if (sampleData.event_type != BSDF_EVENT_DIFFUSE) {
        const float VdotN = fmaxf(dot(toEye, pbrMat.N), 0.0f);
        Fenv = EnvironmentTerm_Rtg(pbrMat.specularColor, VdotN, pbrMat.roughness.x);
    }
    auxSpecularAlbedo = make_float4(Fenv.x, Fenv.y, Fenv.z, 0.0f);
    auxSpecHitDist = pathLength;

    // Guard against NaN/Inf.
    if (any_isnan(radiance) || any_isinf(radiance))
        radiance = make_float3(0.0f, 0.0f, 0.0f);
    if (any_isnan(directLum) || any_isinf(directLum))
        directLum = make_float3(0.0f, 0.0f, 0.0f);

    float3 color = radiance + directLum;

    //====================================================================
    // Debug visualization modes.
    //====================================================================
    if (params.outputMode == OUTPUT_MODE_DEPTH) {
        const float depthVis = (auxViewZ >= DLSS_INF_DISTANCE) ? 0.0f : expf(-0.075f * auxViewZ);
        color = make_float3(depthVis, depthVis, depthVis);
    } else if (params.outputMode == OUTPUT_MODE_NORMAL) {
        const float3 n = normalize(make_float3(auxNormalRoughness.x, auxNormalRoughness.y, auxNormalRoughness.z));
        color = make_float3(0.5f * (n.x + 1.0f), 0.5f * (n.y + 1.0f), 0.5f * (n.z + 1.0f));
    } else if (params.outputMode == OUTPUT_MODE_ROUGHNESS) {
        const float r = auxNormalRoughness.w;
        color = make_float3(r, r, r);
    } else if (params.outputMode == OUTPUT_MODE_DIFFUSE) {
        color = make_float3(auxDiffuseAlbedo.x, auxDiffuseAlbedo.y, auxDiffuseAlbedo.z);
    } else if (params.outputMode == OUTPUT_MODE_SPECULAR) {
        const float m = auxDiffuseAlbedo.w;
        color = make_float3(m, m, m);
    } else if (params.outputMode == OUTPUT_MODE_MOTION) {
        color = make_float3(
            fminf(fmaxf(0.5f + auxMotion.x * 8.0f, 0.0f), 1.0f), fminf(fmaxf(0.5f + auxMotion.y * 8.0f, 0.0f), 1.0f),
            0.0f
        );
    } else if (params.outputMode == OUTPUT_MODE_SPEC_HITDIST) {
        const float sd = fmaxf(auxSpecHitDist, 0.0f);
        const float vis = 1.0f - expf(-0.05f * sd);
        color = make_float3(vis, vis, vis);
    }

    //====================================================================
    // Write all output buffers.
    //====================================================================
    float4* colorBuffer = reinterpret_cast<float4*>(params.colorOutput);
    colorBuffer[pixelIndex] = make_float4(color.x, color.y, color.z, pbrMat.opacity);

    if (params.normalRoughnessOutput != 0ull)
        reinterpret_cast<float4*>(params.normalRoughnessOutput)[pixelIndex] = auxNormalRoughness;
    if (params.motionVectorOutput != 0ull)
        reinterpret_cast<float2*>(params.motionVectorOutput)[pixelIndex] = auxMotion;
    if (params.depthOutput != 0ull)
        reinterpret_cast<float*>(params.depthOutput)[pixelIndex] = auxViewZ;
    if (params.diffuseAlbedoOutput != 0ull)
        reinterpret_cast<float4*>(params.diffuseAlbedoOutput)[pixelIndex] = auxDiffuseAlbedo;
    if (params.specularAlbedoOutput != 0ull)
        reinterpret_cast<float4*>(params.specularAlbedoOutput)[pixelIndex] = auxSpecularAlbedo;
    if (params.specHitDistOutput != 0ull)
        reinterpret_cast<float*>(params.specHitDistOutput)[pixelIndex] = auxSpecHitDist;
}

extern "C" __global__ void __miss__primary()
{
    const float3 skyColor = eval_environment(optixGetWorldRayDirection());
    // Match RayPayload layout from ray_common.h / primary_rchit.h.
    optixSetPayload_0(__float_as_uint(DLSS_INF_DISTANCE));
    optixSetPayload_1(__float_as_uint(skyColor.x));
    optixSetPayload_2(__float_as_uint(skyColor.y));
    optixSetPayload_3(__float_as_uint(skyColor.z));
}

extern "C" __global__ void __closesthit__primary()
{
    RayPayload payload;
    payload.hitT = optixGetRayTmax();
    const float2 attrib = optixGetTriangleBarycentrics();
    const float3 bary = make_float3(1.0f - attrib.x - attrib.y, attrib.x, attrib.y);
    const unsigned int instanceId = optixGetInstanceId();
    const unsigned int primitiveId = optixGetPrimitiveIndex();

    payload.instanceId = (int)instanceId;
    payload.meshId = instanceId;
    payload.primitiveId = primitiveId;
    payload.barycentrics = bary;

    if (params.instanceMaterialIdAddress != 0ull && instanceId < params.instanceCount) {
        // Read actual mesh attributes from scene buffers for smoother normals.
        float3 worldNormal = normalize(make_float3(bary.x, bary.y, bary.z));
        float3 worldTangent = make_float3(1.0f, 0.0f, 0.0f);
        float bitangentSign = 1.0f;
        float2 uv = make_float2(bary.x, bary.y);
        float2 uv1 = uv;

        if (params.instanceRenderPrimIdAddress != 0ull && params.renderPrimitiveAddress != 0ull) {
            const unsigned int* instanceRenderPrimIds
                = reinterpret_cast<const unsigned int*>(params.instanceRenderPrimIdAddress);
            const unsigned int renderPrimId = instanceRenderPrimIds[instanceId];
            if (renderPrimId < params.renderPrimCount) {
                const RenderPrimitive* renderPrims
                    = reinterpret_cast<const RenderPrimitive*>(params.renderPrimitiveAddress);
                const RenderPrimitive renderPrim = renderPrims[renderPrimId];

                const uint3 tri = getTriangleIndices(renderPrim, primitiveId);
                const float3 nLocal = getInterpolatedVertexNormal(renderPrim, tri, bary);
                if (dot(nLocal, nLocal) > 1.0e-12f) {
                    worldNormal = normalize(optixTransformNormalFromObjectToWorldSpace(nLocal));
                } else {
                    // Fallback to geometric normal when vertex normals are unavailable/degenerate.
                    const float3 p0 = getVertexPosition(renderPrim, tri.x);
                    const float3 p1 = getVertexPosition(renderPrim, tri.y);
                    const float3 p2 = getVertexPosition(renderPrim, tri.z);
                    const float3 ngLocal = normalize(cross(p1 - p0, p2 - p0));
                    if (dot(ngLocal, ngLocal) > 1.0e-12f)
                        worldNormal = normalize(optixTransformNormalFromObjectToWorldSpace(ngLocal));
                }

                const float4 tLocal4 = getInterpolatedVertexTangent(renderPrim, tri, bary);
                if (dot(make_float3(tLocal4.x, tLocal4.y, tLocal4.z), make_float3(tLocal4.x, tLocal4.y, tLocal4.z))
                    > 1.0e-12f)
                    worldTangent = normalize(
                        optixTransformVectorFromObjectToWorldSpace(make_float3(tLocal4.x, tLocal4.y, tLocal4.z))
                    );
                else {
                    const float3 up = (fabsf(worldNormal.z) < 0.999f) ? make_float3(0.0f, 0.0f, 1.0f)
                                                                      : make_float3(0.0f, 1.0f, 0.0f);
                    worldTangent = normalize(cross(up, worldNormal));
                }
                bitangentSign = (tLocal4.w == 0.0f) ? 1.0f : tLocal4.w;
                uv = getInterpolatedVertexTexCoord0(renderPrim, tri, bary);
                uv1 = getInterpolatedVertexTexCoord1(renderPrim, tri, bary);
            }
        }

        // Face-forward hit normal to the incoming ray direction to avoid
        // backside BSDF events on reflective surfaces.
        const float3 toEye = -normalize(optixGetWorldRayDirection());
        if (dot(worldNormal, toEye) < 0.0f) {
            worldNormal = -worldNormal;
            bitangentSign = -bitangentSign;
        }

        payload.normal_envmapRadiance = worldNormal;
        payload.tangent = worldTangent;
        payload.bitangentSign = bitangentSign;
        payload.uv = uv;
        payload.uv1 = uv1;
        const unsigned int* instanceMats = reinterpret_cast<const unsigned int*>(params.instanceMaterialIdAddress);
        payload.materialId = instanceMats[instanceId];
    } else {
        payload.normal_envmapRadiance = normalize(make_float3(bary.x, bary.y, bary.z));
        payload.tangent = make_float3(1.0f, 0.0f, 0.0f);
        payload.bitangentSign = 1.0f;
        payload.uv = make_float2(bary.x, bary.y);
        payload.uv1 = payload.uv;
        payload.materialId = 0u;
    }
    setRayPayload(payload);
}

extern "C" __global__ void __miss__secondary()
{
    // Visible path for shadow ray.
    optixSetPayload_0(1u);
}

extern "C" __global__ void __closesthit__secondary()
{
    // Occluded path for shadow ray.
    optixSetPayload_0(0u);
}

extern "C" __global__ void __anyhit__secondary()
{
    // Any hit means blocked for visibility rays.
    optixSetPayload_0(0u);
    optixTerminateRay();
}

#endif  // OPTIX_PROGRAMS_H
