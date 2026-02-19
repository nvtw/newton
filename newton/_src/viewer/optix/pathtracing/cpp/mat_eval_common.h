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

// Material evaluation and HDR sampling - CUDA/OptiX version
// Converted from Vulkan GLSL mat_eval_common.glsl
// NOTE: This file is designed to be inlined into a single CUDA source.
//       Include scene_common.h and pbr_common.h content before this file.
//       Must be included AFTER texture/sampler declarations in launch params.

#ifndef MAT_EVAL_COMMON_H
#define MAT_EVAL_COMMON_H

//=============================================================================
// Material evaluation (from nvvkhl/shaders/pbr_mat_eval.h)
//=============================================================================
#ifndef MAT_EVAL_H
#define MAT_EVAL_H 1

struct MeshState {
    float3 N;
    float3 T;
    float3 B;
    float3 Ng;
    float2 tc[2];
    bool isInside;
};

#define MICROFACET_MIN_ROUGHNESS 0.0014142f

struct RuntimeTextureDesc {
    unsigned int offset;
    unsigned int width;
    unsigned int height;
    unsigned int _pad;
};

static __forceinline__ __device__ float4 sample_texture_rgba(unsigned int texIndex, float2 uv)
{
    if (params.textureDescAddress == 0ull || params.textureDataAddress == 0ull || texIndex >= params.textureCount)
        return make_float4(1.0f, 1.0f, 1.0f, 1.0f);

    const RuntimeTextureDesc* descs = reinterpret_cast<const RuntimeTextureDesc*>(params.textureDescAddress);
    const RuntimeTextureDesc desc = descs[texIndex];
    if (desc.width == 0u || desc.height == 0u)
        return make_float4(1.0f, 1.0f, 1.0f, 1.0f);

    const float u = uv.x - floorf(uv.x);
    const float v = uv.y - floorf(uv.y);
    const float fx = u * float(desc.width - 1u);
    const float fy = v * float(desc.height - 1u);
    const unsigned int x0 = min((unsigned int)fx, desc.width - 1u);
    const unsigned int y0 = min((unsigned int)fy, desc.height - 1u);
    const unsigned int x1 = min(x0 + 1u, desc.width - 1u);
    const unsigned int y1 = min(y0 + 1u, desc.height - 1u);
    const float tx = fx - float(x0);
    const float ty = fy - float(y0);

    const float4* texels = reinterpret_cast<const float4*>(params.textureDataAddress) + desc.offset;
    const float4 c00 = texels[y0 * desc.width + x0];
    const float4 c10 = texels[y0 * desc.width + x1];
    const float4 c01 = texels[y1 * desc.width + x0];
    const float4 c11 = texels[y1 * desc.width + x1];
    const float4 cx0 = c00 * (1.0f - tx) + c10 * tx;
    const float4 cx1 = c01 * (1.0f - tx) + c11 * tx;
    return cx0 * (1.0f - ty) + cx1 * ty;
}

static __forceinline__ __device__ float4
getTexture(const GltfTextureInfo& tinfo, const float2 tc[2], cudaTextureObject_t* textureObjects)
{
    if (tinfo.index < 0)
        return make_float4(1.0f, 1.0f, 1.0f, 1.0f);

    float2 texCoord = transformUV(tinfo.uvTransform, tc[tinfo.texCoord]);
    return sample_texture_rgba((unsigned int)tinfo.index, texCoord);
}

static __forceinline__ __device__ bool isTexturePresent(const GltfTextureInfo& tinfo) { return tinfo.index > -1; }

static __forceinline__ __device__ float3
convertSGToMR(float3 diffuseColor, float3 specularColor, float glossiness, float& metallic, float2& roughness)
{
    const float dielectricSpecular = 0.04f;
    float specularIntensity = fmaxf(specularColor.x, fmaxf(specularColor.y, specularColor.z));
    float isMetal = (specularIntensity > dielectricSpecular + 0.05f)
        ? 1.0f
        : ((specularIntensity > dielectricSpecular + 0.01f) ? (specularIntensity - dielectricSpecular - 0.01f) / 0.04f
                                                            : 0.0f);
    metallic = isMetal;
    float3 baseColor;
    if (metallic > 0.0f) {
        baseColor = specularColor;
    } else {
        baseColor = diffuseColor / (1.0f - dielectricSpecular * (1.0f - metallic));
        baseColor = clamp(baseColor, 0.0f, 1.0f);
    }
    float r = 1.0f - glossiness;
    roughness = make_float2(r * r, r * r);
    return baseColor;
}

// Simplified material evaluation without texture sampling
// For full texture support, use the version with textureObjects parameter
static __forceinline__ __device__ PbrMaterial
evaluateMaterial(const GltfShadeMaterial& material, float3 normal, float3 tangent, float3 bitangent, float2 texCoord)
{
    PbrMaterial pbrMat = defaultPbrMaterial();

    // Base color
    pbrMat.baseColor
        = make_float3(material.pbrBaseColorFactor.x, material.pbrBaseColorFactor.y, material.pbrBaseColorFactor.z);
    pbrMat.opacity = material.pbrBaseColorFactor.w;

    // Roughness and metallic
    float roughness = material.pbrRoughnessFactor;
    roughness = fmaxf(roughness, MICROFACET_MIN_ROUGHNESS);
    pbrMat.roughness = make_float2(roughness * roughness, roughness * roughness);
    pbrMat.metallic = fminf(fmaxf(material.pbrMetallicFactor, 0.0f), 1.0f);

    // Normals and tangent frame
    pbrMat.N = normal;
    pbrMat.Ng = normal;
    pbrMat.T = tangent;
    pbrMat.B = bitangent;

    // Emissive
    pbrMat.emissive = fmaxf(material.emissiveFactor, make_float3(0.0f, 0.0f, 0.0f));

    // Specular
    pbrMat.specularColor = material.specularColorFactor;
    pbrMat.specular = material.specularFactor;

    // IOR
    pbrMat.ior1 = 1.0f;
    pbrMat.ior2 = material.ior;

    // Transmission
    pbrMat.transmission = material.transmissionFactor;
    pbrMat.attenuationColor = material.attenuationColor;
    pbrMat.attenuationDistance = material.attenuationDistance;
    pbrMat.isThinWalled = (material.thicknessFactor == 0.0f);

    // Clearcoat
    pbrMat.clearcoat = material.clearcoatFactor;
    pbrMat.clearcoatRoughness = fmaxf(material.clearcoatRoughness, 0.001f);
    pbrMat.Nc = pbrMat.N;

    // Iridescence
    pbrMat.iridescence = material.iridescenceFactor;
    pbrMat.iridescenceIor = material.iridescenceIor;
    pbrMat.iridescenceThickness = material.iridescenceThicknessMaximum;

    // Sheen
    pbrMat.sheenColor = material.sheenColorFactor;
    pbrMat.sheenRoughness = fmaxf(material.sheenRoughnessFactor, MICROFACET_MIN_ROUGHNESS);

    // Occlusion
    pbrMat.occlusion = material.occlusionStrength;

    // Dispersion
    pbrMat.dispersion = material.dispersion;

    // Diffuse transmission
    pbrMat.diffuseTransmissionFactor = material.diffuseTransmissionFactor;
    pbrMat.diffuseTransmissionColor = material.diffuseTransmissionColor;

    return pbrMat;
}

// Full material evaluation with texture sampling
static __forceinline__ __device__ PbrMaterial
evaluateMaterial(const GltfShadeMaterial& material, const MeshState& state, cudaTextureObject_t* textureObjects)
{
    PbrMaterial pbrMat = defaultPbrMaterial();

    if (material.usePbrSpecularGlossiness == 0) {
        float4 baseColor = material.pbrBaseColorFactor;
        if (isTexturePresent(material.pbrBaseColorTexture)) {
            float4 texColor = getTexture(material.pbrBaseColorTexture, state.tc, textureObjects);
            baseColor = make_float4(
                baseColor.x * texColor.x, baseColor.y * texColor.y, baseColor.z * texColor.z, baseColor.w * texColor.w
            );
        }
        pbrMat.baseColor = make_float3(baseColor.x, baseColor.y, baseColor.z);
        pbrMat.opacity = baseColor.w;

        float roughness = material.pbrRoughnessFactor;
        float metallic = material.pbrMetallicFactor;
        if (isTexturePresent(material.pbrMetallicRoughnessTexture)) {
            float4 metallicRoughnessSample = getTexture(material.pbrMetallicRoughnessTexture, state.tc, textureObjects);
            roughness *= metallicRoughnessSample.y;  // Green channel
            metallic *= metallicRoughnessSample.z;  // Blue channel
        }
        roughness = fmaxf(roughness, MICROFACET_MIN_ROUGHNESS);
        pbrMat.roughness = make_float2(roughness * roughness, roughness * roughness);
        pbrMat.metallic = fminf(fmaxf(metallic, 0.0f), 1.0f);
    } else {
        float4 diffuse = material.pbrDiffuseFactor;
        float glossiness = material.pbrGlossinessFactor;
        float3 specular = material.pbrSpecularFactor;
        if (isTexturePresent(material.pbrDiffuseTexture)) {
            float4 texColor = getTexture(material.pbrDiffuseTexture, state.tc, textureObjects);
            diffuse = make_float4(
                diffuse.x * texColor.x, diffuse.y * texColor.y, diffuse.z * texColor.z, diffuse.w * texColor.w
            );
        }
        if (isTexturePresent(material.pbrSpecularGlossinessTexture)) {
            float4 specularGlossinessSample
                = getTexture(material.pbrSpecularGlossinessTexture, state.tc, textureObjects);
            specular = specular
                * make_float3(specularGlossinessSample.x, specularGlossinessSample.y, specularGlossinessSample.z);
            glossiness *= specularGlossinessSample.w;
        }
        pbrMat.baseColor = convertSGToMR(
            make_float3(diffuse.x, diffuse.y, diffuse.z), specular, glossiness, pbrMat.metallic, pbrMat.roughness
        );
        pbrMat.opacity = diffuse.w;
    }

    // Occlusion
    pbrMat.occlusion = material.occlusionStrength;
    if (isTexturePresent(material.occlusionTexture)) {
        float occlusion = getTexture(material.occlusionTexture, state.tc, textureObjects).x;
        pbrMat.occlusion = 1.0f + pbrMat.occlusion * (occlusion - 1.0f);
    }

    // Normals
    pbrMat.N = state.N;
    pbrMat.T = state.T;
    pbrMat.B = state.B;
    pbrMat.Ng = state.Ng;

    if (isTexturePresent(material.normalTexture)) {
        float4 normalSample = getTexture(material.normalTexture, state.tc, textureObjects);
        float3 normal_vector = make_float3(normalSample.x, normalSample.y, normalSample.z);
        normal_vector = normal_vector * 2.0f - make_float3(1.0f, 1.0f, 1.0f);
        normal_vector = normal_vector * make_float3(material.normalTextureScale, material.normalTextureScale, 1.0f);
        // Apply TBN matrix
        pbrMat.N = normalize(state.T * normal_vector.x + state.B * normal_vector.y + state.N * normal_vector.z);
    }

    // Emissive
    pbrMat.emissive = material.emissiveFactor;
    if (isTexturePresent(material.emissiveTexture)) {
        float4 emissiveSample = getTexture(material.emissiveTexture, state.tc, textureObjects);
        pbrMat.emissive = pbrMat.emissive * make_float3(emissiveSample.x, emissiveSample.y, emissiveSample.z);
    }
    pbrMat.emissive = fmaxf(pbrMat.emissive, make_float3(0.0f, 0.0f, 0.0f));

    // Specular
    pbrMat.specularColor = material.specularColorFactor;
    if (isTexturePresent(material.specularColorTexture)) {
        float4 specColorSample = getTexture(material.specularColorTexture, state.tc, textureObjects);
        pbrMat.specularColor
            = pbrMat.specularColor * make_float3(specColorSample.x, specColorSample.y, specColorSample.z);
    }

    pbrMat.specular = material.specularFactor;
    if (isTexturePresent(material.specularTexture)) {
        pbrMat.specular *= getTexture(material.specularTexture, state.tc, textureObjects).w;
    }

    // IOR
    float ior1 = 1.0f;
    float ior2 = material.ior;
    if (state.isInside && (material.thicknessFactor > 0)) {
        ior1 = material.ior;
        ior2 = 1.0f;
    }
    pbrMat.ior1 = ior1;
    pbrMat.ior2 = ior2;

    // Transmission
    pbrMat.transmission = material.transmissionFactor;
    if (isTexturePresent(material.transmissionTexture)) {
        pbrMat.transmission *= getTexture(material.transmissionTexture, state.tc, textureObjects).x;
    }

    pbrMat.attenuationColor = material.attenuationColor;
    pbrMat.attenuationDistance = material.attenuationDistance;
    pbrMat.isThinWalled = (material.thicknessFactor == 0.0f);

    // Clearcoat
    pbrMat.clearcoat = material.clearcoatFactor;
    pbrMat.clearcoatRoughness = material.clearcoatRoughness;
    pbrMat.Nc = pbrMat.N;
    if (isTexturePresent(material.clearcoatTexture)) {
        pbrMat.clearcoat *= getTexture(material.clearcoatTexture, state.tc, textureObjects).x;
    }
    if (isTexturePresent(material.clearcoatRoughnessTexture)) {
        pbrMat.clearcoatRoughness *= getTexture(material.clearcoatRoughnessTexture, state.tc, textureObjects).y;
    }
    pbrMat.clearcoatRoughness = fmaxf(pbrMat.clearcoatRoughness, 0.001f);

    // Iridescence
    float iridescence = material.iridescenceFactor;
    float iridescenceThickness = material.iridescenceThicknessMaximum;
    pbrMat.iridescenceIor = material.iridescenceIor;
    if (isTexturePresent(material.iridescenceTexture)) {
        iridescence *= getTexture(material.iridescenceTexture, state.tc, textureObjects).x;
    }
    if (isTexturePresent(material.iridescenceThicknessTexture)) {
        float t = getTexture(material.iridescenceThicknessTexture, state.tc, textureObjects).y;
        iridescenceThickness = mix(material.iridescenceThicknessMinimum, material.iridescenceThicknessMaximum, t);
    }
    pbrMat.iridescence = (iridescenceThickness > 0.0f) ? iridescence : 0.0f;
    pbrMat.iridescenceThickness = iridescenceThickness;

    // Sheen
    pbrMat.sheenColor = material.sheenColorFactor;
    if (isTexturePresent(material.sheenColorTexture)) {
        float4 sheenSample = getTexture(material.sheenColorTexture, state.tc, textureObjects);
        pbrMat.sheenColor = pbrMat.sheenColor * make_float3(sheenSample.x, sheenSample.y, sheenSample.z);
    }

    pbrMat.sheenRoughness = material.sheenRoughnessFactor;
    if (isTexturePresent(material.sheenRoughnessTexture)) {
        pbrMat.sheenRoughness *= getTexture(material.sheenRoughnessTexture, state.tc, textureObjects).w;
    }
    pbrMat.sheenRoughness = fmaxf(MICROFACET_MIN_ROUGHNESS, pbrMat.sheenRoughness);

    pbrMat.dispersion = material.dispersion;

    // Diffuse transmission
    pbrMat.diffuseTransmissionFactor = material.diffuseTransmissionFactor;
    if (isTexturePresent(material.diffuseTransmissionTexture)) {
        pbrMat.diffuseTransmissionFactor *= getTexture(material.diffuseTransmissionTexture, state.tc, textureObjects).w;
    }
    pbrMat.diffuseTransmissionColor = material.diffuseTransmissionColor;
    if (isTexturePresent(material.diffuseTransmissionColorTexture)) {
        float4 dtcSample = getTexture(material.diffuseTransmissionColorTexture, state.tc, textureObjects);
        pbrMat.diffuseTransmissionColor = make_float3(dtcSample.x, dtcSample.y, dtcSample.z);
    }

    return pbrMat;
}

#endif  // MAT_EVAL_H

//=============================================================================
// HDR environment sampling (from nvvkhl/shaders/hdr_env_sampling.h)
//=============================================================================
#ifndef HDR_ENV_SAMPLING_H
#define HDR_ENV_SAMPLING_H 1

// Environment sampling requires:
// - hdrTexture: cudaTextureObject_t for the HDR environment map
// - envSamplingData: EnvAccel* array for importance sampling
// - width, height: dimensions of the HDR texture

static __forceinline__ __device__ float4 environmentSample(
    cudaTextureObject_t hdrTexture,
    const EnvAccel* envSamplingData,
    unsigned int width,
    unsigned int height,
    float3 randVal,
    float3& toLight
)
{
    float3 xi = randVal;
    unsigned int size = width * height;
    unsigned int idx = min((unsigned int)(xi.x * float(size)), size - 1U);
    EnvAccel sample_data = envSamplingData[idx];
    unsigned int env_idx;
    if (xi.y < sample_data.q) {
        env_idx = idx;
        xi.y /= sample_data.q;
    } else {
        env_idx = sample_data.alias;
        xi.y = (xi.y - sample_data.q) / (1.0f - sample_data.q);
    }
    const unsigned int px = env_idx % width;
    unsigned int py = env_idx / width;
    const float u = float(px + xi.y) / float(width);
    const float phi = u * (2.0f * M_PI) - M_PI;
    float sin_phi = sinf(phi);
    float cos_phi = cosf(phi);
    const float step_theta = M_PI / float(height);
    const float theta0 = float(py) * step_theta;
    const float cos_theta = cosf(theta0) * (1.0f - xi.z) + cosf(theta0 + step_theta) * xi.z;
    const float theta = acosf(cos_theta);
    const float sin_theta = sinf(theta);
    const float v = theta * M_1_OVER_PI;
    toLight = make_float3(cos_phi * sin_theta, cos_theta, sin_phi * sin_theta);
    return tex2D<float4>(hdrTexture, u, v);
}

#endif  // HDR_ENV_SAMPLING_H

#endif  // MAT_EVAL_COMMON_H
