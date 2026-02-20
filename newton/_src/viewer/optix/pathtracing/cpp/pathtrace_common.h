/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef PATHTRACE_COMMON_H
#define PATHTRACE_COMMON_H

struct CompactMaterial {
    float3 baseColor;
    float3 emissive;
    float roughness;
    float metallic;
    float transmission;
    float ior;
    float3 specularColor;
    float specular;
    float clearcoat;
    float clearcoatRoughness;
    int clearcoatNormalTexIndex;
    int clearcoatNormalTexCoord;
    float opacity;
    int baseColorTexIndex;
    int baseColorTexCoord;
    int metallicRoughnessTexIndex;
    int metallicRoughnessTexCoord;
    int normalTexIndex;
    int normalTexCoord;
    int emissiveTexIndex;
    int emissiveTexCoord;
    float normalScale;
    float2x3 baseColorUvTransform;
    float2x3 metallicRoughnessUvTransform;
    float2x3 normalUvTransform;
    float2x3 emissiveUvTransform;
    float2x3 clearcoatNormalUvTransform;
};

static __forceinline__ __device__ float rand01(unsigned int& state)
{
    unsigned int r = pcg(state);
    return float(r & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

static __forceinline__ __device__ float3 sample_cosine_hemisphere(float3 n, unsigned int& rng)
{
    const float u1 = rand01(rng);
    const float u2 = rand01(rng);
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PI * u2;
    const float x = r * cosf(phi);
    const float y = r * sinf(phi);
    const float z = sqrtf(fmaxf(0.0f, 1.0f - u1));

    const float3 up = (fabsf(n.z) < 0.999f) ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(0.0f, 1.0f, 0.0f);
    const float3 t = normalize(cross(up, n));
    const float3 b = cross(n, t);
    return normalize(t * x + b * y + n * z);
}

static __forceinline__ __device__ float max3(float3 v) { return fmaxf(v.x, fmaxf(v.y, v.z)); }

static __forceinline__ __device__ float3 fresnel_schlick(float cosTheta, float3 F0)
{
    const float oneMinusCos = 1.0f - fminf(fmaxf(cosTheta, 0.0f), 1.0f);
    const float oneMinusCos5 = oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos;
    return F0 + (make_float3(1.0f, 1.0f, 1.0f) - F0) * oneMinusCos5;
}

static __forceinline__ __device__ float3 sample_glossy_reflection(float3 reflDir, float roughness, unsigned int& rng)
{
    const float3 glossyLobe = sample_cosine_hemisphere(reflDir, rng);
    const float blend = fminf(fmaxf(roughness * roughness, 0.0f), 1.0f);
    return normalize(reflDir * (1.0f - blend) + glossyLobe * blend);
}

static __forceinline__ __device__ float3 sample_uniform_sphere(unsigned int& rng)
{
    const float u1 = rand01(rng);
    const float u2 = rand01(rng);
    const float z = 1.0f - 2.0f * u1;
    const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    const float phi = 2.0f * M_PI * u2;
    return make_float3(r * cosf(phi), z, r * sinf(phi));
}

static __forceinline__ __device__ bool refract_dir(float3 i, float3 n, float eta, float3& out_dir)
{
    const float cosi = fminf(fmaxf(dot(-i, n), 0.0f), 1.0f);
    const float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
    if (k < 0.0f)
        return false;
    out_dir = normalize(eta * i + (eta * cosi - sqrtf(k)) * n);
    return true;
}

static __forceinline__ __device__ void lookup_material_from_payload(
    unsigned int materialId,
    float3 normal,
    float3 tangent,
    float2 uv,
    float3& outBaseColor,
    float3& outEmissive,
    float& outRoughness,
    float& outMetallic,
    float& outTransmission,
    float& outIor,
    float3& outSpecularColor,
    float& outSpecular,
    float& outClearcoat,
    float& outClearcoatRoughness,
    int& outClearcoatNormalTexIndex,
    int& outClearcoatNormalTexCoord,
    float& outOpacity,
    int& outBaseColorTexIndex,
    int& outBaseColorTexCoord,
    int& outMetallicRoughnessTexIndex,
    int& outMetallicRoughnessTexCoord,
    int& outNormalTexIndex,
    int& outNormalTexCoord,
    int& outEmissiveTexIndex,
    int& outEmissiveTexCoord,
    float& outNormalScale,
    float2x3& outBaseColorUvTransform,
    float2x3& outMetallicRoughnessUvTransform,
    float2x3& outNormalUvTransform,
    float2x3& outEmissiveUvTransform,
    float2x3& outClearcoatNormalUvTransform
)
{
    outBaseColor = make_float3(0.8f, 0.8f, 0.8f);
    outEmissive = make_float3(0.0f, 0.0f, 0.0f);
    outRoughness = 0.5f;
    outMetallic = 0.0f;
    outTransmission = 0.0f;
    outIor = 1.5f;
    outSpecularColor = make_float3(1.0f, 1.0f, 1.0f);
    outSpecular = 1.0f;
    outClearcoat = 0.0f;
    outClearcoatRoughness = 0.01f;
    outClearcoatNormalTexIndex = -1;
    outClearcoatNormalTexCoord = 0;
    outOpacity = 1.0f;
    outBaseColorTexIndex = -1;
    outBaseColorTexCoord = 0;
    outMetallicRoughnessTexIndex = -1;
    outMetallicRoughnessTexCoord = 0;
    outNormalTexIndex = -1;
    outNormalTexCoord = 0;
    outEmissiveTexIndex = -1;
    outEmissiveTexCoord = 0;
    outNormalScale = 1.0f;
    outBaseColorUvTransform = make_float2x3_identity();
    outMetallicRoughnessUvTransform = make_float2x3_identity();
    outNormalUvTransform = make_float2x3_identity();
    outEmissiveUvTransform = make_float2x3_identity();
    outClearcoatNormalUvTransform = make_float2x3_identity();

    if (params.compactMaterialAddress == 0ull)
        return;
    if (materialId >= params.materialCount)
        return;

    const CompactMaterial* materials = reinterpret_cast<const CompactMaterial*>(params.compactMaterialAddress);
    const CompactMaterial mat = materials[materialId];
    outBaseColor = mat.baseColor;
    outEmissive = mat.emissive;
    outRoughness = fminf(fmaxf(mat.roughness, 0.0f), 1.0f);
    outMetallic = fminf(fmaxf(mat.metallic, 0.0f), 1.0f);
    outTransmission = fminf(fmaxf(mat.transmission, 0.0f), 1.0f);
    outIor = fmaxf(mat.ior, 1.0f);
    outSpecularColor = mat.specularColor;
    outSpecular = fmaxf(mat.specular, 0.0f);
    outClearcoat = fmaxf(mat.clearcoat, 0.0f);
    outClearcoatRoughness = fmaxf(mat.clearcoatRoughness, 0.001f);
    outClearcoatNormalTexIndex = mat.clearcoatNormalTexIndex;
    outClearcoatNormalTexCoord = mat.clearcoatNormalTexCoord;
    outOpacity = fminf(fmaxf(mat.opacity, 0.0f), 1.0f);
    outBaseColorTexIndex = mat.baseColorTexIndex;
    outBaseColorTexCoord = mat.baseColorTexCoord;
    outMetallicRoughnessTexIndex = mat.metallicRoughnessTexIndex;
    outMetallicRoughnessTexCoord = mat.metallicRoughnessTexCoord;
    outNormalTexIndex = mat.normalTexIndex;
    outNormalTexCoord = mat.normalTexCoord;
    outEmissiveTexIndex = mat.emissiveTexIndex;
    outEmissiveTexCoord = mat.emissiveTexCoord;
    outNormalScale = fmaxf(mat.normalScale, 0.0f);
    outBaseColorUvTransform = mat.baseColorUvTransform;
    outMetallicRoughnessUvTransform = mat.metallicRoughnessUvTransform;
    outNormalUvTransform = mat.normalUvTransform;
    outEmissiveUvTransform = mat.emissiveUvTransform;
    outClearcoatNormalUvTransform = mat.clearcoatNormalUvTransform;
}

#endif  // PATHTRACE_COMMON_H
