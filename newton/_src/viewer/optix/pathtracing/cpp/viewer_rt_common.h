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

#ifndef VIEWER_RT_COMMON_H
#define VIEWER_RT_COMMON_H

#define ENV_MAP_FORMAT_RGBA32F 0u
#define ENV_MAP_FORMAT_RGBA16F 1u

// Common helpers shared by pathtracing viewer entry points.
// Keep this file free of entry kernels to avoid copy/paste growth in optix_programs.h.

static __forceinline__ __device__ float4x4 load_mat4_from_array(const float* m)
{
    float4x4 out;
    out[0] = make_float4(m[0], m[1], m[2], m[3]);
    out[1] = make_float4(m[4], m[5], m[6], m[7]);
    out[2] = make_float4(m[8], m[9], m[10], m[11]);
    out[3] = make_float4(m[12], m[13], m[14], m[15]);
    return out;
}

static __forceinline__ __device__ bool evaluate_pbr_from_payload(
    unsigned int materialId,
    float3 normal,
    float3 tangent,
    float bitangentSign,
    float2 uv,
    float2 uv1,
    PbrMaterial& outMat)
{
    if (materialId >= params.materialCount)
        return false;

    const float3 n = normalize(normal);
    float3 t = tangent - n * dot(n, tangent);
    if (dot(t, t) < 1.0e-12f)
    {
        // Robust fallback basis for normals near +/-X, +/-Y, +/-Z.
        const float3 up = (fabsf(n.z) < 0.999f) ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(0.0f, 1.0f, 0.0f);
        t = cross(up, n);
    }
    t = normalize(t);
    const float3 b = normalize(cross(n, t)) * ((bitangentSign == 0.0f) ? 1.0f : bitangentSign);

    // Use compact-material lookup path for robust ABI compatibility.
    // Direct reinterpret-cast of full GltfShadeMaterial can diverge due to
    // host/device packing differences and causes severe color corruption.
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
    lookup_material_from_payload(
        materialId,
        normal,
        tangent,
        uv,
        baseColor,
        emissive,
        roughness,
        metallic,
        transmission,
        ior,
        specularColor,
        specular,
        clearcoat,
        clearcoatRoughness,
        clearcoatNormalTexIndex,
        clearcoatNormalTexCoord,
        opacity,
        baseColorTexIndex,
        baseColorTexCoord,
        metallicRoughnessTexIndex,
        metallicRoughnessTexCoord,
        normalTexIndex,
        normalTexCoord,
        emissiveTexIndex,
        emissiveTexCoord,
        normalScale,
        baseColorUvTransform,
        metallicRoughnessUvTransform,
        normalUvTransform,
        emissiveUvTransform,
        clearcoatNormalUvTransform);
    outMat = defaultPbrMaterial(baseColor, metallic, roughness, n, n);
    outMat.T = t;
    outMat.B = b;
    outMat.emissive = emissive;
    outMat.opacity = opacity;
    outMat.transmission = transmission;
    outMat.ior1 = 1.0f;
    outMat.ior2 = fmaxf(ior, 1.0f);
    outMat.specular = specular;
    outMat.specularColor = specularColor;
    outMat.clearcoat = clearcoat;
    outMat.clearcoatRoughness = clearcoatRoughness;
    outMat.Nc = n;

    // Apply glTF textures through the packed texture buffers.
    const float2 tc[2] = {uv, uv1};
    GltfTextureInfo texInfo;
    texInfo.texCoord = 0;

    if (baseColorTexIndex >= 0)
    {
        texInfo.index = baseColorTexIndex;
        texInfo.texCoord = (baseColorTexCoord == 1) ? 1 : 0;
        texInfo.uvTransform = baseColorUvTransform;
        const float4 c = getTexture(texInfo, tc, nullptr);
        outMat.baseColor = outMat.baseColor * make_float3(c.x, c.y, c.z);
        outMat.opacity = outMat.opacity * c.w;
    }
    if (metallicRoughnessTexIndex >= 0)
    {
        texInfo.index = metallicRoughnessTexIndex;
        texInfo.texCoord = (metallicRoughnessTexCoord == 1) ? 1 : 0;
        texInfo.uvTransform = metallicRoughnessUvTransform;
        const float4 mr = getTexture(texInfo, tc, nullptr);
        const float r = fmaxf(sqrtf(fmaxf(outMat.roughness.x, 0.0f)) * mr.y, MICROFACET_MIN_ROUGHNESS);
        outMat.roughness = make_float2(r * r, r * r);
        outMat.metallic = fminf(fmaxf(outMat.metallic * mr.z, 0.0f), 1.0f);
    }
    bool needsTangentUpdate = false;
    if (normalTexIndex >= 0)
    {
        texInfo.index = normalTexIndex;
        texInfo.texCoord = (normalTexCoord == 1) ? 1 : 0;
        texInfo.uvTransform = normalUvTransform;
        const float4 nmap = getTexture(texInfo, tc, nullptr);
        float3 ntex = make_float3(nmap.x, nmap.y, nmap.z) * 2.0f - make_float3(1.0f, 1.0f, 1.0f);
        ntex = ntex * make_float3(normalScale, normalScale, 1.0f);
        outMat.N = normalize(outMat.T * ntex.x + outMat.B * ntex.y + outMat.N * ntex.z);
        outMat.Ng = outMat.N;
        outMat.Nc = outMat.N;
        needsTangentUpdate = true;
    }
    if (clearcoatNormalTexIndex >= 0)
    {
        texInfo.index = clearcoatNormalTexIndex;
        texInfo.texCoord = (clearcoatNormalTexCoord == 1) ? 1 : 0;
        texInfo.uvTransform = clearcoatNormalUvTransform;
        const float4 nmap = getTexture(texInfo, tc, nullptr);
        const float3 ntex = make_float3(nmap.x, nmap.y, nmap.z) * 2.0f - make_float3(1.0f, 1.0f, 1.0f);
        outMat.Nc = normalize(outMat.T * ntex.x + outMat.B * ntex.y + outMat.Nc * ntex.z);
    }
    if (needsTangentUpdate)
    {
        outMat.B = cross(outMat.N, outMat.T);
        const float bitangentSign = (dot(b, outMat.B) < 0.0f) ? -1.0f : 1.0f;
        outMat.B = outMat.B * bitangentSign;
        outMat.T = cross(outMat.B, outMat.N) * bitangentSign;
    }
    if (emissiveTexIndex >= 0)
    {
        texInfo.index = emissiveTexIndex;
        texInfo.texCoord = (emissiveTexCoord == 1) ? 1 : 0;
        texInfo.uvTransform = emissiveUvTransform;
        const float4 e = getTexture(texInfo, tc, nullptr);
        outMat.emissive = outMat.emissive * make_float3(e.x, e.y, e.z);
    }
    return true;
}

static __forceinline__ __device__ PhysicalSkyParameters sky_params_from_launch()
{
    PhysicalSkyParameters sky;
    sky.rgbUnitConversion =
        make_float3(params.skyInfo.rgbUnitConversion[0], params.skyInfo.rgbUnitConversion[1], params.skyInfo.rgbUnitConversion[2]);
    sky.multiplier = params.skyInfo.multiplier;
    sky.haze = params.skyInfo.haze;
    sky.redblueshift = params.skyInfo.redblueshift;
    sky.saturation = params.skyInfo.saturation;
    sky.horizonHeight = params.skyInfo.horizonHeight;
    sky.groundColor = make_float3(params.skyInfo.groundColor[0], params.skyInfo.groundColor[1], params.skyInfo.groundColor[2]);
    sky.horizonBlur = params.skyInfo.horizonBlur;
    sky.nightColor = make_float3(params.skyInfo.nightColor[0], params.skyInfo.nightColor[1], params.skyInfo.nightColor[2]);
    sky.sunDiskIntensity = params.skyInfo.sunDiskIntensity;
    sky.sunDirection =
        normalize(make_float3(params.skyInfo.sunDirection[0], params.skyInfo.sunDirection[1], params.skyInfo.sunDirection[2]));
    sky.sunDiskScale = params.skyInfo.sunDiskScale;
    sky.sunGlowIntensity = params.skyInfo.sunGlowIntensity;
    sky.yIsUp = params.skyInfo.yIsUp;
    return sky;
}

static __forceinline__ __device__ float3 rotate_environment_dir(float3 dir, float angle, int yIsUp)
{
    const float s = sinf(angle);
    const float c = cosf(angle);

    if (yIsUp != 0)
    {
        // Y-up: rotate in XZ plane.
        return make_float3(c * dir.x + s * dir.z, dir.y, -s * dir.x + c * dir.z);
    }

    // Z-up: rotate in XY plane.
    return make_float3(c * dir.x - s * dir.y, s * dir.x + c * dir.y, dir.z);
}

static __forceinline__ __device__ float env_half_to_float(unsigned short bits)
{
    unsigned int sign = ((unsigned int)(bits & 0x8000u)) << 16;
    int exp = (int)((bits >> 10) & 0x1Fu);
    unsigned int mant = (unsigned int)(bits & 0x03FFu);

    if (exp == 0)
    {
        if (mant == 0u)
            return __uint_as_float(sign);
        while ((mant & 0x0400u) == 0u)
        {
            mant <<= 1;
            exp -= 1;
        }
        exp += 1;
        mant &= 0x03FFu;
    }
    else if (exp == 31)
    {
        return __uint_as_float(sign | 0x7F800000u | (mant << 13));
    }

    unsigned int outExp = (unsigned int)(exp + (127 - 15));
    unsigned int outBits = sign | (outExp << 23) | (mant << 13);
    return __uint_as_float(outBits);
}

static __forceinline__ __device__ float4 load_env_texel_at(unsigned int x, unsigned int y)
{
    const unsigned int w = params.envMapWidth;
    const unsigned int idx = y * w + x;
    if (params.envMapFormat == ENV_MAP_FORMAT_RGBA16F)
    {
        const unsigned short* texels16 = reinterpret_cast<const unsigned short*>(params.envMapAddress);
        const unsigned short* h4 = texels16 + idx * 4u;
        return make_float4(
            env_half_to_float(h4[0]),
            env_half_to_float(h4[1]),
            env_half_to_float(h4[2]),
            env_half_to_float(h4[3]));
    }
    const float4* texels32 = reinterpret_cast<const float4*>(params.envMapAddress);
    return texels32[idx];
}

// Spherical UV mapping - matches C# EnvMap.getSphericalUv()
// Maps 3D direction to 2D texture coordinates for lat-long environment maps.
// Y-up convention: Y is vertical axis (zenith/nadir)
static __forceinline__ __device__ float2 get_spherical_uv_csharp(float3 v)
{
    const float gamma = asinf(-v.y);
    const float theta = atan2f(v.z, v.x);
    return make_float2(theta * M_1_OVER_PI * 0.5f + 0.5f, gamma * M_1_OVER_PI + 0.5f);
}

// Bilinear texture sample with wrapping in U, clamping in V
static __forceinline__ __device__ float4 sample_envmap_uv(float u, float v)
{
    const unsigned int w = params.envMapWidth;
    const unsigned int h = params.envMapHeight;
    const float fx = u * float(w) - 0.5f;
    const float fy = v * float(h) - 0.5f;
    const int x0 = (int)floorf(fx);
    const int y0 = (int)floorf(fy);
    const float tx = fx - floorf(fx);
    const float ty = fy - floorf(fy);
    const unsigned int ix0 = (unsigned int)((x0 % (int)w + (int)w) % (int)w);
    const unsigned int iy0 = (unsigned int)min(max(y0, 0), (int)h - 1);
    const unsigned int ix1 = (ix0 + 1u) % w;
    const unsigned int iy1 = min(iy0 + 1u, h - 1u);
    const float4 c00 = load_env_texel_at(ix0, iy0);
    const float4 c10 = load_env_texel_at(ix1, iy0);
    const float4 c01 = load_env_texel_at(ix0, iy1);
    const float4 c11 = load_env_texel_at(ix1, iy1);
    const float4 c0 = c00 * (1.0f - tx) + c10 * tx;
    const float4 c1 = c01 * (1.0f - tx) + c11 * tx;
    return c0 * (1.0f - ty) + c1 * ty;
}

// Evaluate environment map for a given world direction.
// Matches GLSL primary.rmiss: procedural sky uses the direction unrotated;
// envRotation only applies to HDR envmap lookups.
static __forceinline__ __device__ float3 eval_environment(float3 worldDir)
{
    // If using procedural physical sky model
    if (TEST_FLAG(params.frameInfo.flags, FLAGS_ENVMAP_SKY))
    {
        const PhysicalSkyParameters sky = sky_params_from_launch();
        const float3 env = evalPhysicalSky(sky, normalize(worldDir));
        return env * make_float3(
            params.frameInfo.envIntensity[0],
            params.frameInfo.envIntensity[1],
            params.frameInfo.envIntensity[2]);
    }

    // If HDR environment map is loaded
    if (params.envMapAddress != 0ull && params.envMapWidth > 0u && params.envMapHeight > 0u)
    {
        const float3 d = rotate_environment_dir(normalize(worldDir), -params.frameInfo.envRotation, 1);
        const float2 uv = get_spherical_uv_csharp(d);
        const float u = uv.x - floorf(uv.x);
        // Match C# HdrEnvironment/StbImage orientation: V=0 at top row.
        const float v = fminf(fmaxf(uv.y, 0.0f), 1.0f);
        const float4 c = sample_envmap_uv(u, v);
        return make_float3(c.x, c.y, c.z) * make_float3(
            params.frameInfo.envIntensity[0],
            params.frameInfo.envIntensity[1],
            params.frameInfo.envIntensity[2]);
    }

    // Fallback gradient environment
    return make_float3(0.2f, 0.3f, 0.5f);
}

// Environment importance sampling using the alias method
// Directly ported from C# MiniOptixScene/EnvMap.cs environmentSample()
// Reference: https://arxiv.org/pdf/1901.05423.pdf, section 2.6, "The Alias Method"
static __forceinline__ __device__ bool sample_environment_importance(
    unsigned int& rng,
    float3& outDir,
    float3& outRadiance,
    float& outPdf)
{
    // Check if environment map and acceleration structure are available
    if (params.envMapAddress == 0ull || params.envMapWidth == 0u || params.envMapHeight == 0u ||
        params.envAccelAddress == 0ull)
    {
        return false;
    }

    const int width = (int)params.envMapWidth;
    const int height = (int)params.envMapHeight;
    const int size = width * height;

    // Generate 3 random values for sampling
    const float xi_x = rand01(rng);
    const float xi_y = rand01(rng);
    const float xi_z = rand01(rng);

    // Uniformly pick a texel index idx in the environment map
    int idx = min((int)(xi_x * float(size)), size - 1);

    // Fetch the sampling data for that texel, containing the ratio q between its
    // emitted radiance and the average of the environment map, and the texel alias
    // EnvAccel struct: { uint alias; float q; } - only 8 bytes!
    const EnvAccel* accel = reinterpret_cast<const EnvAccel*>(params.envAccelAddress);
    EnvAccel sample_data = accel[idx];

    int env_idx;
    float local_xi_y = xi_y;

    if (xi_y < sample_data.q)
    {
        // If the random variable is lower than the intensity ratio q, we directly pick
        // this texel, and renormalize the random variable for later use
        env_idx = idx;
        local_xi_y = xi_y / fmaxf(sample_data.q, 1.0e-8f);
    }
    else
    {
        // Otherwise we pick the alias of the texel, renormalize the random variable
        env_idx = (int)sample_data.alias;
        local_xi_y = (xi_y - sample_data.q) / fmaxf(1.0f - sample_data.q, 1.0e-8f);
    }

    // Compute the 2D integer coordinates of the texel
    const int px = env_idx % width;
    const int py = env_idx / width;

    // Uniformly sample the solid angle subtended by the pixel.
    // Generate both the UV for texture lookup and a direction in spherical coordinates
    const float u = (float(px) + local_xi_y) / float(width);
    const float phi = u * (2.0f * M_PI) - M_PI;
    const float sin_phi = sinf(phi);
    const float cos_phi = cosf(phi);

    const float step_theta = M_PI / float(height);
    const float theta0 = float(py) * step_theta;
    const float cos_theta = cosf(theta0) * (1.0f - xi_z) + cosf(theta0 + step_theta) * xi_z;
    const float theta = acosf(fminf(fmaxf(cos_theta, -1.0f), 1.0f));
    const float sin_theta = sinf(theta);
    const float v = theta * M_1_OVER_PI;

    // Convert to a light direction vector in Cartesian coordinates
    // Y-up convention: Y is vertical (cos_theta), X and Z are horizontal
    const float3 localDir = make_float3(cos_phi * sin_theta, cos_theta, sin_phi * sin_theta);

    // Apply environment rotation (negative to match eval_environment)
    outDir = rotate_environment_dir(localDir, -params.frameInfo.envRotation, 1);

    // Lookup the environment value using bilinear filtering
    // The alpha channel contains max(R,G,B) / integral (the PDF)
    const float4 c = sample_envmap_uv(u, v);
    outRadiance = make_float3(c.x, c.y, c.z) * make_float3(
        params.frameInfo.envIntensity[0],
        params.frameInfo.envIntensity[1],
        params.frameInfo.envIntensity[2]);

    // PDF is stored directly in alpha channel - matches C# exactly
    // C# MiniPathTracer: lightPdf = sampleResult.W;
    // No Jacobian needed - the alias method samples proportional to importance
    // which already accounts for solid angle weighting
    outPdf = fmaxf(c.w, 1.0e-8f);

    return true;
}

#endif  // VIEWER_RT_COMMON_H
